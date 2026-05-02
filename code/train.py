import os
import sys
import math
import random
import argparse
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from lora import LoRA
from dora2 import DoRAPaper
from dora3 import DoRA3

import time


# ============================================================
# Utils
# ============================================================

_CAUSAL_PREFIXES = ("llama", "mistral", "opt", "gpt2", "gpt-", "phi", "falcon", "gemma")

def detect_model_type(model_name: str) -> str:
    name = model_name.lower()
    return "causal" if any(p in name for p in _CAUSAL_PREFIXES) else "encoder"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # model / save
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--model_type", type=str, choices=["encoder", "causal", "auto"], default="auto")
    parser.add_argument("--method", type=str, choices=["lora", "dora3", "dora2", "both"], default="dora2")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=100)

    # data / training
    parser.add_argument("--dataset", type=str, choices=["boolq", "piqa", "siqa", "hellaswag", "arc_challenge", "arc_easy", "winogrande", "openbookqa", "all"], default="boolq")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)

    # adapter
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    # layer targeting
    parser.add_argument(
        "--target_modules",
        nargs="*",
        default=None,
        help=(
            "Optional list of substrings. Only nn.Linear layers whose full name "
            "contains one of these substrings will be adapted. "
            "Example: query key value dense"
        ),
    )

    # misc
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every_epoch", action="store_true")

    return parser.parse_args()


# ============================================================
# Adapter-wrapped Linear
# ============================================================

class AdaptedLinear(nn.Module):
    """
    Wrap a frozen nn.Linear with either LoRA or DoRA.
    The pretrained weight is stored inside the adapter.
    The original bias is kept frozen.
    """
    def __init__(
        self,
        linear_layer: nn.Linear,
        method: str,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not isinstance(linear_layer, nn.Linear):
            raise TypeError("AdaptedLinear expects an nn.Linear layer")

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        W = linear_layer.weight.detach().clone()

        if linear_layer.bias is not None:
            self.bias = nn.Parameter(
                linear_layer.bias.detach().clone(),
                requires_grad=False,
            )
        else:
            self.bias = None

        if method == "lora":
            self.adapter = LoRA(
                W=W,
                rank=rank,
                p=dropout,
                alpha=alpha,
            )
        elif method == "dora3":
            self.adapter = DoRA3(
                W=W,
                rank=rank,
                p=dropout,
                alpha=alpha
            )
        elif method == "dora2":
            self.adapter = DoRAPaper(
                W=W,
                rank=rank,
                p=dropout,
                alpha=alpha
            )
        else:
            raise ValueError("method must be 'lora' or 'dora3' or 'dora2'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.adapter(x)
        if self.bias is not None:
            out = out + self.bias
        return out


def should_replace_layer(full_name: str, target_modules: Optional[List[str]]) -> bool:
    if target_modules is None or len(target_modules) == 0:
        return True
    return any(substr in full_name for substr in target_modules)


def replace_linear_layers_with_adapters(
    module: nn.Module,
    method: str,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    prefix: str = "",
) -> None:
    """
    Recursively replace selected nn.Linear layers with AdaptedLinear.
    """
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear) and should_replace_layer(full_name, target_modules):
            setattr(
                module,
                name,
                AdaptedLinear(
                    linear_layer=child,
                    method=method,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                ),
            )
        else:
            replace_linear_layers_with_adapters(
                child,
                method=method,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
                prefix=full_name,
            )


def freeze_non_adapter_params(model: nn.Module) -> None:
    """
    Freeze everything, then unfreeze only adapter parameters.
    """
    for p in model.parameters():
        p.requires_grad = False

    for module in model.modules():
        if isinstance(module, AdaptedLinear):
            for p in module.adapter.parameters():
                p.requires_grad = True


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# ============================================================
# Dataset formatting
# ============================================================

class MultiChoiceDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def load_boolq_examples(split: str) -> List[Dict[str, Any]]:
    cache_file = f"boolq_{split}_examples.pt"

    if os.path.exists(cache_file):
        print(f"Loading BoolQ examples from {cache_file}...")
        return torch.load(cache_file)

    print("Loading BoolQ from Hugging Face...")
    ds = load_dataset("boolq", split=split)
    print(f"Loaded raw BoolQ split with {len(ds)} examples")

    examples = []
    for i, ex in enumerate(ds):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(ds)} BoolQ examples...")
        examples.append(
            {
                "dataset": "boolq",
                "prompt": f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:",
                "choices": ["no", "yes"],
                "label": int(ex["answer"]),
            }
        )

    torch.save(examples, cache_file)
    print(f"Saved BoolQ examples to {cache_file}")
    return examples



def load_piqa_examples(split: str) -> List[Dict[str, Any]]:
    ds = load_dataset("piqa", split=split, trust_remote_code=True)
    examples = []

    for ex in ds:
        examples.append(
            {
                "dataset": "piqa",
                "prompt": f"Question: {ex['goal']}\nAnswer:",
                "choices": [ex["sol1"], ex["sol2"]],
                "label": int(ex["label"]),
            }
        )

    return examples


def load_siqa_examples(split: str) -> List[Dict[str, Any]]:
    ds = load_dataset("social_i_qa", split=split, trust_remote_code=True)
    examples = []

    for ex in ds:
        examples.append(
            {
                "dataset": "siqa",
                "prompt": f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer:",
                "choices": [ex["answerA"], ex["answerB"], ex["answerC"]],
                "label": int(ex["label"]) - 1,
            }
        )

    return examples


def load_hellaswag_examples(split: str) -> List[Dict[str, Any]]:
    ds = load_dataset("hellaswag", split=split, trust_remote_code=True)
    examples = []

    for ex in ds:
        examples.append(
            {
                "dataset": "hellaswag",
                "prompt": ex["ctx"],
                "choices": ex["endings"],
                "label": int(ex["label"]),
            }
        )

    return examples


def load_all_examples(split: str) -> List[Dict[str, Any]]:
    print("Loading all 8 commonsense datasets...")
    _val = "validation"
    examples = []
    examples += load_boolq_examples(split)
    examples += load_piqa_examples(split if split != "test" else _val)
    examples += load_siqa_examples(split if split != "test" else _val)
    examples += load_hellaswag_examples(split if split != "test" else _val)
    examples += load_winogrande_examples(split if split != "test" else _val)
    examples += load_arc_examples(split, "ARC-Challenge")
    examples += load_arc_examples(split, "ARC-Easy")
    examples += load_openbookqa_examples(split)
    print(f"Total combined examples: {len(examples)}")
    return examples


def load_arc_examples(split: str, config: str = "ARC-Challenge") -> List[Dict[str, Any]]:
    ds = load_dataset("ai2_arc", config, split=split)
    examples = []

    for ex in ds:
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        answer_key = ex["answerKey"]

        if answer_key in labels:
            label = labels.index(answer_key)
        else:
            label = int(answer_key) - 1

        examples.append(
            {
                "dataset": config.lower().replace("-", "_"),
                "prompt": f"Question: {ex['question']}\nAnswer:",
                "choices": choices,
                "label": label,
            }
        )

    return examples


def load_winogrande_examples(split: str) -> List[Dict[str, Any]]:
    ds = load_dataset("winogrande", "winogrande_xl", split=split)
    examples = []

    for ex in ds:
        examples.append(
            {
                "dataset": "winogrande",
                "prompt": ex["sentence"],
                "choices": [ex["option1"], ex["option2"]],
                "label": int(ex["answer"]) - 1,
            }
        )

    return examples


def load_openbookqa_examples(split: str) -> List[Dict[str, Any]]:
    ds = load_dataset("openbookqa", "main", split=split)
    examples = []

    for ex in ds:
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        answer_key = ex["answerKey"]

        if answer_key in labels:
            label = labels.index(answer_key)
        else:
            label = int(answer_key) - 1

        examples.append(
            {
                "dataset": "openbookqa",
                "prompt": f"Question: {ex['question_stem']}\nAnswer:",
                "choices": choices,
                "label": label,
            }
        )

    return examples


# ============================================================
# Collation
# ============================================================

@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    num_choices: List[int]


def collate_fn(examples: List[Dict[str, Any]], tokenizer, max_length: int) -> Batch:
    text_a = []
    text_b = []
    labels = []
    num_choices = []

    for ex in examples:
        prompt = ex["prompt"]
        choices = ex["choices"]

        labels.append(ex["label"])
        num_choices.append(len(choices))

        for choice in choices:
            text_a.append(prompt)
            text_b.append(choice)

    enc = tokenizer(
        text_a,
        text_b,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return Batch(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        labels=torch.tensor(labels, dtype=torch.long),
        num_choices=num_choices,
    )


@dataclass
class CausalBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label_ids: torch.Tensor
    labels: torch.Tensor
    num_choices: List[int]


def collate_fn_causal(examples: List[Dict[str, Any]], tokenizer, max_length: int) -> CausalBatch:
    all_input_ids = []
    all_label_ids = []
    labels = []
    num_choices = []

    for ex in examples:
        prompt = ex["prompt"]
        choices = ex["choices"]
        labels.append(ex["label"])
        num_choices.append(len(choices))

        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)

        for choice in choices:
            choice_ids = tokenizer(" " + choice, add_special_tokens=False)["input_ids"]
            full_ids = (prompt_ids + choice_ids)[:max_length]
            lbl_ids = ([-100] * prompt_len + choice_ids)[:max_length]
            all_input_ids.append(torch.tensor(full_ids, dtype=torch.long))
            all_label_ids.append(torch.tensor(lbl_ids, dtype=torch.long))

    max_len = max(ids.shape[0] for ids in all_input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_input, padded_mask, padded_lbl = [], [], []
    for inp, lbl in zip(all_input_ids, all_label_ids):
        pad_len = max_len - inp.shape[0]
        padded_input.append(torch.cat([inp, torch.full((pad_len,), pad_id)]))
        padded_mask.append(torch.cat([torch.ones(inp.shape[0], dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
        padded_lbl.append(torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)]))

    return CausalBatch(
        input_ids=torch.stack(padded_input),
        attention_mask=torch.stack(padded_mask),
        label_ids=torch.stack(padded_lbl),
        labels=torch.tensor(labels, dtype=torch.long),
        num_choices=num_choices,
    )


# ============================================================
# Loss over variable number of choices
# ============================================================

def grouped_choice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_choices: List[int],
):
    """
    logits: (total_choices, 1)
    labels: (batch,)
    num_choices: length batch, each entry is 2 or 4 here
    """
    logits = logits.squeeze(-1)

    grouped = []
    idx = 0
    for n in num_choices:
        grouped.append(logits[idx: idx + n])
        idx += n

    max_choices = max(num_choices)
    padded = []

    for g in grouped:
        if g.numel() < max_choices:
            pad = torch.full(
                (max_choices - g.numel(),),
                fill_value=-1e9,
                device=g.device,
                dtype=g.dtype,
            )
            g = torch.cat([g, pad], dim=0)
        padded.append(g)

    choice_logits = torch.stack(padded, dim=0)  # (batch, max_choices)
    loss = nn.CrossEntropyLoss()(choice_logits, labels)
    return loss, choice_logits


def causal_choice_loss(
    logits: torch.Tensor,
    label_ids: torch.Tensor,
    labels: torch.Tensor,
    num_choices: List[int],
):
    """
    logits:    (total_choices, seq_len, vocab_size)
    label_ids: (total_choices, seq_len) — -100 for prompt tokens
    labels:    (batch,)
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = label_ids[:, 1:]

    mask = shift_labels != -100
    gather_labels = shift_labels.clone()
    gather_labels[~mask] = 0

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, gather_labels.unsqueeze(2)).squeeze(2)
    choice_scores = (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    grouped, idx = [], 0
    for n in num_choices:
        grouped.append(choice_scores[idx: idx + n])
        idx += n

    max_choices = max(num_choices)
    padded = []
    for g in grouped:
        if g.numel() < max_choices:
            pad = torch.full((max_choices - g.numel(),), -1e9, device=g.device, dtype=g.dtype)
            g = torch.cat([g, pad])
        padded.append(g)

    choice_logits = torch.stack(padded)
    loss = nn.CrossEntropyLoss()(choice_logits, labels)
    return loss, choice_logits


# ============================================================
# Training
# ============================================================

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    args,
):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    args,
    model_type: str = "encoder",
):
    model.train()
    running_loss = 0.0
    epoch_loss_sum = 0.0

    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        if model_type == "causal":
            label_ids = batch.label_ids.to(device)
            loss, _ = causal_choice_loss(outputs.logits, label_ids, labels, batch.num_choices)
        else:
            loss, _ = grouped_choice_loss(outputs.logits, labels, batch.num_choices)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        epoch_loss_sum += loss.item()

        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            print(f"epoch {epoch} | step {step}/{len(train_loader)} | loss {avg_loss:.4f}")
            running_loss = 0.0

    print(f"Epoch {epoch} | Average Loss: {epoch_loss_sum / len(train_loader):.4f}")
    return epoch_loss_sum / len(train_loader)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.method == "both":
        for method in ["lora", "dora2"]:
            cmd = [sys.executable] + sys.argv[:]
            idx = cmd.index("--method")
            cmd[idx + 1] = method
            print(f"\n{'='*60}\nStarting {method.upper()} training\n{'='*60}")
            subprocess.run(cmd, check=True)
        return

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model_type = args.model_type if args.model_type != "auto" else detect_model_type(args.model_name)
    print(f"Model type: {model_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=1,
        )

    replace_linear_layers_with_adapters(
        module=model,
        method=args.method,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=args.target_modules,
    )

    freeze_non_adapter_params(model)
    model.to(device)

    trainable, total = count_parameters(model)
    pct = 100.0 * trainable / total
    print(f"Method: {args.method.upper()} | Rank: {args.rank}")
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")

    # --------------------
    # Load datasets
    # --------------------
    print(f"Loading {args.dataset}...")
    if args.dataset == "boolq":
        train_dataset = MultiChoiceDataset(load_boolq_examples("train"))
    elif args.dataset == "piqa":
        train_dataset = MultiChoiceDataset(load_piqa_examples("train"))
    elif args.dataset == "siqa":
        train_dataset = MultiChoiceDataset(load_siqa_examples("train"))
    elif args.dataset == "hellaswag":
        train_dataset = MultiChoiceDataset(load_hellaswag_examples("train"))
    elif args.dataset == "arc_challenge":
        train_dataset = MultiChoiceDataset(load_arc_examples("train", "ARC-Challenge"))
    elif args.dataset == "arc_easy":
        train_dataset = MultiChoiceDataset(load_arc_examples("train", "ARC-Easy"))
    elif args.dataset == "winogrande":
        train_dataset = MultiChoiceDataset(load_winogrande_examples("train"))
    elif args.dataset == "openbookqa":
        train_dataset = MultiChoiceDataset(load_openbookqa_examples("train"))
    elif args.dataset == "all":
        train_dataset = MultiChoiceDataset(load_all_examples("train"))

    if model_type == "causal":
        _collate = lambda batch: collate_fn_causal(batch, tokenizer, args.max_length)
    else:
        _collate = lambda batch: collate_fn(batch, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    start_epoch = 1
    global_step = 0

    if args.resume_from_checkpoint is not None:
        ckpt = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")

    # --------------------
    # Train
    # --------------------
    total_train_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()

        epoch_avg_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            args=args,
            model_type=model_type,
        )

        epoch_time_sec = time.time() - epoch_start_time
        global_step += len(train_loader)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n=== Epoch {epoch} Summary ===")
        print(f"Average loss: {epoch_avg_loss:.4f}")
        print(f"Epoch time: {epoch_time_sec:.2f} sec ({epoch_time_sec / 60:.2f} min)")
        print(f"Learning rate: {current_lr:.8f}")
        print(f"Global steps completed: {global_step}\n")

        if args.save_every_epoch:
            epoch_ckpt = os.path.join(args.save_dir, f"{args.method}_epoch_{epoch}.pt")
            save_checkpoint(
                path=epoch_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                args=args,
            )
            print(f"Saved checkpoint to {epoch_ckpt}")

    total_train_time_sec = time.time() - total_train_start
    print(f"\nTotal training time: {total_train_time_sec:.2f} sec ({total_train_time_sec / 60:.2f} min)")

    final_ckpt = os.path.join(args.save_dir, f"{args.method}_{args.dataset}_final.pt")
    save_checkpoint(
        path=final_ckpt,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        step=global_step,
        args=args,
    )
    print(f"Saved final checkpoint to {final_ckpt}")

if __name__ == "__main__":
    main()