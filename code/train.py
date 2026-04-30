import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from lora import LoRA
from dora import DoRA
from dora2 import DoRAPaper

import time


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # model / save
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--method", type=str, choices=["lora", "dora", "dora2"], default="dora")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # data / training
    parser.add_argument("--dataset", type=str, choices=["boolq", "hellaswag", "piqa"], default="boolq")
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
        elif method == "dora":
            self.adapter = DoRA(
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
            raise ValueError("method must be 'lora' or 'dora' or 'dora2'")

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
    ds = load_dataset("piqa", split=split)
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


def load_hellaswag_examples(split: str) -> List[Dict[str, Any]]:
    ds = load_dataset("hellaswag", split=split)
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
):
    model.train()
    running_loss = 0.0
    epoch_loss_sum = 0.0

    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

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
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # RoBERTa already has a pad token, but this check keeps the code safer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # num_labels=1 because we score each (prompt, choice) pair with one scalar logit
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
    print(f"Trainable params: {trainable} / {total} ({pct:.4f}%)")

    # --------------------
    # Load datasets
    # --------------------
    print(f"Loading {args.dataset}...")
    if args.dataset == "boolq":
        train_dataset = MultiChoiceDataset(load_boolq_examples("train"))
    elif args.dataset == "hellaswag":
        train_dataset = MultiChoiceDataset(load_hellaswag_examples("train"))
    elif args.dataset == "piqa":
        train_dataset = MultiChoiceDataset(load_piqa_examples("train"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length),
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

if __name__ == "__main__":
    main()