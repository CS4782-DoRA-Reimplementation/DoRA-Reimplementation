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
from dora_ablation import DoRAFullAblation, DoRAFrozenMagnitude, DoRAMagnitudeOnly

import time


# ============================================================
# Utils
# ============================================================

_CAUSAL_PREFIXES = ("llama", "mistral", "opt", "gpt2", "gpt-", "phi", "falcon", "gemma", "qwen", "smollm")

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
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "lora",
            "dora3",
            "dora2",
            "dora_full",
            "dora_frozen_magnitude",
            "dora_magnitude_only",
            "both",
            "ablation",
        ],
        default="dora3",
        help=(
            "Adapter method. Use 'dora3' for the standard DoRA implementation. "
            "Ablation variants: 'dora_full' (train magnitude + direction), "
            "'dora_frozen_magnitude' (direction-only), "
            "'dora_magnitude_only' (magnitude-only). "
            "'ablation' runs all three ablation variants sequentially."
        ),
    )
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=100)

    # data / training
    parser.add_argument("--dataset", type=str, choices=["boolq", "piqa", "siqa", "hellaswag", "arc_challenge", "arc_easy", "winogrande", "openbookqa", "all"], default="boolq")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs without val loss improvement)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)

    # adapter — paper values for commonsense reasoning
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--dropout", type=float, default=0.05)

    # quantization
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit (requires bitsandbytes, for LLaMA)")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory at cost of speed")

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
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")

    return parser.parse_args()


# ============================================================
# Adapter-wrapped Linear
# ============================================================

class AdaptedLinear(nn.Module):
    """
    Adds a LoRA or DoRA adapter on top of a frozen nn.Linear.

    LoRA:  output = base(x) + (alpha/rank) * (dropout(x) @ A.T) @ B.T
           Purely additive — W is never modified.

    DoRA3: output = m * (W0 + delta) / ||W0 + delta|| @ x.T + bias
           Magnitude/direction decomposition per the DoRA paper.
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

        self.method = method
        self._adapter_device_set = False
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # Keep original frozen linear — base(x) = W @ x + b
        for p in linear_layer.parameters():
            p.requires_grad = False
        self.base = linear_layer

        W = _get_weight(linear_layer)

        if method == "lora":
            # Additive low-rank delta: (alpha/rank) * B @ A
            self.scale = alpha / rank
            self.B = nn.Parameter(torch.zeros(self.out_features, rank))
            self.A = nn.Parameter(torch.empty(rank, self.in_features))
            nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
            self.dropout = nn.Dropout(dropout)
        elif method == "dora3":
            self.adapter = DoRA3(W=W, rank=rank, p=dropout, alpha=alpha)
        elif method == "dora2":
            self.adapter = DoRAPaper(W=W, rank=rank, p=dropout, alpha=alpha)
        elif method == "dora_full":
            self.adapter = DoRAFullAblation(W=W, rank=rank, p=dropout, alpha=alpha)
        elif method == "dora_frozen_magnitude":
            self.adapter = DoRAFrozenMagnitude(W=W, rank=rank, p=dropout, alpha=alpha)
        elif method == "dora_magnitude_only":
            self.adapter = DoRAMagnitudeOnly(W=W, rank=rank, p=dropout, alpha=alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "lora":
            A = self.A.to(device=x.device, dtype=x.dtype)
            B = self.B.to(device=x.device, dtype=x.dtype)
            delta = self.scale * (self.dropout(x) @ A.T) @ B.T
            return self.base(x) + delta
        else:
            if not self._adapter_device_set:
                self.adapter.to(device=x.device)
                self._adapter_device_set = True
            out = self.adapter(x)
            if self.base.bias is not None:
                out = out + self.base.bias.to(device=x.device, dtype=out.dtype)
            return out


def should_replace_layer(full_name: str, target_modules: Optional[List[str]]) -> bool:
    if target_modules is None or len(target_modules) == 0:
        return True
    return any(substr in full_name for substr in target_modules)


def _is_linear(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    try:
        import bitsandbytes as bnb
        return isinstance(module, bnb.nn.Linear4bit)
    except ImportError:
        return False


def _get_weight(linear_layer: nn.Module) -> torch.Tensor:
    try:
        import bitsandbytes as bnb
        if isinstance(linear_layer, bnb.nn.Linear4bit):
            return bnb.functional.dequantize_4bit(
                linear_layer.weight.data,
                linear_layer.weight.quant_state,
            ).to(torch.float16).detach()
    except ImportError:
        pass
    return linear_layer.weight.detach()


def replace_linear_layers_with_adapters(
    module: nn.Module,
    method: str,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    prefix: str = "",
) -> None:
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if _is_linear(child) and should_replace_layer(full_name, target_modules):
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
    # Freeze all base model weights
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze adapter parameters
    for module in model.modules():
        if isinstance(module, AdaptedLinear):
            if module.method == "lora":
                module.A.requires_grad = True
                module.B.requires_grad = True
            else:
                for p in module.adapter.parameters():
                    p.requires_grad = True

    # Unfreeze task head for encoder models only (lm_head is too large for causal LMs)
    _HEAD_KEYS = ("classifier", "pooler", "score")
    for name, p in model.named_parameters():
        if any(k in name for k in _HEAD_KEYS):
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
    import random as _random
    import numpy as _np
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "args": vars(args),
        "rng_state": {
            "torch": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": _np.random.get_state(),
            "python": _random.getstate(),
        },
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

    optimizer.zero_grad()
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

        loss = loss / args.grad_accum
        loss.backward()

        if step % args.grad_accum == 0 or step == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item() * args.grad_accum
        epoch_loss_sum += loss.item() * args.grad_accum

        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            print(f"epoch {epoch} | step {step}/{len(train_loader)} | loss {avg_loss:.4f}")
            running_loss = 0.0

    print(f"Epoch {epoch} | Average Loss: {epoch_loss_sum / len(train_loader):.4f}")
    return epoch_loss_sum / len(train_loader)


@torch.no_grad()
def evaluate_val(model, loader, device, model_type="encoder"):
    model.eval()
    total_loss, total_correct, total_examples = 0.0, 0, 0

    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        if model_type == "causal":
            label_ids = batch.label_ids.to(device)
            loss, choice_logits = causal_choice_loss(outputs.logits, label_ids, labels, batch.num_choices)
        else:
            loss, choice_logits = grouped_choice_loss(outputs.logits, labels, batch.num_choices)

        preds = torch.argmax(choice_logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.numel()
        total_loss += loss.item()

    model.train()
    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_examples if total_examples > 0 else 0.0
    return avg_loss, accuracy


def load_val_examples_for_dataset(dataset_name: str) -> List[Dict[str, Any]]:
    _val = "validation"
    if dataset_name == "boolq":
        return load_boolq_examples(_val)
    elif dataset_name == "piqa":
        return load_piqa_examples(_val)
    elif dataset_name == "siqa":
        return load_siqa_examples(_val)
    elif dataset_name == "hellaswag":
        return load_hellaswag_examples(_val)
    elif dataset_name == "arc_challenge":
        return load_arc_examples(_val, "ARC-Challenge")
    elif dataset_name == "arc_easy":
        return load_arc_examples(_val, "ARC-Easy")
    elif dataset_name == "winogrande":
        return load_winogrande_examples(_val)
    elif dataset_name == "openbookqa":
        return load_openbookqa_examples(_val)
    elif dataset_name == "all":
        return load_all_examples(_val)
    return []


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.method in ("both", "ablation"):
        if args.method == "both":
            sweep = ["lora", "dora3"]
        else:
            sweep = ["dora_full", "dora_frozen_magnitude", "dora_magnitude_only"]
        for method in sweep:
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_type == "causal":
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        if not hasattr(model_config, "pad_token_id") or model_config.pad_token_id is None:
            model_config.pad_token_id = tokenizer.eos_token_id

        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                config=model_config,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                config=model_config,
                trust_remote_code=True,
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

    if args.gradient_checkpointing and model_type == "causal" and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    freeze_non_adapter_params(model)
    if not args.load_in_4bit:
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

    print("Loading validation set...")
    val_examples = load_val_examples_for_dataset(args.dataset)
    val_loader = DataLoader(
        MultiChoiceDataset(val_examples),
        batch_size=args.batch_size,
        shuffle=False,
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
        ckpt = torch.load(args.resume_from_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        # Move optimizer states to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        # Restore RNG state so batch ordering matches the original run
        if "rng_state" in ckpt:
            import random as _random
            import numpy as _np
            rng = ckpt["rng_state"]
            torch.random.set_rng_state(rng["torch"])
            if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            _np.random.set_state(rng["numpy"])
            _random.setstate(rng["python"])
            print("Restored RNG state from checkpoint.")
        print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")

    # --------------------
    # Train
    # --------------------
    total_train_start = time.time()
    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt = os.path.join(args.save_dir, f"{args.method}_{args.dataset}_best.pt")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            args=args,
            model_type=model_type,
        )

        val_loss, val_acc = evaluate_val(model, val_loader, device, model_type)

        epoch_time_sec = time.time() - epoch_start_time
        global_step += len(train_loader)

        print(f"\n=== Epoch {epoch} Summary ===")
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")
        print(f"Epoch time: {epoch_time_sec:.2f} sec ({epoch_time_sec / 60:.2f} min)")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.8f}")
        print(f"Global steps: {global_step}\n")

        # Save best checkpoint on val loss improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(path=best_ckpt, model=model, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch, step=global_step, args=args)
            print(f"New best val loss {best_val_loss:.4f} — saved to {best_ckpt}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{args.patience} epochs.")
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        if args.save_every_epoch:
            epoch_ckpt = os.path.join(args.save_dir, f"{args.method}_epoch_{epoch}.pt")
            save_checkpoint(path=epoch_ckpt, model=model, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch, step=global_step, args=args)
            print(f"Saved checkpoint to {epoch_ckpt}")

    total_train_time_sec = time.time() - total_train_start
    print(f"\nTotal training time: {total_train_time_sec:.2f} sec ({total_train_time_sec / 60:.2f} min)")

    final_ckpt = os.path.join(args.save_dir, f"{args.method}_{args.dataset}_final.pt")
    save_checkpoint(path=final_ckpt, model=model, optimizer=optimizer,
                    scheduler=scheduler, epoch=epoch, step=global_step, args=args)
    print(f"Saved final checkpoint to {final_ckpt}")

if __name__ == "__main__":
    main()