import argparse
import os
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from train import (
    set_seed,
    replace_linear_layers_with_adapters,
    freeze_non_adapter_params,
    MultiChoiceDataset,
    load_boolq_examples,
    load_hellaswag_examples,
    load_piqa_examples,
    collate_fn,
    grouped_choice_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, choices=["boolq", "hellaswag", "piqa"], default="boolq")
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to evaluate (validation or test). Falls back to validation if test is unavailable.",
    )

    parser.add_argument("--dora_method", type=str, choices=["dora", "dora2"], default="dora2",
                        help="Which DoRA implementation to use (dora=dora.py, dora2=dora2.py)")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", nargs="*", default=None)

    parser.add_argument(
        "--dora_checkpoint",
        type=str,
        default="./checkpoints/dora_final_roberta.pt",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default="./checkpoints/lora_final_roberta.pt",
    )
    return parser.parse_args()


def _extract_state_dict(ckpt_obj: Dict[str, Any]) -> Dict[str, Any]:
    # Support both full training checkpoints and raw state_dict files.
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]
    return ckpt_obj


def build_model(method: str, args, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
    )
    replace_linear_layers_with_adapters(
        module=model,
        method=method,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=args.target_modules,
    )
    freeze_non_adapter_params(model)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model, loader, device: torch.device):
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss, choice_logits = grouped_choice_loss(outputs.logits, labels, batch.num_choices)

        preds = torch.argmax(choice_logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.numel()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_examples if total_examples > 0 else 0.0
    return {"loss": avg_loss, "accuracy": accuracy, "n_examples": total_examples}


def resolve_eval_examples(dataset: str, split: str):
    loaders = {
        "boolq": load_boolq_examples,
        "hellaswag": load_hellaswag_examples,
        "piqa": load_piqa_examples,
    }
    load_fn = loaders[dataset]
    try:
        return load_fn(split)
    except Exception as exc:
        if split != "validation":
            print(f"Could not load {dataset} split '{split}' ({exc}). Falling back to 'validation'.")
            return load_fn("validation")
        raise


def run_one(label: str, method: str, ckpt_path: str, args, loader, device: torch.device):
    if not os.path.exists(ckpt_path):
        print(f"[{label}] Missing checkpoint: {ckpt_path}")
        return None

    print(f"[{label}] Loading checkpoint: {ckpt_path}")
    model = build_model(method=method, args=args, device=device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    # strict=False keeps evaluation robust to minor naming differences.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[{label}] Missing keys while loading: {len(missing)}")
    if unexpected:
        print(f"[{label}] Unexpected keys while loading: {len(unexpected)}")

    metrics = evaluate_model(model, loader, device)
    print(
        f"[{label}] loss={metrics['loss']:.4f} | "
        f"accuracy={metrics['accuracy']:.4f} | "
        f"n={metrics['n_examples']}"
    )
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_examples = resolve_eval_examples(args.dataset, args.split)
    eval_dataset = MultiChoiceDataset(eval_examples)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length),
    )

    dora_metrics = run_one(
        label="DoRA",
        method=args.dora_method,
        ckpt_path=args.dora_checkpoint,
        args=args,
        loader=eval_loader,
        device=device,
    )
    lora_metrics = run_one(
        label="LoRA",
        method="lora",
        ckpt_path=args.lora_checkpoint,
        args=args,
        loader=eval_loader,
        device=device,
    )

    print("\n=== Comparison ===")
    if dora_metrics is not None:
        print(
            f"DoRA  | loss={dora_metrics['loss']:.4f} | "
            f"acc={dora_metrics['accuracy']:.4f}"
        )
    else:
        print("DoRA  | checkpoint missing")

    if lora_metrics is not None:
        print(
            f"LoRA  | loss={lora_metrics['loss']:.4f} | "
            f"acc={lora_metrics['accuracy']:.4f}"
        )
    else:
        print("LoRA  | checkpoint missing")

    if dora_metrics is not None and lora_metrics is not None:
        acc_diff = dora_metrics["accuracy"] - lora_metrics["accuracy"]
        loss_diff = dora_metrics["loss"] - lora_metrics["loss"]
        # Positive acc delta means DoRA outperformed LoRA on this split.
        print(f"Delta (DoRA - LoRA): acc={acc_diff:+.4f}, loss={loss_diff:+.4f}")


if __name__ == "__main__":
    main()