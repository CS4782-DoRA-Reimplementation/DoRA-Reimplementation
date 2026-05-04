import argparse
import os
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from train import (
    set_seed,
    detect_model_type,
    replace_linear_layers_with_adapters,
    freeze_non_adapter_params,
    MultiChoiceDataset,
    load_boolq_examples,
    load_piqa_examples,
    load_siqa_examples,
    load_hellaswag_examples,
    load_arc_examples,
    load_winogrande_examples,
    load_openbookqa_examples,
    collate_fn,
    collate_fn_causal,
    grouped_choice_loss,
    causal_choice_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--model_type", type=str, choices=["encoder", "causal", "auto"], default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, choices=["boolq", "piqa", "siqa", "hellaswag", "arc_challenge", "arc_easy", "winogrande", "openbookqa"], default="boolq")
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to evaluate (validation or test). Falls back to validation if test is unavailable.",
    )

    parser.add_argument("--dora_method", type=str, choices=["dora", "dora2", "dora3"], default="dora3",
                        help="Which DoRA implementation to use")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--eval_all", action="store_true", help="Evaluate DoRA3 checkpoint across all datasets and print a table")
    parser.add_argument("--eval_dora3_full", action="store_true", help="Evaluate DoRA3 on val+test for all datasets and print a full results table")
    parser.add_argument("--eval_dora3_single", action="store_true", help="Evaluate DoRA3 on a single dataset (use --dataset) on val and test splits")
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


def _load_causal_model(args, tokenizer):
    from transformers import AutoModelForCausalLM, AutoConfig
    model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    if not hasattr(model_config, "pad_token_id") or model_config.pad_token_id is None:
        model_config.pad_token_id = tokenizer.eos_token_id
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        return AutoModelForCausalLM.from_pretrained(
            args.model_name, config=model_config,
            quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
        )
    return AutoModelForCausalLM.from_pretrained(
        args.model_name, config=model_config,
        torch_dtype=torch.float16, trust_remote_code=True,
    )


def build_model(method: str, args, device: torch.device, model_type: str = "encoder", tokenizer=None):
    if model_type == "causal":
        model = _load_causal_model(args, tokenizer)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    replace_linear_layers_with_adapters(
        module=model,
        method=method,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=args.target_modules,
    )
    freeze_non_adapter_params(model)
    if not args.load_in_4bit:
        model.to(device)
    model.eval()
    return model


def build_baseline_model(args, device: torch.device, model_type: str = "encoder", tokenizer=None):
    if model_type == "causal":
        model = _load_causal_model(args, tokenizer)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    if not args.load_in_4bit:
        model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model, loader, device: torch.device, model_type: str = "encoder"):
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

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

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_examples if total_examples > 0 else 0.0
    return {"loss": avg_loss, "accuracy": accuracy, "n_examples": total_examples}


def resolve_eval_examples(dataset: str, split: str):
    if dataset == "arc_challenge":
        return load_arc_examples(split, "ARC-Challenge")
    if dataset == "arc_easy":
        return load_arc_examples(split, "ARC-Easy")
    if dataset == "piqa":
        return load_piqa_examples(split if split != "test" else "validation")
    if dataset == "siqa":
        return load_siqa_examples(split if split != "test" else "validation")
    if dataset == "winogrande":
        return load_winogrande_examples(split if split != "test" else "validation")

    loaders = {
        "boolq": load_boolq_examples,
        "hellaswag": load_hellaswag_examples,
        "openbookqa": load_openbookqa_examples,
    }
    load_fn = loaders[dataset]
    try:
        return load_fn(split)
    except Exception as exc:
        if split != "validation":
            print(f"Could not load {dataset} split '{split}' ({exc}). Falling back to 'validation'.")
            return load_fn("validation")
        raise


def run_one(label: str, method: str, ckpt_path: str, args, loader, device: torch.device, model_type: str = "encoder", tokenizer=None):
    if not os.path.exists(ckpt_path):
        print(f"[{label}] Missing checkpoint: {ckpt_path}")
        return None

    print(f"[{label}] Loading checkpoint: {ckpt_path}")
    model = build_model(method=method, args=args, device=device, model_type=model_type, tokenizer=tokenizer)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[{label}] Missing keys while loading: {len(missing)}")
    if unexpected:
        print(f"[{label}] Unexpected keys while loading: {len(unexpected)}")

    metrics = evaluate_model(model, loader, device, model_type=model_type)
    print(
        f"[{label}] loss={metrics['loss']:.4f} | "
        f"accuracy={metrics['accuracy']:.4f} | "
        f"n={metrics['n_examples']}"
    )
    return metrics


def run_baseline(args, loader, device: torch.device, model_type: str = "encoder", tokenizer=None):
    print("[Baseline] Evaluating plain pretrained model (no adapter checkpoint)")
    model = build_baseline_model(args=args, device=device, model_type=model_type, tokenizer=tokenizer)
    metrics = evaluate_model(model, loader, device, model_type=model_type)
    print(
        f"[Baseline] loss={metrics['loss']:.4f} | "
        f"accuracy={metrics['accuracy']:.4f} | "
        f"n={metrics['n_examples']}"
    )
    return metrics


ALL_DATASETS = ["boolq", "piqa", "siqa", "hellaswag", "winogrande", "arc_challenge", "arc_easy", "openbookqa"]

def evaluate_all_datasets(args, device, model_type, tokenizer):
    print(f"\nLoading DoRA3 model from {args.dora_checkpoint}...")
    model = build_model(method=args.dora_method, args=args, device=device, model_type=model_type, tokenizer=tokenizer)
    ckpt = torch.load(args.dora_checkpoint, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    results = {}
    for dataset in ALL_DATASETS:
        print(f"Evaluating {dataset}...", end=" ", flush=True)
        try:
            examples = resolve_eval_examples(dataset, "validation")
            loader = DataLoader(
                MultiChoiceDataset(examples),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=lambda batch: collate_fn_causal(batch, tokenizer, args.max_length)
                    if model_type == "causal" else collate_fn(batch, tokenizer, args.max_length),
            )
            metrics = evaluate_model(model, loader, device, model_type=model_type)
            results[dataset] = metrics
            print(f"acc={metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"FAILED ({e})")
            results[dataset] = None

    print("\n" + "=" * 55)
    print(f"{'Dataset':<18} {'Accuracy':>10} {'Loss':>10} {'N':>8}")
    print("-" * 55)
    accs = []
    for dataset in ALL_DATASETS:
        m = results[dataset]
        if m:
            print(f"{dataset:<18} {m['accuracy']:>10.4f} {m['loss']:>10.4f} {m['n_examples']:>8}")
            accs.append(m['accuracy'])
        else:
            print(f"{dataset:<18} {'N/A':>10} {'N/A':>10} {'N/A':>8}")
    print("-" * 55)
    if accs:
        print(f"{'Average':<18} {sum(accs)/len(accs):>10.4f}")
    print("=" * 55)
    return results


def evaluate_dora3_full(args, device, model_type, tokenizer):
    """Evaluate DoRA3 checkpoint on val and test splits for every dataset, print a full table."""
    print(f"\nLoading DoRA3 checkpoint: {args.dora_checkpoint}")
    model = build_model(method=args.dora_method, args=args, device=device, model_type=model_type, tokenizer=tokenizer)
    ckpt = torch.load(args.dora_checkpoint, map_location="cpu", weights_only=False)
    ckpt_meta = ckpt if isinstance(ckpt, dict) else {}
    state_dict = _extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    model.eval()

    saved_epoch = ckpt_meta.get("epoch", "?")
    saved_step  = ckpt_meta.get("step", "?")
    saved_args  = ckpt_meta.get("args", {})

    if model_type == "causal":
        _collate = lambda batch: collate_fn_causal(batch, tokenizer, args.max_length)
    else:
        _collate = lambda batch: collate_fn(batch, tokenizer, args.max_length)

    def _eval_split(dataset, split):
        try:
            examples = resolve_eval_examples(dataset, split)
            loader = DataLoader(
                MultiChoiceDataset(examples),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=_collate,
            )
            return evaluate_model(model, loader, device, model_type=model_type)
        except Exception as e:
            return None

    results = {}
    for dataset in ALL_DATASETS:
        print(f"  {dataset}: val...", end="", flush=True)
        val_m = _eval_split(dataset, "validation")
        print(f" acc={val_m['accuracy']:.4f}" if val_m else " FAILED", end="")
        print(f"  test...", end="", flush=True)
        test_m = _eval_split(dataset, "test")
        print(f" acc={test_m['accuracy']:.4f}" if test_m else " N/A")
        results[dataset] = {"val": val_m, "test": test_m}

    # ── Header ──────────────────────────────────────────────────────────────
    W = 100
    print("\n" + "=" * W)
    print(f"  DoRA3 Results  |  checkpoint: {args.dora_checkpoint}")
    print(f"  Epoch: {saved_epoch}  |  Step: {saved_step}  |  "
          f"Rank: {saved_args.get('rank', args.rank)}  |  "
          f"Alpha: {saved_args.get('alpha', args.alpha)}  |  "
          f"LR: {saved_args.get('lr', args.lr)}  |  "
          f"Model: {saved_args.get('model_name', args.model_name)}")
    print("=" * W)
    hdr = (f"{'Dataset':<18} "
           f"{'Val Acc':>8} {'Val Loss':>9} {'Val N':>7}   "
           f"{'Test Acc':>9} {'Test Loss':>10} {'Test N':>7}")
    print(hdr)
    print("-" * W)

    val_accs, test_accs = [], []
    for dataset in ALL_DATASETS:
        v = results[dataset]["val"]
        t = results[dataset]["test"]
        val_str  = (f"{v['accuracy']:>8.4f} {v['loss']:>9.4f} {v['n_examples']:>7}"
                    if v else f"{'N/A':>8} {'N/A':>9} {'N/A':>7}")
        test_str = (f"{t['accuracy']:>9.4f} {t['loss']:>10.4f} {t['n_examples']:>7}"
                    if t else f"{'N/A':>9} {'N/A':>10} {'N/A':>7}")
        print(f"{dataset:<18} {val_str}   {test_str}")
        if v:
            val_accs.append(v["accuracy"])
        if t:
            test_accs.append(t["accuracy"])

    print("-" * W)
    val_avg  = f"{sum(val_accs)/len(val_accs):>8.4f}"  if val_accs  else f"{'N/A':>8}"
    test_avg = f"{sum(test_accs)/len(test_accs):>9.4f}" if test_accs else f"{'N/A':>9}"
    print(f"{'Average':<18} {val_avg} {'':>9} {'':>7}   {test_avg}")
    print("=" * W)
    return results


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

    model_type = args.model_type if args.model_type != "auto" else detect_model_type(args.model_name)
    print(f"Model type: {model_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.eval_dora3_full:
        evaluate_dora3_full(args, device, model_type, tokenizer)
        return

    if args.eval_dora3_single:
        if model_type == "causal":
            _collate = lambda batch: collate_fn_causal(batch, tokenizer, args.max_length)
        else:
            _collate = lambda batch: collate_fn(batch, tokenizer, args.max_length)

        print(f"\nLoading DoRA3 checkpoint: {args.dora_checkpoint}")
        model = build_model(method=args.dora_method, args=args, device=device, model_type=model_type, tokenizer=tokenizer)
        ckpt = torch.load(args.dora_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(_extract_state_dict(ckpt), strict=False)
        model.eval()

        for split in ["validation", "test"]:
            try:
                examples = resolve_eval_examples(args.dataset, split)
                loader = DataLoader(MultiChoiceDataset(examples), batch_size=args.batch_size, shuffle=False, collate_fn=_collate)
                m = evaluate_model(model, loader, device, model_type=model_type)
                print(f"[DoRA3 | {args.dataset} | {split}] acc={m['accuracy']:.4f} | loss={m['loss']:.4f} | n={m['n_examples']}")
            except Exception as e:
                print(f"[DoRA3 | {args.dataset} | {split}] N/A ({e})")
        return

    if args.eval_all:
        evaluate_all_datasets(args, device, model_type, tokenizer)
        return

    eval_examples = resolve_eval_examples(args.dataset, args.split)
    eval_dataset = MultiChoiceDataset(eval_examples)

    if model_type == "causal":
        _collate = lambda batch: collate_fn_causal(batch, tokenizer, args.max_length)
    else:
        _collate = lambda batch: collate_fn(batch, tokenizer, args.max_length)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    baseline_metrics = run_baseline(args=args, loader=eval_loader, device=device, model_type=model_type, tokenizer=tokenizer)
    dora_metrics = run_one(
        label="DoRA",
        method=args.dora_method,
        ckpt_path=args.dora_checkpoint,
        args=args,
        loader=eval_loader,
        device=device,
        model_type=model_type,
        tokenizer=tokenizer,
    )
    lora_metrics = run_one(
        label="LoRA",
        method="lora",
        ckpt_path=args.lora_checkpoint,
        args=args,
        loader=eval_loader,
        device=device,
        model_type=model_type,
        tokenizer=tokenizer,
    )

    print("\n=== Comparison ===")
    if baseline_metrics is not None:
        print(
            f"Base  | loss={baseline_metrics['loss']:.4f} | "
            f"acc={baseline_metrics['accuracy']:.4f}"
        )
    else:
        print("Base  | unavailable")

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
    if baseline_metrics is not None and dora_metrics is not None:
        acc_diff = dora_metrics["accuracy"] - baseline_metrics["accuracy"]
        loss_diff = dora_metrics["loss"] - baseline_metrics["loss"]
        print(f"Delta (DoRA - Base): acc={acc_diff:+.4f}, loss={loss_diff:+.4f}")
    if baseline_metrics is not None and lora_metrics is not None:
        acc_diff = lora_metrics["accuracy"] - baseline_metrics["accuracy"]
        loss_diff = lora_metrics["loss"] - baseline_metrics["loss"]
        print(f"Delta (LoRA - Base): acc={acc_diff:+.4f}, loss={loss_diff:+.4f}")


if __name__ == "__main__":
    main()