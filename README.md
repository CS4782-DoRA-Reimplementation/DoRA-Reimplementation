# DoRA Reimplementation

CS4782 final project reimplementing **DoRA: Weight-Decomposed Low-Rank Adaptation** in a scaled-down text classification / multiple-choice QA setting. The original paper improves parameter-efficient fine-tuning by decomposing pretrained weights into magnitude and direction, then applying low-rank updates to the directional component.

## Introduction

This repository contains our attempt to reproduce and stress-test the main DoRA idea against LoRA baselines. We focus on whether DoRA can outperform LoRA when adapted to a smaller Phi-2 commonsense reasoning setup.

## Chosen Result

We aimed to reproduce the paper's central finding that DoRA improves over LoRA while remaining parameter efficient. The relevant mechanism is the DoRA weight update:

```text
W0 = mV,   V = W0 / ||W0||c,   W' = m * (V + BA) / ||V + BA||c
```

This corresponds to the original paper's magnitude-direction decomposition: freeze the pretrained weight direction baseline, train magnitude `m`, and learn a low-rank directional update `BA`.

## GitHub Contents

- `code/train.py`: training loop, dataset loading, adapter injection, checkpointing, and ablation runs.
- `code/evaluate.py`: evaluation for single datasets or all 8 commonsense datasets.
- `code/dora.py`, `code/dora2.py`, `code/dora_ablation.py`, `code/lora.py`: DoRA, LoRA, and ablation adapter implementations.
- `checkpoints/`: generated locally when training; model checkpoints are not included in the repo.

## Re-implementation Details

We used a frozen Phi-2 causal language model backbone and replaced selected `nn.Linear` layers with adapter-wrapped linear layers. For each multiple-choice example, answer choices are scored by average answer-token log probability, and accuracy is reported over validation/test splits.

Datasets: BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-Challenge, ARC-Easy, and OpenBookQA. Main setup: rank `r = 16`, learning rate `2e-4`, max sequence length `128` or `256`, and comparison against paper LoRA / paper DoRA baselines.

The main implementation challenge was adapting DoRA from the paper's larger LLM setting into a smaller reproducible Phi-2 experiment while keeping the base model frozen and handling variable-choice QA datasets consistently.

## Reproduction Steps

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets accelerate bitsandbytes
```

Train DoRA on one dataset:

```bash
python code/train.py \
  --model_name microsoft/phi-2 \
  --model_type causal \
  --method dora \
  --dataset boolq \
  --rank 16 \
  --alpha 32 \
  --lr 2e-4 \
  --max_length 256 \
  --batch_size 4 \
  --epochs 10 \
  --gradient_checkpointing \
  --load_in_4bit
```

Train across all supported datasets:

```bash
python code/train.py \
  --model_name microsoft/phi-2 \
  --model_type causal \
  --method dora \
  --dataset all \
  --rank 16 \
  --alpha 32 \
  --lr 2e-4 \
  --max_length 256 \
  --batch_size 4 \
  --epochs 10 \
  --gradient_checkpointing \
  --load_in_4bit
```

Evaluate a checkpoint across all 8 datasets:

```bash
python code/evaluate.py \
  --model_name microsoft/phi-2 \
  --model_type causal \
  --dora_checkpoint checkpoints/dora_all_best.pt \
  --dora_method dora \
  --rank 16 \
  --alpha 32 \
  --max_length 256 \
  --batch_size 4 \
  --load_in_4bit \
  --eval_all
```

Recommended compute: CUDA GPU with 4-bit loading for Phi-2. Apple MPS is supported by the code path, but full Phi-2 training is much more practical on a GPU; CPU-only runs are expected to be very slow.

## Results / Insights

| Method | Average accuracy across 8 datasets | Trainable parameters |
| --- | ---: | ---: |
| Paper LoRA | 0.7470 | 28,100,600 |
| Paper DoRA | 0.7750 | 28,800,600 |
| Our DoRA | **0.8034** | **25,241,600** |

Our DoRA implementation achieved the best average accuracy across the 8 commonsense datasets while using the fewest trainable parameters in our scaled-down Phi-2 setup.

## Conclusion

This reimplementation supports the core DoRA intuition: separating magnitude from direction can improve parameter-efficient adaptation beyond standard LoRA. In our smaller Phi-2 setting, DoRA outperformed the comparison baselines, though broader sweeps over rank, learning rate, and longer training would make the result stronger.

## References

- Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., & Chen, M.-H. **DoRA: Weight-Decomposed Low-Rank Adaptation**.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. **LoRA: Low-Rank Adaptation of Large Language Models**.
- Hugging Face `transformers` and `datasets` libraries.

## Acknowledgements

This project was completed for Cornell University CS 4782 by Raj Patel, Nicholas Lombardo, Jake Berko, and Sruti Gudapati.
