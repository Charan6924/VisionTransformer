# SigLIP Vision Transformer

A from-scratch PyTorch implementation of the SigLIP (Sigmoid Loss for Language Image Pre-Training) vision encoder, paired with a pretrained BERT text encoder and trained on the CC3M dataset.

Based on the paper: [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) — Zhai et al., Google, 2023.

---

## Overview

This project implements the full SigLIP pretraining pipeline:

- **Vision encoder** — Vision Transformer (ViT) built from scratch in PyTorch
- **Text encoder** — Pretrained `bert-base-uncased` with a learned projection head
- **SigLIP loss** — Sigmoid binary cross-entropy over image-text pairs, as opposed to CLIP's softmax
- **Training** — Trained end-to-end on CC3M (~2.5M image-text pairs) with bf16 mixed precision on a single H100 GPU

---

## Architecture

```
Image (224×224) ──► Patch Embedding ──► Transformer Encoder (12L) ──► Mean Pool ──► Vision Projection ──► L2 Norm ──► image embedding
                                                                                                                              │
                                                                                                                    SigLIP Loss (sigmoid)
                                                                                                                              │
Text ────────────► BERT (bert-base-uncased) ──► [CLS] token ──► Text Projection ──► L2 Norm ──────────────────► text embedding
```

**Vision Encoder config:**
| Parameter | Value |
|---|---|
| Image size | 224×224 |
| Patch size | 16×16 |
| Num patches | 196 |
| Hidden size | 768 |
| Layers | 12 |
| Attention heads | 12 |
| MLP intermediate size | 3072 |

---

## SigLIP Loss

Unlike CLIP which uses softmax over the batch, SigLIP treats every image-text pair as an independent binary classification problem:

```python
labels = 2 * torch.eye(B) - 1  # +1 for matched pairs, -1 for unmatched
loss = -F.logsigmoid(labels * logits).sum() / B
```

This makes the loss more stable at smaller batch sizes and removes the implicit assumption that only one positive exists per row.

---

## Project Structure

```
VisionTransformer/
├── transformer.py      # Vision Transformer implementation
├── encoder.py          # Pretrained BERT text encoder + tokenizer
├── main.py             # Full SigLIP model, loss, and training loop
├── checkpoints/        # Saved model checkpoints
└── logs/
    └── metrics.csv     # Training metrics logged every 50 steps
```

---

## Training

**Dataset:** CC3M (Conceptual Captions 3M) in webdataset format  
**Hardware:** Single NVIDIA H100 (80GB)  
**Precision:** bf16 mixed precision  
**Batch size:** 256  
**Optimizer:** AdamW (β1=0.9, β2=0.98, wd=0.01)  
**LR schedule:** Linear warmup + cosine decay  
**Epochs:** 15

### Setup

```bash
# install dependencies
uv add torch torchvision transformers webdataset

# download CC3M dataset (~140GB)
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='pixparse/cc3m-wds',
    repo_type='dataset',
    local_dir='./cc3m_shards',
)
"
```

### Run training

```bash
uv run python main.py
```

Update the shard path in `main.py` before running:
```python
cfg = TrainConfig(
    train_shards="./cc3m_shards/cc3m-train-{0000..0575}.tar",
    batch_size=256,
    epochs=15,
    learning_rate=1e-4,
    checkpoint_dir="./checkpoints",
    log_file="./logs/metrics.csv",
)
```

### Resume from checkpoint

```python
cfg = TrainConfig(
    ...
    resume_from="./checkpoints/step_0010000.pt",
)
```

---

## Logged Metrics

Every 50 steps the following are written to `logs/metrics.csv`:

| Column | Description |
|---|---|
| `step` | Global training step |
| `epoch` | Current epoch |
| `loss` | Per-step SigLIP loss |
| `avg_epoch_loss` | Running average loss for the epoch |
| `lr` | Current learning rate |
| `logit_scale` | Learned temperature (exp of logit_scale param) |
| `logit_bias` | Learned bias term |
| `elapsed_s` | Seconds elapsed since epoch start |

---

## Dependencies

```
torch
torchvision
transformers
webdataset
huggingface_hub
```



```
