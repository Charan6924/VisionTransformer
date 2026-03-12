import os
import csv
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import webdataset as wds
from torchvision import transforms
from dataclasses import dataclass

from transformer import SiglipVisionModel, SiglipVisionConfig
from encoder import SiglipTextEncoder, SiglipTokenizer


@dataclass
class TrainConfig:
    train_shards: str = "./cc3m_shards/cc3m-train-{0000..0575}.tar" 
    num_workers: int = 3
    batch_size: int = 256
    max_length: int = 64
    embed_dim: int = 768
    freeze_text_encoder: bool = False
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0
    log_every: int = 50          # steps
    save_every: int = 1000       # steps
    checkpoint_dir: str = "./checkpoints"
    log_file: str = "./logs/metrics.csv"
    resume_from: str = None     



class SiglipLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_embeds, text_embeds, logit_scale, logit_bias=None):
        logits = logit_scale.exp() * (image_embeds @ text_embeds.T)  
        if logit_bias is not None:
            logits = logits + logit_bias

        B = logits.shape[0]
        labels = 2 * torch.eye(B, device=logits.device) - 1 #matrix of 1s on diagonal -1 everywhere else
        loss = -F.logsigmoid(labels * logits).sum() / B
        return loss

class SiglipModel(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        vision_cfg = SiglipVisionConfig()
        self.vision_encoder = SiglipVisionModel(vision_cfg)
        self.text_encoder = SiglipTextEncoder(
            embed_dim=cfg.embed_dim,
            freeze_encoder=cfg.freeze_text_encoder,
        )
        self.vision_projection = nn.Linear(vision_cfg.hidden_size, cfg.embed_dim, bias=False)
        self.logit_bias = nn.Parameter(torch.zeros([]))

    def encode_image(self, pixel_values):
        features = self.vision_encoder(pixel_values)
        pooled = features.mean(dim=1)                          
        projected = self.vision_projection(pooled)             
        return F.normalize(projected, dim=-1)

    def encode_text(self, input_ids, attention_mask, token_type_ids=None):
        return self.text_encoder(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,)

    def forward(self, pixel_values, input_ids, attention_mask, token_type_ids=None):
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask, token_type_ids)
        return image_embeds, text_embeds, self.text_encoder.logit_scale, self.logit_bias

def build_dataloader(cfg: TrainConfig):
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = (
        wds.WebDataset(cfg.train_shards, resampled=True, shardshuffle = True)
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg", "txt")
        .map_tuple(transform, lambda x: x)
        .batched(cfg.batch_size, partial=False)
    )

    loader = DataLoader(
        dataset,
        batch_size=None,       
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return loader

def get_lr(step: int, cfg: TrainConfig, total_steps: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

class CSVLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.file = open(path, "a", newline="")
        self.writer = None

    def log(self, row: dict):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=row.keys())
            if os.path.getsize(self.path) == 0:
                self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()

def save_checkpoint(model, optimizer, scaler, step, epoch, loss, cfg: TrainConfig):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"step_{step:07d}.pt")
    torch.save({
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "loss": loss,
    }, path)
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"  [ckpt] resumed from step {ckpt['step']}")
    return ckpt["step"], ckpt["epoch"]

def train(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = SiglipModel(cfg).to(device)
    criterion = SiglipLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay,betas=(0.9, 0.98),)
    scaler = torch.amp.GradScaler('cuda')
    tokenizer = SiglipTokenizer(max_length=cfg.max_length)
    logger = CSVLogger(cfg.log_file)

    start_step, start_epoch = 0, 0
    if cfg.resume_from:
        start_step, start_epoch = load_checkpoint(cfg.resume_from, model, optimizer, scaler)

    loader = build_dataloader(cfg)
    steps_per_epoch = 2_500_000 // cfg.batch_size
    total_steps = steps_per_epoch * cfg.epochs

    step = start_step
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (images, texts) in enumerate(loader):
            lr = get_lr(step, cfg, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            images = images.to(device, non_blocking=True)
            tokens = tokenizer(list(texts), device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                image_embeds, text_embeds, logit_scale, logit_bias = model(
                    images,
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens.get("token_type_ids"),
                )
                loss = criterion(image_embeds, text_embeds, logit_scale, logit_bias)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            epoch_loss += loss_val
            step += 1

            if step % cfg.log_every == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(
                    f"epoch {epoch+1:03d} | step {step:07d} | "
                    f"loss {loss_val:.4f} | avg {avg_loss:.4f} | "
                    f"lr {lr:.2e} | scale {logit_scale.exp().item():.2f}"
                )
                logger.log({
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": round(loss_val, 6),
                    "avg_epoch_loss": round(avg_loss, 6),
                    "lr": round(lr, 8),
                    "logit_scale": round(logit_scale.exp().item(), 4),
                    "logit_bias": round(logit_bias.item(), 6),
                    "elapsed_s": round(time.time() - epoch_start, 1),
                })

            if step % cfg.save_every == 0:
                save_checkpoint(model, optimizer, scaler, step, epoch, loss_val, cfg)

        save_checkpoint(model, optimizer, scaler, step, epoch + 1, loss_val, cfg)
        print(f"Epoch {epoch+1} done — avg loss: {epoch_loss / steps_per_epoch:.4f}")

    logger.close()
    print("Training complete.")


if __name__ == "__main__":
    cfg = TrainConfig(
        train_shards = "./cc3m_shards/cc3m-train-{0000..0575}.tar",
        batch_size=256,
        epochs=15,
        learning_rate=1e-4,
        checkpoint_dir="./checkpoints",
        log_file="./logs/metrics.csv",
        resume_from = '/scratch/pioneer/users/cxv166/VisionTransformer/checkpoints/step_0078000.pt'
    )
    train(cfg)