# -*- coding: utf-8 -*-
"""训练脚本：AdamW + 余弦退火 + 混合精度(AMP) + 早停，输出最优权重与训练日志。"""

import os
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import CFG
from src.data.dataset import TMJDataset, list_samples, split_by_patient
from src.models.unet3d import build_model
from src.losses import DiceCELoss


def cosine_lr(optimizer, base_lr, step, total_steps, warmup=10):
    """带预热的余弦退火学习率。"""
    if step < warmup:
        lr = base_lr * (step + 1) / warmup
    else:
        t = (step - warmup) / max(total_steps - warmup, 1)
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * t))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses, dices = [], []
    for img, lab in loader:
        img, lab = img.to(device), lab.to(device)
        logits = model(img)
        losses.append(criterion(logits, lab).item())
        pred = logits.argmax(1)
        # 逐类 Dice（不含背景）
        for c in range(1, logits.shape[1]):
            p, g = pred == c, lab == c
            denom = p.sum() + g.sum()
            if denom > 0:
                dices.append((2 * (p & g).sum() / denom).item())
    return float(np.mean(losses)), float(np.mean(dices)) if dices else 0.0


def train(cfg=CFG):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    samples = list_samples(cfg.proc_dir)
    assert samples, f"在 {cfg.proc_dir} 中未找到预处理数据，请先运行预处理。"
    train_s, val_s, test_s = split_by_patient(samples, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    print(f"样本数  train/val/test = {len(train_s)}/{len(val_s)}/{len(test_s)}（按患者划分）")

    train_ds = TMJDataset(train_s, cfg.patch_size, cfg.pos_sample_prob, augment=True,
                          iters_per_epoch=cfg.iters_per_epoch)
    val_ds = TMJDataset(val_s, cfg.patch_size, pos_sample_prob=1.0, augment=False)
    train_ld = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, 1, shuffle=False)

    model = build_model(cfg).to(device)
    criterion = DiceCELoss(cfg.num_classes, cfg.dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")

    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, "best_model.pt")
    log = []
    best_dice, bad_epochs = -1.0, 0
    steps_per_epoch = max(len(train_ld), 1)
    total_steps = steps_per_epoch * cfg.epochs
    step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        ep_loss = []
        for img, lab in train_ld:
            cosine_lr(optimizer, cfg.lr, step, total_steps)
            img, lab = img.to(device), lab.to(device)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device == "cuda"):
                loss = criterion(model(img), lab)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ep_loss.append(loss.item())
            step += 1
        val_loss, val_dice = validate(model, val_ld, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{epoch:03d}/{cfg.epochs}] train_loss={np.mean(ep_loss):.4f} "
              f"val_loss={val_loss:.4f} val_Dice={val_dice:.4f} lr={lr_now:.2e} "
              f"({time.time()-t0:.0f}s)")
        log.append({"epoch": epoch, "train_loss": float(np.mean(ep_loss)),
                    "val_loss": val_loss, "val_dice": val_dice, "lr": lr_now})
        if val_dice > best_dice:
            best_dice, bad_epochs = val_dice, 0
            torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                        "epoch": epoch, "val_dice": val_dice}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                print(f"早停触发（{cfg.early_stop_patience} 轮验证 Dice 未提升）")
                break

    with open(os.path.join(cfg.out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"训练结束。最优验证 Dice = {best_dice:.4f}，权重保存在 {ckpt_path}")
    return best_dice


if __name__ == "__main__":
    train()
