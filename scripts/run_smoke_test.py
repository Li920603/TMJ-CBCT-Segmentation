# -*- coding: utf-8 -*-
"""
冒烟测试：用合成数据快速验证 数据->模型->训练->推理->评估->可视化 全链路。
CPU 上约 5~8 分钟跑完。真实训练前请务必先跑通此脚本。

用法:  python scripts/run_smoke_test.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from dataclasses import replace

from src.config import CFG
from src.data.synthetic import generate_synthetic_dataset
from src.train import train
from src.infer import sliding_window_inference
from src.metrics import evaluate_case
from src.visualize import visualize_case
from src.models.unet3d import build_model
from src.data.dataset import list_samples, split_by_patient
import nibabel as nib


def main():
    # 小型化配置：小模型 + 小 patch + 少 epoch，确保 CPU 也能快速跑通
    cfg = replace(CFG,
                  proc_dir="data/smoke",
                  out_dir="outputs/smoke",
                  base_features=8,
                  num_levels=3,
                  norm="batch",
                  patch_size=(32, 32, 32),
                  batch_size=2,
                  iters_per_epoch=20,
                  pos_sample_prob=0.5,
                  epochs=20,
                  early_stop_patience=15,
                  num_workers=0,
                  amp=False)

    print("=" * 50)
    print("步骤 1/4：生成合成数据")
    print("=" * 50)
    generate_synthetic_dataset(cfg.proc_dir, n_cases=6, shape=(96, 96, 96))

    print("\n" + "=" * 50)
    print("步骤 2/4：训练（冒烟：20 epoch 小模型）")
    print("=" * 50)
    best_dice = train(cfg)

    print("\n" + "=" * 50)
    print("步骤 3/4：滑动窗口推理 + 定量评估")
    print("=" * 50)
    ckpt = torch.load(os.path.join(cfg.out_dir, "best_model.pt"), map_location="cpu",
                      weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    samples = list_samples(cfg.proc_dir)
    _, _, test_s = split_by_patient(samples, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    img_path, lab_path, cid = test_s[0]
    image = np.asanyarray(nib.load(img_path).dataobj).astype(np.float32)
    gt = np.asanyarray(nib.load(lab_path).dataobj).astype(np.uint8)
    pred = sliding_window_inference(model, image, cfg.patch_size, cfg.num_classes,
                                    overlap=0.5, device="cpu")
    from src.postprocess import keep_largest_component
    pred = keep_largest_component(pred, label=1)
    metrics = evaluate_case(pred, gt, spacing=cfg.target_spacing, labels=(1,))
    print(f"测试样本 {cid} 指标:")
    for c, m in metrics.items():
        print(f"  类别 {c}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} "
              f"HD95={m['hd95']:.2f}mm ASD={m['asd']:.2f}mm")

    print("\n" + "=" * 50)
    print("步骤 4/4：可视化")
    print("=" * 50)
    out_png = os.path.join(cfg.out_dir, f"vis_{cid}.png")
    os.makedirs(cfg.out_dir, exist_ok=True)
    visualize_case(image, gt, pred, out_png, title=f"Smoke Test - {cid}")
    print(f"可视化已保存: {out_png}")

    assert best_dice > 0.3, f"冒烟测试失败：验证 Dice={best_dice:.4f} 过低"
    print(f"\n冒烟测试通过！验证 Dice = {best_dice:.4f}，全链路工作正常。")
    print("下一步：下载真实数据 python -m src.data.download_dataset --out data/raw")


if __name__ == "__main__":
    main()
