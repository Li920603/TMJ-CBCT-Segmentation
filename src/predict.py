# -*- coding: utf-8 -*-
"""
单例预测：对一个预处理后的 NIfTI 体数据（或预处理后的 ROI）运行训练好的模型，
输出分割掩膜 NIfTI 与三视图对比图（若有标注则同时评估）。

用法：
    python -m src.predict --image data/processed/xxx_L/image.nii.gz \
        --ckpt outputs/best_model.pt --out outputs/predictions [--label .../label.nii.gz]
"""

import os
import argparse
import numpy as np
import nibabel as nib
import torch

from src.evaluate import load_model
from src.infer import sliding_window_inference
from src.postprocess import keep_largest_component
from src.metrics import evaluate_case
from src.visualize import visualize_case


def predict(image_path, ckpt_path, out_dir, label_path=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ckpt = load_model(ckpt_path, device)
    ref = nib.load(image_path)
    image = np.asanyarray(ref.dataobj).astype(np.float32)

    pred = sliding_window_inference(model, image, cfg.patch_size, cfg.num_classes,
                                    overlap=0.5, device=device)
    pred = keep_largest_component(pred, label=1)

    os.makedirs(out_dir, exist_ok=True)
    cid = os.path.basename(os.path.dirname(image_path)) or "case"
    out_nii = os.path.join(out_dir, f"{cid}_pred.nii.gz")
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), ref.affine), out_nii)
    print(f"预测掩膜已保存: {out_nii}")

    if label_path and os.path.exists(label_path):
        gt = np.asanyarray(nib.load(label_path).dataobj).astype(np.uint8)
        m = evaluate_case(pred, gt, spacing=cfg.target_spacing,
                          labels=range(1, cfg.num_classes))
        for c, mm in m.items():
            print(f"类别 {c}: Dice={mm['dice']:.4f} IoU={mm['iou']:.4f} "
                  f"HD95={mm['hd95']:.2f}mm ASD={mm['asd']:.2f}mm")
        out_png = os.path.join(out_dir, f"{cid}_vis.png")
        visualize_case(image, gt, pred, out_png, title=f"Prediction - {cid}")
        print(f"可视化已保存: {out_png}")
    return pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/predictions")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    predict(args.image, args.ckpt, args.out, args.label)
