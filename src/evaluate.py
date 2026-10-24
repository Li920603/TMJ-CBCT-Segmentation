# -*- coding: utf-8 -*-
"""
测试集批量评估：对测试集中的每例样本做滑动窗口推理 + 后处理，
输出逐例与总体（均值±标准差）的 Dice / IoU / HD95 / ASD，并保存 CSV/JSON。

用法：
    python -m src.evaluate --ckpt outputs/best_model.pt [--out outputs/test_metrics]
"""

import os
import csv
import json
import argparse
import numpy as np
import torch
import nibabel as nib

from src.config import CFG, Config
from src.data.dataset import list_samples, split_by_patient
from src.models.unet3d import build_model
from src.infer import sliding_window_inference
from src.postprocess import keep_largest_component
from src.metrics import evaluate_case


def load_model(ckpt_path, device="cpu"):
    """加载训练保存的 checkpoint，自动还原模型配置。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("cfg", {})
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
    cfg.patch_size = tuple(cfg.patch_size)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def evaluate_test_set(cfg, ckpt_path, out_dir, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg, ckpt = load_model(ckpt_path, device)
    samples = list_samples(cfg.proc_dir)
    _, _, test_s = split_by_patient(samples, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    assert test_s, "测试集为空，请检查数据划分。"
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for img_path, lab_path, cid in test_s:
        image = np.asanyarray(nib.load(img_path).dataobj).astype(np.float32)
        gt = np.asanyarray(nib.load(lab_path).dataobj).astype(np.uint8)
        pred = sliding_window_inference(model, image, model_cfg.patch_size,
                                        model_cfg.num_classes, overlap=cfg.sw_overlap,
                                        device=device)
        pred = keep_largest_component(pred, label=1)
        # 保存预测结果，便于 3D Slicer 打开检查
        case_dir = os.path.join(out_dir, cid)
        os.makedirs(case_dir, exist_ok=True)
        ref = nib.load(img_path)
        nib.save(nib.Nifti1Image(pred.astype(np.uint8), ref.affine),
                 os.path.join(case_dir, "pred.nii.gz"))
        m = evaluate_case(pred, gt, spacing=cfg.target_spacing,
                          labels=range(1, model_cfg.num_classes))
        row = {"case": cid}
        for c, mm in m.items():
            for k, v in mm.items():
                row[f"class{c}_{k}"] = v
        rows.append(row)
        print(f"[{cid}] " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "case"))

    # 汇总统计
    keys = [k for k in rows[0] if k != "case"]
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        summary[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
                      "n": int(np.sum(~np.isnan(vals)))}
    print("\n===== 测试集汇总（mean ± std）=====")
    for k, s in summary.items():
        print(f"  {k}: {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")

    with open(os.path.join(out_dir, "test_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case"] + keys)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, "test_metrics_summary.json"), "w") as f:
        json.dump({"per_case": rows, "summary": summary,
                   "checkpoint": os.path.basename(ckpt_path),
                   "ckpt_epoch": ckpt.get("epoch"), "ckpt_val_dice": ckpt.get("val_dice")},
                  f, indent=2)
    print(f"\n结果已保存: {out_dir}/test_metrics.csv, test_metrics_summary.json")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/test_metrics")
    ap.add_argument("--proc_dir", default=None, help="覆盖预处理数据目录（默认取配置）")
    args = ap.parse_args()
    cfg = CFG
    if args.proc_dir:
        from dataclasses import replace
        cfg = replace(CFG, proc_dir=args.proc_dir)
    evaluate_test_set(cfg, args.ckpt, args.out)
