# -*- coding: utf-8 -*-
"""结果可视化：三视图（轴位/冠状/矢状）预测叠加对比图，论文素材。"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def overlay_slice(img2d, mask2d, color="Reds", alpha=0.4):
    """灰度图上叠加分割掩膜轮廓。"""
    plt.imshow(img2d, cmap="gray")
    if mask2d.sum() > 0:
        plt.contour(mask2d, levels=[0.5], colors="r", linewidths=1.0)


def visualize_case(image, gt, pred, out_path, title="TMJ Segmentation"):
    """image/gt/pred: (D, H, W)。输出 2×3 对比图：GT 与预测各一行的三视图。"""
    c = [s // 2 for s in image.shape]
    views = [
        (image[c[0]], gt[c[0]], pred[c[0]], "Axial"),
        (image[:, c[1]], gt[:, c[1]], pred[:, c[1]], "Coronal"),
        (image[:, :, c[2]], gt[:, :, c[2]], pred[:, :, c[2]], "Sagittal"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for j, (im, g, p, name) in enumerate(views):
        plt.sca(axes[0, j]); overlay_slice(im.T, g.T); plt.title(f"GT - {name}"); plt.axis("off")
        plt.sca(axes[1, j]); overlay_slice(im.T, p.T); plt.title(f"Pred - {name}"); plt.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
