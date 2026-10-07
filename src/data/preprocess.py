# -*- coding: utf-8 -*-
"""
数据预处理管道 —— 对应开题报告要求的四步：
  1) 方向标准化：统一到 RAS 方向
  2) 体素重采样：统一到 target_spacing（影像三线性插值 / 标注最近邻插值）
  3) 强度归一化：骨组织窗截断 + z-score 标准化
  4) ROI 提取：以标注包围盒为中心裁剪感兴趣区域

依赖仅 numpy / scipy / nibabel，CPU 即可运行。
"""

import os
import numpy as np
import nibabel as nib
from scipy import ndimage


# ---------------------------------------------------------------- 方向标准化
def to_canonical(nii: nib.Nifti1Image) -> nib.Nifti1Image:
    """把影像重定向到最接近的 RAS 标准方向，消除 CBCT 头朝向差异。"""
    return nib.as_closest_canonical(nii)


# ---------------------------------------------------------------- 体素重采样
def resample_image(img: np.ndarray, old_spacing, new_spacing, order: int) -> np.ndarray:
    """
    按体素间距比例缩放体数据。
    img: (D, H, W) 的 numpy 数组（nibabel get_fdata 转置后）
    order: 影像用 1（三线性），标注必须用 0（最近邻）——否则标签会被插值污染！
    """
    zoom_factors = [o / n for o, n in zip(old_spacing, new_spacing)]
    resampled = ndimage.zoom(img, zoom_factors, order=order)
    return resampled


def resample_nifti(nii: nib.Nifti1Image, new_spacing, is_label: bool) -> nib.Nifti1Image:
    data = np.asanyarray(nii.dataobj)
    old_spacing = nii.header.get_zooms()[:3]          # (x, y, z)
    # nibabel 数组轴顺序为 (x, y, z)，直接逐轴缩放即可
    zoom_factors = [o / n for o, n in zip(old_spacing, new_spacing)]
    new_data = ndimage.zoom(data, zoom_factors, order=0 if is_label else 1)
    if is_label:
        new_data = np.round(new_data).astype(np.uint8)
    else:
        new_data = new_data.astype(np.float32)
    new_affine = nii.affine.copy()
    scale = np.diag([n / o for o, n in zip(old_spacing, new_spacing)] + [1.0])
    new_affine = new_affine @ scale
    return nib.Nifti1Image(new_data, new_affine)


# ---------------------------------------------------------------- 强度归一化
def normalize_intensity(img: np.ndarray, clip_range) -> np.ndarray:
    """窗宽窗位截断 + z-score 标准化（统计量在全图上计算）。"""
    img = np.clip(img, clip_range[0], clip_range[1]).astype(np.float32)
    mean, std = img.mean(), img.std()
    if std < 1e-6:
        std = 1.0
    return (img - mean) / std


# ---------------------------------------------------------------- ROI 提取
def bbox_from_mask(mask: np.ndarray, margin: int = 0):
    """计算前景包围盒并外扩 margin，裁剪到图像范围内。"""
    idx = np.where(mask > 0)
    if len(idx[0]) == 0:
        raise ValueError("标注为空，无法提取 ROI")
    lo = [max(int(i.min()) - margin, 0) for i in idx]
    hi = [min(int(i.max()) + margin + 1, s) for i, s in zip(idx, mask.shape)]
    return lo, hi


def crop_to_bbox(img: np.ndarray, lo, hi) -> np.ndarray:
    return img[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


def pad_to_size(arr: np.ndarray, target, pad_value=0) -> np.ndarray:
    """把数组对称 padding 到至少 target 尺寸。"""
    pads = []
    for s, t in zip(arr.shape, target):
        total = max(t - s, 0)
        pads.append((total // 2, total - total // 2))
    return np.pad(arr, pads, mode="constant", constant_values=pad_value)


# ---------------------------------------------------------------- 单例处理
def process_case(image_path, label_path, out_image_path, out_label_path, cfg):
    """处理单例（一侧 TMJ）：方向标准化→重采样→归一化→ROI 裁剪→保存。"""
    img_nii = to_canonical(nib.load(image_path))
    lab_nii = to_canonical(nib.load(label_path))

    img_nii = resample_nifti(img_nii, cfg.target_spacing, is_label=False)
    lab_nii = resample_nifti(lab_nii, cfg.target_spacing, is_label=True)

    img = np.asanyarray(img_nii.dataobj).astype(np.float32)
    lab = (np.asanyarray(lab_nii.dataobj) > 0).astype(np.uint8)  # 二值化：>0 即为关节

    img = normalize_intensity(img, cfg.clip_range)

    lo, hi = bbox_from_mask(lab, margin=cfg.roi_margin)
    img = crop_to_bbox(img, lo, hi)
    lab = crop_to_bbox(lab, lo, hi)

    img = pad_to_size(img, cfg.roi_size, pad_value=float(img.min()))
    lab = pad_to_size(lab, cfg.roi_size, pad_value=0)

    os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
    nib.save(nib.Nifti1Image(img.astype(np.float32), img_nii.affine), out_image_path)
    nib.save(nib.Nifti1Image(lab.astype(np.uint8), lab_nii.affine), out_label_path)
    return img.shape


def run_preprocessing(cfg):
    """遍历 raw_dir 下所有患者，把左右侧 TMJ 各存为一个样本。"""
    cases = sorted(d for d in os.listdir(cfg.raw_dir)
                   if os.path.isdir(os.path.join(cfg.raw_dir, d)))
    print(f"共发现 {len(cases)} 例患者")
    for case in cases:
        case_dir = os.path.join(cfg.raw_dir, case)
        image_path = os.path.join(case_dir, cfg.image_name)
        if not os.path.exists(image_path):
            print(f"  [跳过] {case}: 未找到 {cfg.image_name}（需先运行 dicom_to_nifti.py）")
            continue
        for suffix, side in zip(cfg.label_suffixes, ["L", "R"]):
            label_path = os.path.join(case_dir, f"{case}{suffix}")
            if not os.path.exists(label_path):
                print(f"  [跳过] {case}-{side}: 标注不存在")
                continue
            out_img = os.path.join(cfg.proc_dir, f"{case}_{side}", "image.nii.gz")
            out_lab = os.path.join(cfg.proc_dir, f"{case}_{side}", "label.nii.gz")
            shape = process_case(image_path, label_path, out_img, out_lab, cfg)
            print(f"  [完成] {case}_{side} -> ROI shape {shape}")
    print("预处理全部完成。")


if __name__ == "__main__":
    from src.config import CFG
    run_preprocessing(CFG)
