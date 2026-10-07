# -*- coding: utf-8 -*-
"""
DICOM 序列 -> NIfTI 体数据转换。
每例患者文件夹中包含 Slice_XXXX.dcm 切片与 slice.index（完整切片顺序索引）。
依据 DICOM 头中的像素间距 / 层间距构建仿射矩阵。
需要: pip install pydicom
"""

import os
import re
import numpy as np
import nibabel as nib

try:
    import pydicom
except ImportError:
    pydicom = None


def load_dicom_series(case_dir):
    """读取一个患者文件夹中的全部 DICOM，按实例号排序并堆叠为 3D 体数据。"""
    assert pydicom is not None, "请先安装 pydicom: pip install pydicom"
    files = [f for f in os.listdir(case_dir) if f.lower().endswith(".dcm")]
    files.sort(key=lambda f: int(re.search(r"(\d+)", f).group(1)))
    assert files, f"{case_dir} 中没有 DICOM 文件"
    slices = [pydicom.dcmread(os.path.join(case_dir, f)) for f in files]

    first = slices[0]
    px = [float(v) for v in first.PixelSpacing]                 # [row, col] (mm)
    thick = float(getattr(first, "SliceThickness", px[0]))      # 层厚 (mm)
    vol = np.stack([s.pixel_array for s in slices], axis=0)     # (D, H, W)
    vol = vol.astype(np.float32)
    # 应用 rescale（CBCT 常见）
    slope = float(getattr(first, "RescaleSlope", 1.0))
    intercept = float(getattr(first, "RescaleIntercept", 0.0))
    vol = vol * slope + intercept

    # nibabel 约定 (x, y, z)，numpy 为 (z, y, x) -> 转置
    vol_xyz = np.transpose(vol, (2, 1, 0))
    affine = np.diag([px[1], px[0], thick, 1.0])
    return nib.Nifti1Image(vol_xyz, affine)


def run_dicom_to_nifti(raw_dir, image_name="image.nii.gz"):
    cases = sorted(d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d)))
    for case in cases:
        case_dir = os.path.join(raw_dir, case)
        out = os.path.join(case_dir, image_name)
        if os.path.exists(out):
            continue
        try:
            nii = load_dicom_series(case_dir)
            nib.save(nii, out)
            print(f"[完成] {case}: shape={nii.shape} spacing={nii.header.get_zooms()[:3]}")
        except Exception as e:
            print(f"[失败] {case}: {e}")


if __name__ == "__main__":
    run_dicom_to_nifti("data/raw")
