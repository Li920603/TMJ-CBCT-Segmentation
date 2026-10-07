# -*- coding: utf-8 -*-
"""
合成数据生成器：用于在没有真实数据时验证整个训练闭环（冒烟测试）。
模拟 CBCT 中 TMJ 的大致形态——球形髁突 + 上方杯状关节窝，加入平滑噪声、
干扰软组织与灰度不均场。
"""

import numpy as np
import nibabel as nib
import os


def make_synthetic_case(shape=(96, 96, 96), seed=0):
    rng = np.random.RandomState(seed)
    # 背景：低通平滑噪声（更接近真实 CBCT 的软组织/空气背景，而非白噪声）
    from scipy import ndimage
    img = ndimage.gaussian_filter(rng.normal(0, 1.0, shape), sigma=2.0).astype(np.float32)
    img *= 0.3
    lab = np.zeros(shape, dtype=np.uint8)

    # 干扰组织：2~3 个中等灰度的平滑椭球（模拟邻近软组织，增加任务真实性）
    for _ in range(rng.randint(2, 4)):
        bz = [rng.randint(15, s - 15) for s in shape]
        radii = [rng.randint(6, 12) for _ in range(3)]
        zz, yy, xx = np.ogrid[tuple(slice(0, s) for s in shape)]
        blob = ((zz - bz[0]) / radii[0]) ** 2 + ((yy - bz[1]) / radii[1]) ** 2 \
               + ((xx - bz[2]) / radii[2]) ** 2 <= 1
        img[blob] += rng.uniform(0.5, 1.0)

    cz = [rng.randint(30, s - 30) for s in shape]
    r = rng.randint(12, 16)                                     # 髁突半径
    zz, yy, xx = np.ogrid[tuple(slice(0, s) for s in shape)]
    condyle = (zz - cz[0]) ** 2 + (yy - cz[1]) ** 2 + (xx - cz[2]) ** 2 <= r ** 2
    lab[condyle] = 1
    img[condyle] += rng.uniform(2.0, 3.0)                        # 骨组织高灰度

    # 关节窝：髁突上方的薄壳
    fossa = ((zz - (cz[0] - r - 3)) ** 2 + (yy - cz[1]) ** 2 + (xx - cz[2]) ** 2 <= (r + 4) ** 2) \
            & (zz < cz[0] - r + 2)
    lab[fossa] = 1
    img[fossa] += rng.uniform(2.0, 3.0)

    # 灰度不均匀场
    bias = np.sin(np.linspace(0, 3, shape[0]))[:, None, None] * 0.3
    img += bias.astype(np.float32)
    return img, lab


def generate_synthetic_dataset(out_dir, n_cases=6, shape=(96, 96, 96), seed=42):
    for i in range(n_cases):
        img, lab = make_synthetic_case(shape, seed + i)
        d = os.path.join(out_dir, f"synthetic_{i:02d}")
        os.makedirs(d, exist_ok=True)
        aff = np.eye(4)
        nib.save(nib.Nifti1Image(img, aff), os.path.join(d, "image.nii.gz"))
        nib.save(nib.Nifti1Image(lab, aff), os.path.join(d, "label.nii.gz"))
    print(f"已生成 {n_cases} 例合成数据 -> {out_dir}")


if __name__ == "__main__":
    generate_synthetic_dataset("data/processed")
