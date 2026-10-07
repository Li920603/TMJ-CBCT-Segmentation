# -*- coding: utf-8 -*-
"""PyTorch 数据集：加载预处理后的 ROI，支持 patch 采样与在线数据增强。"""

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy import ndimage


def list_samples(proc_dir):
    """扫描预处理目录，返回 (image_path, label_path, case_id) 列表。"""
    samples = []
    for d in sorted(os.listdir(proc_dir)):
        img = os.path.join(proc_dir, d, "image.nii.gz")
        lab = os.path.join(proc_dir, d, "label.nii.gz")
        if os.path.exists(img) and os.path.exists(lab):
            samples.append((img, lab, d))
    return samples


def split_by_patient(samples, val_ratio, test_ratio, seed):
    """按患者（而非按切片/按侧）划分，防止同一患者数据泄漏到训练与测试中。"""
    patients = {}
    for s in samples:
        cid = s[2]
        # 真实数据为 "M000123_L"/"M000123_R"（同一患者左右侧必须同组）；
        # 无 _L/_R 后缀的样本（如合成数据）自身即为一个独立患者
        pid = cid.rsplit("_", 1)[0] if cid.endswith(("_L", "_R")) else cid
        patients.setdefault(pid, []).append(s)
    rng = np.random.RandomState(seed)
    pids = sorted(patients.keys())
    rng.shuffle(pids)
    n = len(pids)
    n_test = max(int(round(n * test_ratio)), 1)
    n_val = max(int(round(n * val_ratio)), 1)
    test_p = set(pids[:n_test])
    val_p = set(pids[n_test:n_test + n_val])
    train, val, test = [], [], []
    for pid, ss in patients.items():
        if pid in test_p:
            test.extend(ss)
        elif pid in val_p:
            val.extend(ss)
        else:
            train.extend(ss)
    return train, val, test


class TMJDataset(Dataset):
    """每个 __getitem__ 返回一个 patch：(1, D, H, W) 影像 与 (D, H, W) 标签。"""

    def __init__(self, samples, patch_size, pos_sample_prob=0.7, augment=False,
                 cache=True, iters_per_epoch=1):
        self.samples = samples
        self.patch_size = tuple(patch_size)
        self.pos_prob = pos_sample_prob
        self.augment = augment
        self.cache = cache
        self.iters = iters_per_epoch          # 每例样本每轮被随机采 patch 的次数
        self._cache = {}

    def _load(self, i):
        i = i % len(self.samples)             # iters_per_epoch > 1 时循环复用样本
        if self.cache and i in self._cache:
            return self._cache[i]
        img_path, lab_path, _ = self.samples[i]
        img = np.asanyarray(nib.load(img_path).dataobj).astype(np.float32)
        lab = np.asanyarray(nib.load(lab_path).dataobj).astype(np.uint8)
        if self.cache:
            self._cache[i] = (img, lab)
        return img, lab

    def __len__(self):
        return len(self.samples) * self.iters

    # ---------------- patch 采样 ----------------
    def _sample_patch(self, img, lab):
        ps = self.patch_size
        shape = img.shape
        if np.random.rand() < self.pos_prob and lab.sum() > 0:
            # 正样本过采样：随机选一个前景体素，以它为中心裁 patch
            fg = np.argwhere(lab > 0)
            center = fg[np.random.randint(len(fg))]
        else:
            center = [np.random.randint(s) for s in shape]
        lo = [int(np.clip(c - p // 2, 0, max(s - p, 0)))
              for c, p, s in zip(center, ps, shape)]
        sl = [slice(l, min(l + p, s)) for l, p, s in zip(lo, ps, shape)]
        img_p, lab_p = img[tuple(sl)], lab[tuple(sl)]
        # 边界不足时补零
        if img_p.shape != ps:
            pad = [(0, p - s) for p, s in zip(ps, img_p.shape)]
            img_p = np.pad(img_p, pad, constant_values=float(img_p.min()))
            lab_p = np.pad(lab_p, pad, constant_values=0)
        return img_p, lab_p

    # ---------------- 在线数据增强 ----------------
    def _augment(self, img, lab):
        # 1) 随机翻转（三个轴独立）
        for ax in range(3):
            if np.random.rand() < 0.5:
                img = np.flip(img, ax)
                lab = np.flip(lab, ax)
        # 2) 随机 90° 整数倍旋转（任选两个轴构成的平面）
        if np.random.rand() < 0.75:
            axes = tuple(np.random.choice(3, 2, replace=False))
            k = np.random.randint(0, 4)
            img = np.rot90(img, k, axes)
            lab = np.rot90(lab, k, axes)
        # 3) 小角度旋转（仅影像用线性插值，标签最近邻）
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-15, 15)
            axes = tuple(np.random.choice(3, 2, replace=False))
            img = ndimage.rotate(img, angle, axes=axes, reshape=False, order=1, mode="nearest")
            lab = ndimage.rotate(lab, angle, axes=axes, reshape=False, order=0, mode="nearest")
        # 4) 强度扰动：亮度 / 对比度
        if np.random.rand() < 0.5:
            img = img * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.1, 0.1)
        # 5) 高斯噪声
        if np.random.rand() < 0.3:
            img = img + np.random.normal(0, 0.05, img.shape).astype(np.float32)
        return np.ascontiguousarray(img), np.ascontiguousarray(lab)

    def __getitem__(self, i):
        img, lab = self._load(i)
        img_p, lab_p = self._sample_patch(img, lab)
        if self.augment:
            img_p, lab_p = self._augment(img_p, lab_p)
        img_t = torch.from_numpy(img_p[None].astype(np.float32))   # (1, D, H, W)
        lab_t = torch.from_numpy(lab_p.astype(np.int64))           # (D, H, W)
        return img_t, lab_t
