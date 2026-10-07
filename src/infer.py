# -*- coding: utf-8 -*-
"""滑动窗口推理：对完整 ROI 体数据分块预测，重叠区域高斯加权平均，拼回整体。"""

import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_weight(patch_size):
    """
    生成高斯重要性图，降低 patch 边缘权重，减少拼接伪影。
    以体素为单位计算，sigma = patch/8（与 nnU-Net 一致），
    并设置下限，保证 50% 重叠时每个体素都能获得有效权重。
    """
    coords = [np.arange(s) - (s - 1) / 2.0 for s in patch_size]
    sigmas = [s / 8.0 for s in patch_size]
    w = np.ones(patch_size, dtype=np.float32)
    for ax, (c, sig) in enumerate(zip(coords, sigmas)):
        shape = [1] * 3
        shape[ax] = len(c)
        w = w * np.exp(-(c.reshape(shape) ** 2) / (2 * sig ** 2))
    return np.clip(w, 1e-3, None).astype(np.float32)


def _iter_windows(shape, patch, overlap):
    stride = [max(int(p * (1 - overlap)), 1) for p in patch]
    starts = []
    for s, p, st in zip(shape, patch, stride):
        if s <= p:
            starts.append([0])
        else:
            xs = list(range(0, s - p + 1, st))
            if xs[-1] != s - p:
                xs.append(s - p)
            starts.append(xs)
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                yield (z, y, x)


@torch.no_grad()
def sliding_window_inference(model, image, patch_size, num_classes, overlap=0.5, device="cpu"):
    """
    image: (D, H, W) numpy 体数据（已归一化）
    返回: (D, H, W) 预测标签
    """
    model.eval()
    patch = tuple(patch_size)
    shape = image.shape
    # 不足 patch 尺寸时先 padding
    pad = [(0, max(p - s, 0)) for p, s in zip(patch, shape)]
    img_pad = np.pad(image, pad, constant_values=float(image.min()))
    out_shape = img_pad.shape

    prob_sum = np.zeros((num_classes,) + out_shape, dtype=np.float32)
    weight_sum = np.zeros(out_shape, dtype=np.float32)
    gw = _gaussian_weight(patch)

    for z, y, x in _iter_windows(out_shape, patch, overlap):
        sl = np.s_[z:z + patch[0], y:y + patch[1], x:x + patch[2]]
        tile = torch.from_numpy(img_pad[sl][None, None]).float().to(device)
        logits = model(tile)
        probs = F.softmax(logits.float(), dim=1)[0].cpu().numpy()
        prob_sum[(slice(None),) + sl] += probs * gw
        weight_sum[sl] += gw

    pred = prob_sum.argmax(0)
    # 去除 padding
    unp = [slice(0, s) for s in shape]
    return pred[tuple(unp)].astype(np.uint8)
