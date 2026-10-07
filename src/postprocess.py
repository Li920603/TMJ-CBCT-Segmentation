# -*- coding: utf-8 -*-
"""后处理：保留最大连通域，清除散落的假阳性体素（医学分割标准后处理）。"""

import numpy as np
from scipy import ndimage


def keep_largest_component(pred: np.ndarray, label: int = 1) -> np.ndarray:
    """对指定类别保留最大连通域，其余该类别体素置 0。"""
    mask = pred == label
    if mask.sum() == 0:
        return pred
    cc, n = ndimage.label(mask)
    if n <= 1:
        return pred
    sizes = ndimage.sum(mask, cc, index=range(1, n + 1))
    largest = np.argmax(sizes) + 1
    out = pred.copy()
    out[mask & (cc != largest)] = 0
    return out


def remove_small_objects(pred: np.ndarray, label: int = 1, min_size: int = 100) -> np.ndarray:
    """移除小于 min_size 体素的连通区域。"""
    mask = pred == label
    if mask.sum() == 0:
        return pred
    cc, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, cc, index=range(1, n + 1))
    out = pred.copy()
    for i, s in enumerate(sizes, start=1):
        if s < min_size:
            out[cc == i] = 0
    return out
