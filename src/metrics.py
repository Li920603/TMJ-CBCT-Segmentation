# -*- coding: utf-8 -*-
"""评估指标：Dice / IoU / HD95 / ASD（表面距离基于 scipy 距离变换实现）。"""

import numpy as np
from scipy import ndimage


def dice_coefficient(pred, gt, label=1):
    p, g = pred == label, gt == label
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return 2.0 * (p & g).sum() / denom


def iou_score(pred, gt, label=1):
    p, g = pred == label, gt == label
    union = (p | g).sum()
    if union == 0:
        return 1.0
    return (p & g).sum() / union


def _surface_points(mask):
    """提取表面体素：mask 与其腐蚀结果的差集。"""
    eroded = ndimage.binary_erosion(mask, border_value=0)
    return mask & ~eroded


def surface_distances(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """返回 pred 表面到 gt 表面的所有最近距离（mm），对称双向。"""
    if pred.sum() == 0 or gt.sum() == 0:
        return None
    s_pred = _surface_points(pred)
    s_gt = _surface_points(gt)
    # 距离变换（以体素间距为单位），在 pred 表面采样
    dt_gt = ndimage.distance_transform_edt(~s_gt, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~s_pred, sampling=spacing)
    d_p2g = dt_gt[s_pred]
    d_g2p = dt_pred[s_gt]
    return np.concatenate([d_p2g, d_g2p])


def hausdorff95(pred, gt, label=1, spacing=(1.0, 1.0, 1.0)):
    d = surface_distances(pred == label, gt == label, spacing)
    if d is None:
        return np.nan
    return float(np.percentile(d, 95))


def asd_score(pred, gt, label=1, spacing=(1.0, 1.0, 1.0)):
    d = surface_distances(pred == label, gt == label, spacing)
    if d is None:
        return np.nan
    return float(d.mean())


def evaluate_case(pred, gt, spacing=(1.0, 1.0, 1.0), labels=(1,)):
    """单例评估，返回 {class: {dice, iou, hd95, asd}} 字典。"""
    result = {}
    for c in labels:
        result[c] = {
            "dice": float(dice_coefficient(pred, gt, c)),
            "iou": float(iou_score(pred, gt, c)),
            "hd95": hausdorff95(pred, gt, c, spacing),
            "asd": asd_score(pred, gt, c, spacing),
        }
    return result
