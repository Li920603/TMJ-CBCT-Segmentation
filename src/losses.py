# -*- coding: utf-8 -*-
"""损失函数：Dice Loss + 交叉熵 混合损失（应对前景/背景体素极度不平衡）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """多分类 soft Dice。对 softmax 概率与 one-hot 标签逐类计算。"""

    def __init__(self, smooth=1e-5, include_background=False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, target):
        # logits: (N, C, D, H, W); target: (N, D, H, W)
        probs = F.softmax(logits, dim=1)
        target_1h = F.one_hot(target, logits.shape[1])          # (N,D,H,W,C)
        target_1h = target_1h.permute(0, 4, 1, 2, 3).float()    # (N,C,D,H,W)
        dims = (0, 2, 3, 4)
        inter = (probs * target_1h).sum(dims)
        denom = probs.sum(dims) + target_1h.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        if not self.include_background:
            dice = dice[1:]     # 前景/背景不平衡时，不计背景类
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """开题报告指定的混合损失：L = w * Dice + (1-w) * CE"""

    def __init__(self, num_classes=2, dice_weight=0.5):
        super().__init__()
        self.dice = SoftDiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.w = dice_weight

    def forward(self, logits, target):
        return self.w * self.dice(logits, target) + (1 - self.w) * self.ce(logits, target)
