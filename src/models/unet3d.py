# -*- coding: utf-8 -*-
"""
3D U-Net：编码器-解码器 + 跳跃连接（开题报告指定架构）。
- 双卷积块（Conv3d + Norm + LeakyReLU）×2
- 下采样：stride=2 卷积；上采样：三线性插值 + 1×1×1 卷积
- 深层与浅层通过跳跃连接融合，兼顾全局语义与细节边界
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_layer(kind, ch):
    """归一化层：instance=nnU-Net 风格；batch=训练/推理统计一致，小数据下显著更稳。"""
    if kind == "batch":
        return nn.BatchNorm3d(ch, affine=True)
    return nn.InstanceNorm3d(ch, affine=True)


class ConvBlock(nn.Module):
    """(Conv -> Norm -> LeakyReLU) × 2"""

    def __init__(self, in_ch, out_ch, norm="instance"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            _norm_layer(norm, out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            _norm_layer(norm, out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm="instance"):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.norm = _norm_layer(norm, out_ch)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.conv = ConvBlock(out_ch, out_ch, norm)

    def forward(self, x):
        return self.conv(self.act(self.norm(self.down(x))))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, norm="instance"):
        super().__init__()
        self.reduce = nn.Conv3d(in_ch, out_ch, 1, bias=False)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, norm)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)      # 跳跃连接：融合高层语义与浅层细节
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base_features=32, num_levels=4,
                 norm="instance"):
        super().__init__()
        f = base_features
        # 编码器
        self.encoders = nn.ModuleList()
        ch = in_channels
        feats = []
        for i in range(num_levels):
            out_f = f * (2 ** i)
            if i == 0:
                self.encoders.append(ConvBlock(ch, out_f, norm))
            else:
                self.encoders.append(Down(ch, out_f, norm))
            ch = out_f
            feats.append(out_f)
        # 解码器（自底向上）
        self.decoders = nn.ModuleList()
        for i in range(num_levels - 2, -1, -1):
            self.decoders.append(Up(feats[i + 1], feats[i], feats[i], norm))
        self.head = nn.Conv3d(feats[0], num_classes, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 2)])
        return self.head(x)


def build_model(cfg) -> nn.Module:
    return UNet3D(cfg.in_channels, cfg.num_classes, cfg.base_features, cfg.num_levels,
                  getattr(cfg, "norm", "instance"))


if __name__ == "__main__":
    m = UNet3D(1, 2, 32, 4)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"3D U-Net 参数量: {n_params/1e6:.2f} M")
    x = torch.randn(1, 1, 64, 64, 64)
    y = m(x)
    print("输入", tuple(x.shape), "-> 输出", tuple(y.shape))
