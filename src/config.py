# -*- coding: utf-8 -*-
"""全局配置：所有实验参数集中在这里管理。"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---------- 数据 ----------
    raw_dir: str = "data/raw"                 # 原始数据根目录（每例患者一个子文件夹）
    proc_dir: str = "data/processed"          # 预处理后输出目录
    image_name: str = "image.nii.gz"          # 每例患者的 CBCT 体数据文件名（由 DICOM 转换而来）
    label_suffixes: tuple = ("-L.nii.gz", "-R.nii.gz")  # 左右侧 TMJ 标注文件名后缀
    num_classes: int = 2                      # 0=背景, 1=TMJ（髁突+关节窝合并标注时=2类）

    # ---------- 预处理 ----------
    target_spacing: tuple = (0.4, 0.4, 0.4)   # 重采样目标体素间距 (mm, z/y/x 顺序对应 numpy 轴)
    clip_range: tuple = (-1000.0, 2000.0)     # CBCT 灰度截断范围（骨组织窗）
    roi_size: tuple = (96, 96, 96)            # 以标注为中心裁剪的 ROI 尺寸 (voxel)
    roi_margin: int = 16                      # 标注包围盒外扩边距 (voxel)

    # ---------- 模型 ----------
    in_channels: int = 1
    base_features: int = 32                   # U-Net 首层特征通道数（显存不足可降到 16）
    num_levels: int = 4                       # 下采样层数
    norm: str = "batch"                       # 归一化：batch（本项目默认，小数据下显著更稳）或 instance（nnU-Net 风格）

    # ---------- 训练 ----------
    patch_size: tuple = (96, 96, 96)          # 训练 patch 尺寸（≤12GB 显存建议 64³）
    batch_size: int = 2
    iters_per_epoch: int = 250                # 每轮迭代次数（patch 训练的关键：每例每轮采多个 patch）
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-5
    pos_sample_prob: float = 0.7              # 以前景体素为中心采样的概率（正样本过采样）
    dice_weight: float = 0.5                  # 混合损失中 Dice 的权重，CE 权重 = 1 - dice_weight
    early_stop_patience: int = 30
    num_workers: int = 4
    amp: bool = True                          # 混合精度训练（仅 GPU 生效）

    # ---------- 数据划分（按患者！） ----------
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    # ---------- 推理 ----------
    sw_overlap: float = 0.5                   # 滑动窗口重叠率

    # ---------- 输出 ----------
    out_dir: str = "outputs"


CFG = Config()
