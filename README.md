# TMJ-CBCT-Segmentation

基于 3D U-Net 的 CBCT 颞下颌关节（Temporomandibular Joint, TMJ）自动分割 —— 毕业设计项目。

**题目**：Automatic Segmentation of Temporomandibular Joint from CBCT Images Based on 3D U-Net
**目标**：输入 3D CBCT 体数据，输出髁突与关节窝的体素级分割掩膜。

## 方法概述

- **预处理**：方向标准化（RAS）→ 体素重采样（0.4mm 各向同性，影像三线性/标注最近邻）→ 强度截断 [-1000, 2000] + z-score 归一化 → 以标注为中心的 ROI 裁剪
- **模型**：3D U-Net（编码器-解码器 + 跳跃连接，BatchNorm + LeakyReLU）
- **损失**：Dice + 交叉熵混合损失（应对前景/背景体素不平衡）
- **训练**：patch-based 训练、前景过采样、随机翻转/旋转/噪声增强、AdamW + 余弦退火、混合精度
- **推理**：滑动窗口 + 高斯加权融合；最大连通域后处理
- **评估**：Dice / IoU / HD95 / ASD，按患者划分数据集防止泄漏

## 目录结构

```
├── src/
│   ├── config.py            # 全部超参数
│   ├── data/
│   │   ├── download_dataset.py   # 下载公开 TMJ 数据集
│   │   ├── dicom_to_nifti.py     # DICOM 序列 -> NIfTI 体数据
│   │   ├── preprocess.py         # 预处理四步管道
│   │   ├── dataset.py            # Dataset + patch 采样 + 数据增强 + 按患者划分
│   │   └── synthetic.py          # 合成数据（冒烟测试用）
│   ├── models/unet3d.py     # 3D U-Net
│   ├── losses.py            # Dice + CE 混合损失
│   ├── metrics.py           # Dice / IoU / HD95 / ASD
│   ├── postprocess.py       # 最大连通域后处理
│   ├── train.py             # 训练入口
│   ├── infer.py             # 滑动窗口推理
│   └── visualize.py         # 三视图叠加对比图
├── scripts/run_smoke_test.py   # 冒烟测试（合成数据全链路自检）
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt

# 0. 冒烟测试：先验证全链路（CPU 几分钟跑完，真实训练前必做）
python scripts/run_smoke_test.py

# 1. 下载数据集（47 例 TMJ CBCT + 左右侧标注）
python -m src.data.download_dataset --out data/raw

# 2. DICOM -> NIfTI
python -m src.data.dicom_to_nifti

# 3. 预处理
python -m src.data.preprocess

# 4. 训练（建议 GPU ≥12GB 显存）
python -m src.train
```

## 数据集

使用公开数据集 [Henryblankwu/TMJ-CBCT-Segmentation-Data](https://github.com/Henryblankwu/TMJ-CBCT-Segmentation-Data)（47 例 CBCT，左右侧 TMJ 的 NIfTI 标注，放射科医生标注、已脱敏，仅供非商业研究用途）。

## 硬件建议

| 用途 | 配置 |
|---|---|
| 预处理 / 评估 / 可视化 | 普通电脑（CPU，16GB 内存） |
| 训练 | GPU 显存 ≥ 12GB（patch 64³）；≥ 24GB 可跑 96³ patch |

## 冒烟测试结果（合成数据，CPU）

6 例合成体数据（按患者 4/1/1 划分），小模型（base=8, 3 层）训练 20 epoch：

| 指标 | 结果 |
|---|---|
| 验证 Dice | **0.9742** |
| 测试集全图 Dice（滑动窗口+最大连通域后处理） | **0.9452** |
| IoU | 0.8961 |
| HD95 | 0.57 mm |
| ASD | 0.26 mm |

### 开发中发现的关键问题（值得写进论文讨论）

1. **归一化层的选择**：InstanceNorm 会对每个推理窗口独立归一化，把纯背景噪声窗口放大成"类骨组织"纹理，导致全图推理时大量假阳性（Dice 仅 0.11）；换成 BatchNorm（训练/推理统计一致）后同配置达到 0.94。
2. **每轮迭代数**：patch 训练时每轮必须采足够多的 patch（本项目 `iters_per_epoch=250`），仅按"每例一次"会导致梯度更新不足。
3. **数据集划分必须按患者**：同一患者的左右侧关节高度相似，混入训练/测试两侧会使指标虚高。

真实数据上的正式结果见论文实验章节。
