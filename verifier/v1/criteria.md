# 验收标准 v1

针对“TMJ CBCT 3D U-Net 分割项目”的代码正确性与流程完整性验收。

## A. 静态检查
- A1 所有 `.py` 文件通过 `py_compile` 编译，无语法错误。
- A2 包结构完整：`src/`、`src/data/`、`src/models/` 均有 `__init__.py`。

## B. 单元级检查
- B1 `UNet3D`：输入 (1,1,64,64,64) 输出 (1,num_classes,64,64,64)，参数量 > 0。
- B2 `DiceCELoss`：完美预测的损失显著低于随机预测；全背景标签输入不产生 NaN/Inf。
- B3 预处理：
  - `resample_nifti` 标注使用最近邻插值，输出标签只含离散整数值；
  - `normalize_intensity` 输出均值≈0、标准差≈1；
  - `bbox_from_mask` + `crop_to_bbox` 能正确裁出前景区域；
  - `process_case` 端到端输出 ROI 形状 ≥ 设定 roi_size。
- B4 `sliding_window_inference`：输出形状与输入一致；`_gaussian_weight` 最小值 > 0 且中心权重大于边角。
- B5 `split_by_patient`：同一患者的左右侧样本不会分入不同子集。

## C. 端到端检查
- C1 `scripts/run_smoke_test.py` 完整跑通（数据生成→训练→推理→评估→可视化）。
- C2 最优验证 Dice ≥ 0.85。
- C3 测试集全图 Dice（滑动窗口+后处理）≥ 0.80。
- C4 生成可视化 PNG 与训练日志 train_log.json。

## D. 仓库一致性
- D1 GitHub 仓库 `Li920603/TMJ-CBCT-Segmentation` main 分支包含全部源码文件。
- D2 本次修复后的代码全部推送到 GitHub，本地与远端一致。
