# -*- coding: utf-8 -*-
"""Verifier v1：静态检查 + 单元级验收。端到端冒烟测试由 run_smoke_test.py 单独执行。
用法: python verifier/v1/check_all.py   （在项目根目录运行；退出码 0 = 全部通过）
"""

import os
import sys
import tempfile
import py_compile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import nibabel as nib

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, "PASS", detail or ""))
        print(f"[PASS] {name} {detail or ''}")
    except Exception as e:
        RESULTS.append((name, "FAIL", str(e)))
        print(f"[FAIL] {name}: {e}")


# ---------------- A 静态检查 ----------------
def a1_compile():
    files = []
    for base in ("src", "scripts"):
        for dp, _, fns in os.walk(os.path.join(ROOT, base)):
            for fn in fns:
                if fn.endswith(".py"):
                    files.append(os.path.join(dp, fn))
    for f in files:
        py_compile.compile(f, doraise=True)
    return f"{len(files)} 个文件编译通过"


def a2_packages():
    for p in ("src/__init__.py", "src/data/__init__.py", "src/models/__init__.py"):
        assert os.path.exists(os.path.join(ROOT, p)), f"缺少 {p}"
    return "包结构完整"


# ---------------- B 单元检查 ----------------
def b1_model():
    from src.models.unet3d import UNet3D
    m = UNet3D(1, 2, base_features=8, num_levels=3, norm="batch")
    m.eval()
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 2, 32, 32, 32), f"输出形状错误 {y.shape}"
    n = sum(p.numel() for p in m.parameters())
    assert n > 0
    return f"输出形状正确, 参数量 {n/1e3:.0f}K"


def b2_loss():
    from src.losses import DiceCELoss
    crit = DiceCELoss(num_classes=2, dice_weight=0.5)
    target = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    target[:, 2:6, 2:6, 2:6] = 1
    perfect = torch.zeros(1, 2, 8, 8, 8)
    perfect[0, 1, 2:6, 2:6, 2:6] = 10.0
    perfect[0, 0, :, :, :] = 1.0
    random_logits = torch.randn(1, 2, 8, 8, 8)
    l_perfect = crit(perfect, target).item()
    l_random = crit(random_logits, target).item()
    assert l_perfect < l_random, f"完美预测损失 {l_perfect} 应小于随机 {l_random}"
    bg = torch.zeros(1, 8, 8, 8, dtype=torch.long)   # 全背景标签
    l_bg = crit(random_logits, bg).item()
    assert np.isfinite(l_bg), "全背景标签损失非有限值"
    return f"perfect={l_perfect:.4f} random={l_random:.4f} bg={l_bg:.4f}"


def b3_preprocess():
    from dataclasses import replace
    from src.config import CFG
    from src.data.preprocess import (resample_nifti, normalize_intensity,
                                     bbox_from_mask, crop_to_bbox, process_case)
    # 标注最近邻：值域保持离散
    lab = np.zeros((20, 20, 20), dtype=np.uint8)
    lab[5:15, 5:15, 5:15] = 1
    nii = nib.Nifti1Image(lab, np.eye(4))
    rs = resample_nifti(nii, (2.0, 2.0, 2.0), is_label=True)
    vals = np.unique(np.asanyarray(rs.dataobj))
    assert set(vals).issubset({0, 1}), f"标注出现非离散值 {vals}"
    # 归一化
    img = np.random.rand(16, 16, 16).astype(np.float32) * 2000 - 500
    norm = normalize_intensity(img, (-1000, 2000))
    assert abs(norm.mean()) < 1e-4 and abs(norm.std() - 1) < 1e-3, "z-score 统计异常"
    # 包围盒裁剪
    lo, hi = bbox_from_mask(lab, margin=2)
    cropped = crop_to_bbox(lab, lo, hi)
    assert cropped.sum() == lab.sum() and cropped.shape == (14, 14, 14), "ROI 裁剪错误"
    # 端到端 process_case（含形状对齐）
    with tempfile.TemporaryDirectory() as td:
        cfg = replace(CFG, target_spacing=(1.0, 1.0, 1.0), roi_size=(32, 32, 32), roi_margin=4)
        img_arr = np.random.rand(40, 40, 40).astype(np.float32) * 1000
        lab_arr = np.zeros((40, 40, 40), dtype=np.uint8)
        lab_arr[10:25, 10:25, 10:25] = 1
        ip = os.path.join(td, "image.nii.gz")
        lp = os.path.join(td, "label.nii.gz")
        nib.save(nib.Nifti1Image(img_arr, np.eye(4)), ip)
        nib.save(nib.Nifti1Image(lab_arr, np.eye(4)), lp)
        oi = os.path.join(td, "out", "image.nii.gz")
        ol = os.path.join(td, "out", "label.nii.gz")
        shape = process_case(ip, lp, oi, ol, cfg)
        assert min(shape) >= 32, f"ROI 输出形状异常 {shape}"
        out_lab = np.asanyarray(nib.load(ol).dataobj)
        assert out_lab.sum() > 0, "预处理后标注为空"
    return f"重采样/归一化/ROI 均正确, ROI shape={shape}"


def b4_infer():
    from src.infer import sliding_window_inference, _gaussian_weight
    from src.models.unet3d import UNet3D
    gw = _gaussian_weight((32, 32, 32))
    assert gw.min() > 0, "高斯权重存在零值"
    assert gw[16, 16, 16] > gw[0, 0, 0], "中心权重应大于边角"
    m = UNet3D(1, 2, base_features=4, num_levels=2, norm="batch")
    m.eval()
    img = np.random.rand(48, 40, 36).astype(np.float32)
    pred = sliding_window_inference(m, img, (32, 32, 32), 2, overlap=0.5)
    assert pred.shape == img.shape, f"推理输出形状 {pred.shape} != {img.shape}"
    assert pred.min() >= 0 and pred.max() <= 1
    return f"非立方体输入 {img.shape} 推理形状正确"


def b5_split():
    from src.data.dataset import split_by_patient
    samples = [(f"p/{p}_{s}/image.nii.gz", f"p/{p}_{s}/label.nii.gz", f"{p}_{s}")
               for p in ["A", "B", "C", "D", "E", "F", "G", "H"] for s in ["L", "R"]]
    tr, va, te = split_by_patient(samples, 0.25, 0.25, 42)
    def pid(cid):
        return cid.rsplit("_", 1)[0]
    sets = [{pid(s[2]) for s in subset} for subset in (tr, va, te)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2]), \
        "同一患者出现在不同子集"
    assert len(tr) + len(va) + len(te) == len(samples)
    return f"train/val/test={len(tr)}/{len(va)}/{len(te)} 无患者泄漏"


if __name__ == "__main__":
    check("A1 全部 Python 文件编译", a1_compile)
    check("A2 包结构", a2_packages)
    check("B1 3D U-Net 前向", b1_model)
    check("B2 Dice+CE 损失", b2_loss)
    check("B3 预处理单元测试", b3_preprocess)
    check("B4 滑动窗口推理", b4_infer)
    check("B5 按患者划分", b5_split)
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"\n结果: {len(RESULTS)-n_fail}/{len(RESULTS)} 通过")
    sys.exit(1 if n_fail else 0)
