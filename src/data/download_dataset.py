# -*- coding: utf-8 -*-
"""
从 GitHub 公开仓库下载 TMJ CBCT 数据集
（Henryblankwu/TMJ-CBCT-Segmentation-Data，47 例患者：DICOM 切片 + 左右侧 NIfTI 标注）。

用法：
    python -m src.data.download_dataset --out data/raw [--patients 5]
"""

import argparse
import json
import os
import urllib.request

API = "https://api.github.com/repos/Henryblankwu/TMJ-CBCT-Segmentation-Data/contents"
RAW = "https://raw.githubusercontent.com/Henryblankwu/TMJ-CBCT-Segmentation-Data/main"


def list_patients():
    with urllib.request.urlopen(f"{API}/TMJ-DataSet") as r:
        items = json.load(r)
    return [it["name"] for it in items if it["type"] == "dir"]


def download_patient(pid, out_dir):
    dest = os.path.join(out_dir, pid)
    os.makedirs(dest, exist_ok=True)
    with urllib.request.urlopen(f"{API}/TMJ-DataSet/{pid}") as r:
        items = json.load(r)
    for it in items:
        if it["type"] != "file":
            continue
        fpath = os.path.join(dest, it["name"])
        if os.path.exists(fpath):
            continue
        url = f"{RAW}/TMJ-DataSet/{pid}/{it['name']}"
        urllib.request.urlretrieve(url, fpath)
    print(f"[完成] {pid}: {len(items)} 个文件")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--patients", type=int, default=0, help="0 = 全部 47 例")
    args = ap.parse_args()
    pids = list_patients()
    print(f"数据集中共 {len(pids)} 例患者")
    for pid in pids[: args.patients or None]:
        download_patient(pid, args.out)
    print("下载完成。下一步：python -m src.data.dicom_to_nifti 将 DICOM 转为 NIfTI 体数据")


if __name__ == "__main__":
    main()
