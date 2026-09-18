#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_annotation_tracks.py
==================================
从实验轨道（BED 峰 / bigWig 信号）训练**扩展注释类**的 EVEE 注释探针。

`train_annotation_probe.py` 只能训练可由 GFF 直接生成的 17 类结构注释
（区域/剪接/密码子/启动子/终止子）。而 MELON_ANNOTATIONS 中的下列类别需要
**实验轨道**才能得到逐位置标签，本脚本即为此入口：

  - 转录因子结合位点 tfbs_*：DAP-seq / ChIP-seq 峰（BED）
      （tfbs_MYB / bHLH / WRKY / AP2ERF / NAC(NOR) / MADS / bZIP …）
  - 染色质可及性 chromatin_accessible_*：ATAC-seq / DNase-seq 峰或信号
      （flesh 果肉 / rind 果皮 / leaf 叶片）
  - 顺式调控强度 cre_high_strength：STARR-seq / MPRA 活性峰
  - 保守性 conserved_cross_species：phyloP / phastCons 信号（bigWig）
  - 通路 motif（如 sugar_metabolism_motif 等）：motif 扫描 / 峰区 BED

设计：与 AnnotationProbe（线性 + sigmoid，多标签 BCE）一致——所有轨道都
转成**逐位置二值标签**（在峰内 / 信号超阈值 = 1）。BED 直接取区间；bigWig
取信号后按阈值或分位数二值化。产出探针含 .json sidecar（类别列表 + d_model），
predict.py 会据此自动重建并输出扰动谱。可与结构注释探针分别训练、分别加载。

轨道配置（JSON）示例 tracks.json：
    {
      "chrom": "1",
      "tracks": [
        {"class": "tfbs_MYB",  "path": "tracks/MYB_dapseq.bed",  "type": "bed"},
        {"class": "tfbs_NAC",  "path": "tracks/NAC_dapseq.bed",  "type": "bed"},
        {"class": "chromatin_accessible_flesh",
         "path": "tracks/flesh_atac.bw", "type": "bigwig", "quantile": 0.90},
        {"class": "conserved_cross_species",
         "path": "tracks/phyloP.bw", "type": "bigwig", "threshold": 0.0}
      ]
    }

用法（真实，需 GPU + PlantCAD2 权重 + 基因组 + 轨道；bigWig 需 pyBigWig）：
    python scripts/train_annotation_tracks.py \
        --tracks tracks.json --genome assets/genome/melon_DHL92.fa \
        --model assets/plantcad2 --d-model 768 --n-windows 80 --epochs 20 \
        --out assets/annotation_probe_tracks.safetensors

离线冒烟测试（纯 CPU，自动合成 BED 轨道，无需任何外部文件 / pyBigWig）：
    python scripts/train_annotation_tracks.py --smoke-test \
        --out /tmp/ann_tracks.safetensors
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probes.annotation_probe import AnnotationProbe, MELON_ANNOTATIONS
from meloncad.encoder import MelonCADEncoder, build_mock_encoder
# 复用结构注释脚本里的训练 / 平衡 / 保存逻辑，避免重复
from scripts.train_annotation_probe import train_probe, balance


# ------------------------------------------------------------------ #
# 轨道读取
# ------------------------------------------------------------------ #
def load_bed_tree(path: str, chrom: str):
    """读 BED（chrom start end [value]），返回该染色体的 IntervalTree。
    若有第 4 列则作为 data 存入（回归模式用其作连续信号；缺失记为 1.0）。"""
    import intervaltree
    import gzip
    tree = intervaltree.IntervalTree()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            p = line.split()
            if len(p) < 3 or p[0] != chrom:
                continue
            s, e = int(p[1]), int(p[2])           # BED: 0-based, 半开
            val = 1.0
            if len(p) >= 4:
                try:
                    val = float(p[3])
                except ValueError:
                    val = 1.0
            if s < e:
                tree.addi(s, e, val)
    return tree


def bigwig_signal(path: str, chrom: str, lo: int, hi: int) -> np.ndarray:
    """读 bigWig 在 [lo,hi) 的逐位置原始信号（回归用）。需 pyBigWig。"""
    try:
        import pyBigWig
    except ImportError:
        raise SystemExit("[错误] 读取 bigWig 需要 pyBigWig：pip install pyBigWig")
    bw = pyBigWig.open(path)
    vals = bw.values(chrom, lo, hi, numpy=True)
    bw.close()
    return np.nan_to_num(vals, nan=0.0).astype(np.float32)


def bigwig_binary(path: str, chrom: str, lo: int, hi: int,
                  threshold: float | None, quantile: float | None) -> np.ndarray:
    """读 bigWig 在 [lo,hi) 的逐位置信号并二值化（超阈值=1）。需 pyBigWig。"""
    try:
        import pyBigWig
    except ImportError:
        raise SystemExit(
            "[错误] 读取 bigWig 需要 pyBigWig：pip install pyBigWig\n"
            "        （或把信号预先转成 BED 峰，用 type=bed）")
    bw = pyBigWig.open(path)
    vals = bw.values(chrom, lo, hi, numpy=True)
    bw.close()
    vals = np.nan_to_num(vals, nan=0.0)
    if quantile is not None:
        thr = float(np.quantile(vals, quantile))
    elif threshold is not None:
        thr = float(threshold)
    else:
        thr = float(np.quantile(vals, 0.90))      # 默认 top-10% 视为阳性
    return (vals > thr).astype(np.float32)


def labels_for_window(track_cfgs, chrom, lo, hi, bed_trees,
                      regression: bool = False) -> np.ndarray:
    """组装窗口 [lo,hi) 的 [L, n_class] 标签。
    binary：在峰内/超阈值=1；regression：连续信号值（BED 第4列 / bigWig 原始）。"""
    L = hi - lo
    labels = np.zeros((L, len(track_cfgs)), dtype=np.float32)
    for ci, tc in enumerate(track_cfgs):
        if tc["type"] == "bed":
            for iv in bed_trees[ci].overlap(lo, hi):
                a, b = max(iv.begin, lo), min(iv.end, hi)
                if a < b:
                    if regression:
                        v = float(iv.data) if iv.data is not None else 1.0
                        labels[a - lo:b - lo, ci] = np.maximum(
                            labels[a - lo:b - lo, ci], v)
                    else:
                        labels[a - lo:b - lo, ci] = 1.0
        else:  # bigwig
            if regression:
                labels[:, ci] = bigwig_signal(tc["path"], chrom, lo, hi)
            else:
                labels[:, ci] = bigwig_binary(
                    tc["path"], chrom, lo, hi,
                    tc.get("threshold"), tc.get("quantile"))
    return labels


def read_chrom_seq(genome_path: str, chrom: str):
    from Bio import SeqIO
    for rec in SeqIO.parse(genome_path, "fasta"):
        if rec.id == chrom:
            return str(rec.seq).upper()
    return None


def save(probe, classes, out: Path, d_model: int,
         task: str = "binary", transform: str = "none") -> None:
    import safetensors.torch
    out.parent.mkdir(parents=True, exist_ok=True)
    probe.cpu()
    safetensors.torch.save_model(probe, str(out))
    meta = {"d_model": d_model, "annotations": classes,
            "n_annotations": len(classes), "source": "experimental_tracks",
            "task": task, "transform": transform}
    out.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2))
    mdir = ROOT / "metadata"
    mdir.mkdir(exist_ok=True)
    (mdir / "annotation_tracks.json").write_text(
        json.dumps({"annotations": classes, "task": task}, ensure_ascii=False))
    print(f"[完成] 扩展注释探针({task}) -> {out}")
    print(f"[完成] 配置 -> {out.with_suffix('.json')} ; "
          f"面板 -> {mdir/'annotation_tracks.json'}")


def collect(track_cfgs, chrom, seq, enc, window, n_windows, seed,
            regression: bool = False):
    rng = random.Random(seed)
    bed_trees = [load_bed_tree(tc["path"], chrom) if tc["type"] == "bed" else None
                 for tc in track_cfgs]
    clen = len(seq)
    all_e, all_l = [], []
    for wi in range(n_windows):
        center = rng.randint(window, clen - window)
        lo, hi = center - window // 2, center + window // 2
        sub = seq[lo:hi]
        if len(sub) < window:
            continue
        lbl = labels_for_window(track_cfgs, chrom, lo, hi, bed_trees, regression)
        if lbl.sum() < 20:
            continue
        try:
            emb = enc.encode_reference(sub)
        except Exception as e:
            print(f"  window {wi}: {type(e).__name__}")
            continue
        L = min(emb.shape[0], lbl.shape[0])
        all_e.append(emb[:L].cpu())
        all_l.append(torch.from_numpy(lbl[:L]))
        if (wi + 1) % 10 == 0:
            print(f"  {wi + 1}/{n_windows}")
    return all_e, all_l


# ------------------------------------------------------------------ #
# 离线冒烟测试：合成 BED 轨道 + mock 编码器
# ------------------------------------------------------------------ #
def run_smoke(out: str) -> None:
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="anntrk_"))
    rng = random.Random(0)
    window = 256
    enc = build_mock_encoder(d_model=256)

    # 合成两条 BED 轨道（落在 [0,window) 内的若干峰），映射到两个扩展类
    classes = ["tfbs_MYB", "chromatin_accessible_flesh"]
    track_cfgs = []
    for cls in classes:
        bed = tmp / f"{cls}.bed"
        with open(bed, "w") as f:
            for _ in range(6):
                s = rng.randint(0, window - 30)
                f.write(f"1\t{s}\t{s + 20}\n")
        track_cfgs.append({"class": cls, "path": str(bed), "type": "bed"})

    print(f"[冒烟测试] 扩展类: {classes}（合成 BED 轨道，d_model=256 mock）")
    seq = "".join(rng.choice("ACGT") for _ in range(window))
    bed_trees = [load_bed_tree(tc["path"], "1") for tc in track_cfgs]
    all_e, all_l = [], []
    for _ in range(2):
        lbl = labels_for_window(track_cfgs, "1", 0, window, bed_trees)
        emb = enc.encode_reference(seq)
        L = min(emb.shape[0], lbl.shape[0])
        all_e.append(emb[:L].cpu())
        all_l.append(torch.from_numpy(lbl[:L]))
    embs = torch.cat(all_e); lbls = torch.cat(all_l)
    embs, lbls = balance(embs, lbls)
    print(f"  训练位点 {embs.shape[0]} × {len(classes)} 类")
    # 用本脚本的类集训练（train_probe 的 N_ANN 基于结构面板，这里直接构 probe）
    probe = _train(embs, lbls, classes, d_model=256,
                   device=torch.device("cpu"), epochs=3)
    save(probe, classes, Path(out), d_model=256, task="binary")
    print("[冒烟测试-binary] 通过：BED 轨道训练二值扩展注释类跑通。")

    # --- 回归子测试：合成 BED4（带信号），按连续信号训练 MSE ---
    print("\n[冒烟测试-regression] 合成 BED4（含信号）训练连续目标...")
    rcls = ["chromatin_signal_flesh"]
    bed4 = tmp / "signal.bed"
    with open(bed4, "w") as f:
        for _ in range(8):
            s = rng.randint(0, window - 30)
            f.write(f"1\t{s}\t{s + 20}\t{round(rng.uniform(1.0, 9.0), 2)}\n")
    rcfg = [{"class": rcls[0], "path": str(bed4), "type": "bed"}]
    rtrees = [load_bed_tree(rcfg[0]["path"], "1")]
    re_, rl_ = [], []
    for _ in range(2):
        lbl = labels_for_window(rcfg, "1", 0, window, rtrees, regression=True)
        emb = enc.encode_reference(seq)
        L = min(emb.shape[0], lbl.shape[0])
        re_.append(emb[:L].cpu()); rl_.append(torch.from_numpy(lbl[:L]))
    rembs = torch.cat(re_); rlbls = torch.cat(rl_)
    rembs, rlbls = balance(rembs, rlbls)
    rprobe = _train(rembs, rlbls, rcls, d_model=256,
                    device=torch.device("cpu"), epochs=3,
                    regression=True, log1p=True)
    save(rprobe, rcls, Path(str(out).replace(".safetensors", "_reg.safetensors")),
         d_model=256, task="regression", transform="log1p")
    print("[冒烟测试-regression] 通过：连续信号回归训练跑通。")
    print("[冒烟测试] 全部通过。")


def _train(embs, lbls, classes, d_model, device, epochs,
           regression: bool = False, log1p: bool = False):
    """训练注释探针（类集为 classes）。binary: BCE+acc；regression: MSE+Pearson。"""
    from torch import nn, optim
    probe = AnnotationProbe(d_model=d_model, annotations=classes).to(device)
    opt = optim.Adam(probe.parameters(), lr=1e-3)
    if regression and log1p:
        lbls = torch.log1p(lbls.clamp(min=0))      # log1p 稳定大信号
    crit = (nn.MSELoss() if regression else
            nn.BCEWithLogitsLoss(pos_weight=torch.full((len(classes),), 3.0).to(device)))
    ds = torch.utils.data.TensorDataset(embs.to(device), lbls.to(device))
    loader = torch.utils.data.DataLoader(ds, batch_size=4096, shuffle=True)
    for ep in range(epochs):
        tot, n = 0.0, 0
        for x, y in loader:
            opt.zero_grad()
            loss = crit(probe(x), y)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * x.shape[0]
            n += x.shape[0]
        with torch.no_grad():
            pred = probe(embs.to(device))
            yy = lbls.to(device)
            if regression:
                mse = float(((pred - yy) ** 2).mean())
                print(f"  epoch {ep + 1:2d}: loss={tot/max(1,n):.4f} train_MSE={mse:.4f}")
            else:
                acc = ((torch.sigmoid(pred) > 0.5).float() == yy).float().mean().item()
                print(f"  epoch {ep + 1:2d}: loss={tot/max(1,n):.4f} acc={acc:.4f}")
    return probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", help="轨道配置 JSON（class/path/type[/threshold/quantile]）")
    ap.add_argument("--genome", help="参考基因组 FASTA")
    ap.add_argument("--model", default="assets/plantcad2")
    ap.add_argument("--chrom", default=None, help="染色体名（缺省取配置里的 chrom）")
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--window", type=int, default=8192)
    ap.add_argument("--n-windows", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["binary", "regression"], default="binary",
                    help="binary=在峰内/超阈值(二值)；regression=拟合连续信号值")
    ap.add_argument("--log1p", action="store_true",
                    help="回归模式对信号做 log1p 变换（稳定大信号）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    if args.smoke_test:
        run_smoke(args.out)
        return
    if not (args.tracks and args.genome):
        raise SystemExit("[错误] 真实训练需 --tracks 与 --genome（或 --smoke-test）。")

    regression = (args.mode == "regression")
    cfg = json.loads(Path(args.tracks).read_text())
    track_cfgs = cfg["tracks"]
    chrom = args.chrom or cfg.get("chrom", "1")
    classes = [t["class"] for t in track_cfgs]
    unknown = [c for c in classes if c not in MELON_ANNOTATIONS]
    if unknown:
        print(f"[提示] 以下类别不在 MELON_ANNOTATIONS 面板（仍可训练，自定义类）：{unknown}")
    print(f"[1/4] 轨道 {len(classes)} 类（chrom={chrom}，mode={args.mode}）：{classes}")
    if regression:
        print("      回归模式：BED 第4列 / bigWig 原始信号作为连续目标"
              + ("，log1p 变换" if args.log1p else ""))

    print("[2/4] 加载 PlantCAD2 编码器...")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    enc = MelonCADEncoder(model_path=args.model, device=str(device))
    print(f"  device={device}  d_model={enc.d_model}")

    print("[3/4] 读取轨道 + 提取训练数据...")
    seq = read_chrom_seq(args.genome, chrom)
    if seq is None:
        raise SystemExit(f"[错误] 基因组中找不到染色体 '{chrom}'。")
    all_e, all_l = collect(track_cfgs, chrom, seq, enc,
                           args.window, args.n_windows, args.seed,
                           regression=regression)
    if not all_e:
        raise SystemExit("[错误] 无有效训练窗口（检查轨道与 chrom 命名是否一致）。")
    embs = torch.cat(all_e); lbls = torch.cat(all_l)
    embs, lbls = balance(embs, lbls, args.seed)
    print(f"  平衡后训练位点 {embs.shape[0]} × {len(classes)} 类")

    print("[4/4] 训练扩展注释探针...")
    probe = _train(embs, lbls, classes, d_model=args.d_model,
                   device=device, epochs=args.epochs,
                   regression=regression, log1p=args.log1p)
    save(probe, classes, Path(args.out), d_model=args.d_model,
         task=args.mode, transform=("log1p" if (regression and args.log1p) else "none"))


if __name__ == "__main__":
    main()
