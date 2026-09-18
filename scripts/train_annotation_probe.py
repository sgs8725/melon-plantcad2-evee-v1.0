#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_annotation_probe.py
=================================
从甜瓜 GFF 注释自动生成逐位置功能标签，训练 EVEE 注释探针
（AnnotationProbe），用于变异扰动谱分析。

与上游番茄 tomato-gfm-evee2 的 train_annotation_probe 同构，但：
  - 基础模型换成 PlantCAD2（MelonCADEncoder.encode_reference 给出反向互补
    不变的逐位置嵌入 [L, d_model]，d_model=768），不再是 PlantGFM(d=1024)；
  - 注释面板取 MELON_ANNOTATIONS 中**可由 GFF 直接生成**的结构类子集
    （region_/splice_/codon_/start_/stop_/frameshift_/promoter_/terminator）；
    甜瓜特异的 motif/TFBS/染色质类需实验轨道（DAP-seq/ATAC 等），不在此脚本，
    可另行用对应实验标签扩展。

经验教训（来自上游流水线报告，已内化）：
  * config 对齐：d_model 必须与所用 PlantCAD2 权重的 hidden_size 一致
    （Small=768 / Medium=1024 / Large=1536）；
  * 正负样本极不平衡 → 用 BCEWithLogitsLoss(pos_weight) 并对负样本下采样；
  * smoke-test 用随机/合成数据，仅验证流程，不代表真实精度。

用法（真实，需 GPU + 已下 PlantCAD2 权重 + 甜瓜 GFF/基因组）：
    python scripts/train_annotation_probe.py \
        --gff assets/genome/melon_DHL92.gff3 \
        --genome assets/genome/melon_DHL92.fa \
        --model assets/plantcad2 --chrom 1 \
        --d-model 768 --n-windows 50 --epochs 20 \
        --out assets/annotation_probe.safetensors

离线冒烟测试（纯 CPU，无需任何外部文件 / 权重）：
    python scripts/train_annotation_probe.py --smoke-test \
        --out /tmp/ann_probe.safetensors
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probes.annotation_probe import AnnotationProbe, MELON_ANNOTATIONS
from meloncad.encoder import MelonCADEncoder, build_mock_encoder

# 可由 GFF 直接生成的结构类注释（与上游同一前缀过滤逻辑）
GFF_PREFIXES = ("region_", "splice_", "codon_", "start_",
                "stop_", "frameshift_", "promoter_", "terminator")
GFF_NAMES = [a for a in MELON_ANNOTATIONS if a.startswith(GFF_PREFIXES)]
N_ANN = len(GFF_NAMES)


# ------------------------------------------------------------------ #
# GFF -> 注释间隔树
# ------------------------------------------------------------------ #
def build_trees(gff_path: str, chrom: str = "1"):
    import intervaltree

    trees = {a: intervaltree.IntervalTree() for a in GFF_NAMES}
    records: dict[str, list] = {a: [] for a in GFF_NAMES}
    mrna_exons: dict[str, list] = {}
    mrna_cds: dict[str, list] = {}
    mrna_strands: dict[str, str] = {}

    opener = gzip.open if str(gff_path).endswith(".gz") else open
    with opener(gff_path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[0] != chrom:
                continue
            ft, start, end, strand, attr = (
                parts[2], int(parts[3]), int(parts[4]), parts[6], parts[8])

            if ft in ("CDS", "five_prime_UTR", "three_prime_UTR"):
                key = ("region_CDS" if ft == "CDS"
                       else "region_5UTR" if ft == "five_prime_UTR"
                       else "region_3UTR")
                if key in records:
                    records[key].append((start, end))
            if ft == "exon":
                if "region_exon_boundary" in records:
                    records["region_exon_boundary"].append((start, start + 2))
                    records["region_exon_boundary"].append((end - 2, end))
                m = re.search(r"Parent=([^;]+)", attr)
                if m:
                    tid = m.group(1)
                    mrna_exons.setdefault(tid, []).append((start, end))
                    mrna_strands[tid] = strand
            elif ft == "CDS":
                m = re.search(r"Parent=([^;]+)", attr)
                if m:
                    tid = m.group(1)
                    mrna_cds.setdefault(tid, []).append((start, end))

    for tid, exs in mrna_exons.items():
        exs = sorted(exs)
        cds = sorted(mrna_cds.get(tid, []))
        strand = mrna_strands.get(tid, "+")
        # 内含子 + 剪接位点
        for i in range(len(exs) - 1):
            if exs[i][1] < exs[i + 1][0]:
                istart, iend = exs[i][1] + 1, exs[i + 1][0] - 1
                if istart < iend:
                    records.get("region_intron", []).append((istart, iend))
                    records.get("splice_donor", []).append((istart - 2, istart + 3))
                    records.get("splice_acceptor", []).append((iend - 2, iend + 3))
        # 密码子相位 + 起止密码子（仅正链演示）
        if cds and strand == "+":
            flat = []
            for cs, ce in cds:
                flat.extend(range(cs, ce))
            for ci, pos in enumerate(flat):
                k = f"codon_pos{(ci % 3) + 1}"
                if k in records:
                    records[k].append((pos, pos + 1))
            if "start_codon" in records:
                records["start_codon"].append((cds[0][0], cds[0][0] + 3))
            if "stop_codon" in records:
                records["stop_codon"].append((cds[-1][1] - 3, cds[-1][1]))
        # 启动子 / 终止子
        if exs:
            if strand == "+":
                records.get("promoter_core", []).append((max(1, exs[0][0] - 50), exs[0][0] + 50))
                records.get("promoter_proximal", []).append((max(1, exs[0][0] - 2000), exs[0][0]))
                records.get("terminator", []).append((exs[-1][1], exs[-1][1] + 300))
            else:
                records.get("promoter_core", []).append((max(1, exs[-1][1] - 50), exs[-1][1] + 50))
                records.get("promoter_proximal", []).append((exs[-1][1], exs[-1][1] + 2000))
                records.get("terminator", []).append((max(1, exs[0][0] - 300), exs[0][0]))

    for ann, rs in records.items():
        for s, e in rs:
            if s < e:
                trees[ann].addi(s, e)
    return trees


def labels_for_window(trees, start: int, end: int) -> np.ndarray:
    labels = np.zeros((end - start, N_ANN), dtype=np.float32)
    for i, ann in enumerate(GFF_NAMES):
        for iv in trees[ann].overlap(start, end):
            lo, hi = max(iv.begin, start), min(iv.end, end)
            if lo < hi:
                labels[lo - start:hi - start, i] = 1.0
    return labels


def read_chrom_seq(genome_path: str, chrom: str) -> str | None:
    from Bio import SeqIO
    for rec in SeqIO.parse(genome_path, "fasta"):
        if rec.id == chrom:
            return str(rec.seq).upper()
    return None


# ------------------------------------------------------------------ #
# 训练
# ------------------------------------------------------------------ #
def train_probe(embs: torch.Tensor, lbls: torch.Tensor, d_model: int,
                device, epochs: int) -> AnnotationProbe:
    probe = AnnotationProbe(d_model=d_model, annotations=GFF_NAMES).to(device)
    opt = optim.Adam(probe.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.full((N_ANN,), 3.0).to(device))
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
            pred = (torch.sigmoid(probe(embs.to(device))) > 0.5).float()
            acc = (pred == lbls.to(device)).float().mean().item()
        print(f"  epoch {ep + 1:2d}: loss={tot / max(1, n):.4f} acc={acc:.4f}")
    return probe


def balance(embs: torch.Tensor, lbls: torch.Tensor, seed: int = 42):
    pos_mask = lbls.sum(dim=1) > 0
    n_pos = int(pos_mask.sum())
    if n_pos == 0:
        return embs, lbls
    g = torch.Generator().manual_seed(seed)
    neg_all = torch.where(~pos_mask)[0]
    n_neg = min(n_pos * 2, neg_all.numel())
    neg_idx = neg_all[torch.randperm(neg_all.numel(), generator=g)[:n_neg]]
    keep = torch.cat([torch.where(pos_mask)[0], neg_idx])
    return embs[keep], lbls[keep]


def save(probe: AnnotationProbe, out: Path, d_model: int) -> None:
    import safetensors.torch
    out.parent.mkdir(parents=True, exist_ok=True)
    probe.cpu()
    safetensors.torch.save_model(probe, str(out))
    # 配置 sidecar（与 train_probe.py 一致：predict.py 据此重建）
    meta = {"d_model": d_model, "annotations": GFF_NAMES, "n_annotations": N_ANN}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    # 复制一份注释面板到 metadata/（可复现）
    mdir = ROOT / "metadata"
    mdir.mkdir(exist_ok=True)
    (mdir / "annotation_probe.json").write_text(
        json.dumps({"annotations": GFF_NAMES}, ensure_ascii=False))
    print(f"[完成] 注释探针 -> {out}")
    print(f"[完成] 配置 -> {out.with_suffix('.json')} ; 面板 -> {mdir/'annotation_probe.json'}")


# ------------------------------------------------------------------ #
# 离线冒烟测试：合成小基因组 + 间隔树，纯 CPU 跑通
# ------------------------------------------------------------------ #
def run_smoke(out: str) -> None:
    import intervaltree
    print(f"[冒烟测试] {N_ANN} 类: {GFF_NAMES}")
    rng = random.Random(0)
    window = 256
    enc = build_mock_encoder(d_model=256)

    # 合成两段间隔（落在窗口内），制造正样本
    trees = {a: intervaltree.IntervalTree() for a in GFF_NAMES}
    for a in GFF_NAMES[:6]:
        s = rng.randint(0, window - 40)
        trees[a].addi(s, s + 30)

    all_e, all_l = [], []
    for _ in range(2):
        seq = "".join(rng.choice("ACGT") for _ in range(window))
        lbl = labels_for_window(trees, 0, window)
        emb = enc.encode_reference(seq)            # [L, 256] (mock)
        L = min(emb.shape[0], lbl.shape[0])
        all_e.append(emb[:L].cpu())
        all_l.append(torch.from_numpy(lbl[:L]))
    embs = torch.cat(all_e); lbls = torch.cat(all_l)
    embs, lbls = balance(embs, lbls)
    print(f"  训练位点 {embs.shape[0]} × {N_ANN} 类（d_model=256 mock）")
    probe = train_probe(embs, lbls, d_model=256, device=torch.device("cpu"), epochs=3)
    save(probe, Path(out), d_model=256)
    print("[冒烟测试] 通过：注释探针训练流程跑通。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gff", help="甜瓜 GFF3（可 .gz）")
    ap.add_argument("--genome", help="参考基因组 FASTA")
    ap.add_argument("--model", default="assets/plantcad2", help="PlantCAD2 权重目录")
    ap.add_argument("--chrom", default="1", help="训练用染色体名（默认 1）")
    ap.add_argument("--d-model", type=int, default=768,
                    help="PlantCAD2 hidden_size（Small=768/Medium=1024/Large=1536）")
    ap.add_argument("--window", type=int, default=8192)
    ap.add_argument("--n-windows", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    if args.smoke_test:
        run_smoke(args.out)
        return
    if not (args.gff and args.genome):
        raise SystemExit("[错误] 真实训练需 --gff 与 --genome（或用 --smoke-test）。")

    print(f"[1/4] 构建注释间隔树（chrom={args.chrom}，{N_ANN} 类）...")
    trees = build_trees(args.gff, args.chrom)

    print("[2/4] 加载 PlantCAD2 编码器...")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    enc = MelonCADEncoder(model_path=args.model, device=str(device))
    print(f"  device={device}  d_model={enc.d_model}")

    print("[3/4] 提取训练数据...")
    seq = read_chrom_seq(args.genome, args.chrom)
    if seq is None:
        raise SystemExit(f"[错误] 基因组中找不到染色体 '{args.chrom}'。")
    clen = len(seq)
    rng = random.Random(args.seed)
    all_e, all_l = [], []
    for wi in range(args.n_windows):
        center = rng.randint(args.window, clen - args.window)
        lo, hi = center - args.window // 2, center + args.window // 2
        sub = seq[lo:hi]
        if len(sub) < args.window:
            continue
        lbl = labels_for_window(trees, lo, hi)
        if lbl.sum() < 50:
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
            print(f"  {wi + 1}/{args.n_windows}")
    if not all_e:
        raise SystemExit("[错误] 无有效训练窗口。")
    embs = torch.cat(all_e); lbls = torch.cat(all_l)
    embs, lbls = balance(embs, lbls, args.seed)
    print(f"  平衡后训练位点 {embs.shape[0]} × {N_ANN} 类")

    print("[4/4] 训练 AnnotationProbe...")
    probe = train_probe(embs, lbls, d_model=args.d_model, device=device, epochs=args.epochs)
    save(probe, Path(args.out), d_model=args.d_model)


if __name__ == "__main__":
    main()
