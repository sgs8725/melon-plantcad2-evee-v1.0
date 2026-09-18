#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_probe.py
======================
训练 EVEE 协方差探针（变异效应预测）。

基础模型 PlantCAD2 全程冻结，本脚本只训练协方差探针——参数量小，
单 GPU 甚至 CPU 即可。激活张量从 data/activations/ 缓存读取。

支持两种模式：
  - 二分类（n_outputs=2）：有利/不利，CE 损失，AUROC 评估。
  - 多性状回归（n_outputs=len(traits)）：多目标 sigmoid，MSE 损失。

用法：
    python scripts/train_probe.py \
        --activations data/activations \
        --labels data/variants.parquet \
        --out assets/probe_covariance.safetensors \
        --task classification --d-model 768 --epochs 30

冒烟测试（标签随机，验证流程可跑通）：
    python scripts/train_probe.py --activations data/activations \
        --labels data/variants.parquet --out /tmp/probe.safetensors \
        --d-model 256 --smoke-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probes.covariance_probe import CovarianceProbe
from breeding.genomic_value import DEFAULT_TRAITS


def load_activation(act_dir: Path, variant_id: str) -> torch.Tensor | None:
    import safetensors.torch
    fname = act_dir / f"{variant_id.replace(':', '_').replace('>', '_')}.safetensors"
    if not fname.exists():
        return None
    return safetensors.torch.load_file(str(fname))["activations"].float()


def auroc(y_true: list[int], y_score: list[float]) -> float:
    """无 sklearn 依赖的 AUROC（按秩计算）。"""
    pairs = sorted(zip(y_score, y_true))
    ranks = {}
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j - 1) / 2 + 1
        for k in range(i, j):
            ranks[k] = avg
        i = j
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = sum(ranks[idx] for idx, (_, t) in enumerate(pairs) if t == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True, help="激活缓存目录")
    ap.add_argument("--labels", required=True, help="变异 parquet（含 label 列）")
    ap.add_argument("--out", required=True, help="探针权重输出 (.safetensors)")
    ap.add_argument("--task", choices=["classification", "regression"],
                    default="classification")
    ap.add_argument("--d-model", type=int, default=768,
                    help="PlantCAD2 隐藏维：Small=768/Medium=1024/Large=1536")
    ap.add_argument("--d-hidden", type=int, default=64)
    ap.add_argument("--d-probe", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--smoke-test", action="store_true",
                    help="标签缺失时随机生成，仅验证流程")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    import polars as pl
    import safetensors.torch

    df = pl.read_parquet(args.labels)
    act_dir = Path(args.activations)

    # ---- 收集 (激活, 标签) ----
    n_traits = len(DEFAULT_TRAITS)
    samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    import random
    rng = random.Random(args.seed)

    for row in df.iter_rows(named=True):
        act = load_activation(act_dir, row["variant_id"])
        if act is None:
            continue
        if args.task == "classification":
            label = row.get("label", -1)
            if label not in (0, 1):
                if args.smoke_test:
                    label = rng.randint(0, 1)
                else:
                    continue
            y = torch.tensor(int(label))
        else:  # regression / multi-trait
            tl = row.get("trait_label", "")
            if tl:
                y = torch.tensor([float(x) for x in tl.split(",")])
            elif args.smoke_test:
                y = torch.rand(n_traits)
            else:
                continue
        samples.append((act, y))

    if not samples:
        sys.exit("没有可用样本：请回填 label/trait_label，或加 --smoke-test。")

    print(f"[数据] 可用样本 {len(samples)} 条")
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_frac))
    val, train = samples[:n_val], samples[n_val:]
    print(f"[数据] 训练 {len(train)} / 验证 {len(val)}")

    # ---- 探针 ----
    n_outputs = 2 if args.task == "classification" else n_traits
    probe = CovarianceProbe(
        d_model=args.d_model, d_hidden=args.d_hidden,
        d_probe=args.d_probe, n_outputs=n_outputs,
    )
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss() if args.task == "classification" \
        else nn.MSELoss()

    def batches(data):
        for i in range(0, len(data), args.batch_size):
            chunk = data[i:i + args.batch_size]
            x = torch.stack([c[0] for c in chunk])      # [B,2,2,K,d]
            y = torch.stack([c[1] for c in chunk])
            yield x, y

    # ---- 训练 ----
    for ep in range(1, args.epochs + 1):
        probe.train()
        tot = 0.0
        for x, y in batches(train):
            logits = probe(x)
            if args.task == "classification":
                loss = loss_fn(logits, y)
            else:
                loss = loss_fn(torch.sigmoid(logits), y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        # ---- 验证 ----
        probe.eval()
        with torch.no_grad():
            if args.task == "classification":
                ys, ss = [], []
                for x, y in batches(val):
                    p = probe.predict_proba(x)
                    ys += y.tolist()
                    ss += p.tolist()
                metric = auroc(ys, ss)
                mname = "val_AUROC"
            else:
                err = 0.0
                nb = 0
                for x, y in batches(val):
                    p = probe.predict_traits(x)
                    err += float(((p - y) ** 2).mean())
                    nb += 1
                metric = err / max(1, nb)
                mname = "val_MSE"
        print(f"  epoch {ep:3d}  train_loss={tot:.4f}  {mname}={metric:.4f}")

    # ---- 保存 ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_model(probe, str(out))
    import json
    meta = {
        "d_model": args.d_model, "d_hidden": args.d_hidden,
        "d_probe": args.d_probe, "n_outputs": n_outputs, "task": args.task,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[完成] 探针权重 -> {out}")
    print(f"[完成] 探针配置 -> {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
