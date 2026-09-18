#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/evaluate.py
===================
对 predict.py 产出的预测结果做独立评估，输出分类 / 回归指标。

上游 tomato-gfm-evee2 流水线报告把 AUROC 作为核心验收指标，并发现
「标签质量 > 基因组覆盖」（GWAS p-value 标注 AUROC≈0.70 优于 meta-QTL 区间
≈0.59）。本脚本把这种评估固化为可复用工具：

  分类（effect_score vs label）：
    AUROC、AUPRC、阈值下的 accuracy / precision / recall / F1、混淆矩阵。
  回归（effect_<trait> vs 真值）：
    每性状 MSE 与 Pearson 相关；总体均值。

依赖极简：仅 polars + 纯 Python 统计（无 sklearn）。

用法：
    # 分类：预测 CSV 含 effect_score；标签来自变异 parquet 的 label 列
    python scripts/evaluate.py --pred results/predictions.csv \
        --labels data/variants_labeled.parquet --task classification

    # 回归：预测 CSV 含 effect_<trait>；真值来自材料级表型标签
    python scripts/evaluate.py --pred results/predictions.csv \
        --labels data/phenotype_labels.parquet --task regression \
        --truth-prefix "" --join-col variant_id

    # 离线冒烟测试（合成数据）
    python scripts/evaluate.py --smoke-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from breeding.genomic_value import DEFAULT_TRAITS


# ------------------------------------------------------------------ #
# 纯 Python 指标
# ------------------------------------------------------------------ #
def auroc(y_true, y_score) -> float:
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


def auprc(y_true, y_score) -> float:
    """平均精度（按分数降序逐点积分 PR 曲线）。"""
    order = sorted(range(len(y_score)), key=lambda i: -y_score[i])
    tp = fp = 0
    n_pos = sum(y_true)
    if n_pos == 0:
        return float("nan")
    ap = 0.0
    prev_recall = 0.0
    for i in order:
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def confusion(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp, fp, fn, tn


def pearson(x, y) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy)


# ------------------------------------------------------------------ #
# 评估
# ------------------------------------------------------------------ #
def eval_classification(pred_df, lab_df, join_col, threshold) -> None:
    import polars as pl

    if "effect_score" not in pred_df.columns:
        raise SystemExit("[错误] 预测 CSV 缺少 effect_score 列（分类任务）。")
    if "label" not in lab_df.columns:
        raise SystemExit("[错误] 标签表缺少 label 列。")
    merged = pred_df.join(lab_df.select([join_col, "label"]), on=join_col, how="inner")
    merged = merged.filter(pl.col("label").is_in([0, 1]))
    if merged.height == 0:
        raise SystemExit("[错误] 合并后无有效（label∈{0,1}）样本。")
    y = merged["label"].to_list()
    s = merged["effect_score"].to_list()
    p = [1 if v > threshold else 0 for v in s]
    tp, fp, fn, tn = confusion(y, p)
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and (prec + rec) else float("nan"))

    print(f"\n========== 分类评估（n={len(y)}，阈值={threshold}）==========")
    print(f"  正样本(不利)={sum(y)}  负样本(有利/中性)={len(y)-sum(y)}")
    print(f"  AUROC      = {auroc(y, s):.4f}")
    print(f"  AUPRC(AP)  = {auprc(y, s):.4f}")
    print(f"  Accuracy   = {acc:.4f}")
    print(f"  Precision  = {prec:.4f}")
    print(f"  Recall     = {rec:.4f}")
    print(f"  F1         = {f1:.4f}")
    print(f"  混淆矩阵: TP={tp} FP={fp} FN={fn} TN={tn}")
    a = auroc(y, s)
    if a == a:
        if a >= 0.70:
            print("  [判读] AUROC≥0.70：探针能有效区分不利/中性变异。")
        elif a >= 0.60:
            print("  [判读] 0.60≤AUROC<0.70：有信号但偏弱，"
                  "多半受标签噪声限制（参考：GWAS p 值标注通常优于区间标注）。")
        else:
            print("  [判读] AUROC<0.60：信号弱，建议优先改善标签质量"
                  "（标签质量 > 基因组覆盖）。")


def eval_regression(pred_df, lab_df, join_col, truth_prefix) -> None:
    import polars as pl

    eff_cols = [c for c in pred_df.columns if c.startswith("effect_")]
    eff_cols = [c for c in eff_cols if c != "effect_score"]
    if not eff_cols:
        raise SystemExit("[错误] 预测 CSV 无 effect_<trait> 列（回归任务）。")
    traits = [c[len("effect_"):] for c in eff_cols]
    merged = pred_df.join(lab_df, on=join_col, how="inner")
    if merged.height == 0:
        raise SystemExit("[错误] 合并后无样本（检查 --join-col）。")

    print(f"\n========== 回归评估（n={merged.height}）==========")
    mses, rs = [], []
    for t in traits:
        truth_col = f"{truth_prefix}{t}"
        if truth_col not in merged.columns:
            print(f"  {t:16s}  (无真值列 {truth_col}，跳过)")
            continue
        sub = merged.select([f"effect_{t}", truth_col]).drop_nulls()
        if sub.height == 0:
            continue
        pred = sub[f"effect_{t}"].to_list()
        truth = [float(v) for v in sub[truth_col].to_list()]
        mse = sum((a - b) ** 2 for a, b in zip(pred, truth)) / len(pred)
        r = pearson(pred, truth)
        mses.append(mse); rs.append(r)
        print(f"  {t:16s}  MSE={mse:.4f}  Pearson_r={r:.4f}  (n={len(pred)})")
    if mses:
        print(f"  ----\n  平均 MSE={sum(mses)/len(mses):.4f}  "
              f"平均 r={sum(r for r in rs if r==r)/max(1,sum(1 for r in rs if r==r)):.4f}")


def run_smoke() -> None:
    import polars as pl
    import random
    rng = random.Random(0)
    n = 200
    # 合成分类：score 与 label 正相关
    lab = [rng.randint(0, 1) for _ in range(n)]
    score = [min(1.0, max(0.0, 0.5 + (0.25 if l else -0.25) + rng.gauss(0, 0.2)))
             for l in lab]
    vid = [f"v{i}" for i in range(n)]
    pred = pl.DataFrame({"variant_id": vid, "effect_score": score})
    labels = pl.DataFrame({"variant_id": vid, "label": lab})
    eval_classification(pred, labels, "variant_id", 0.5)

    # 合成回归：两个性状
    eff = {"variant_id": vid}
    truth = {"variant_id": vid}
    for t in DEFAULT_TRAITS[:2]:
        base = [rng.gauss(0, 1) for _ in range(n)]
        eff[f"effect_{t}"] = base
        truth[t] = [b + rng.gauss(0, 0.3) for b in base]
    eval_regression(pl.DataFrame(eff), pl.DataFrame(truth), "variant_id", "")
    print("\n[冒烟测试] 通过：分类 + 回归评估均跑通。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="predict.py 产出的预测 CSV")
    ap.add_argument("--labels", help="真值表 parquet/csv（含 label 或性状真值列）")
    ap.add_argument("--task", choices=["classification", "regression"],
                    default="classification")
    ap.add_argument("--join-col", default="variant_id", help="连接键（默认 variant_id）")
    ap.add_argument("--threshold", type=float, default=0.5, help="分类判定阈值")
    ap.add_argument("--truth-prefix", default="",
                    help="回归真值列前缀（真值列名 = 前缀+性状名）")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    if args.smoke_test:
        run_smoke()
        return
    if not (args.pred and args.labels):
        raise SystemExit("[错误] 需 --pred 与 --labels（或 --smoke-test）。")

    import polars as pl
    pred_df = (pl.read_csv(args.pred) if args.pred.endswith(".csv")
               else pl.read_parquet(args.pred))
    lab_df = (pl.read_parquet(args.labels) if args.labels.endswith(".parquet")
              else pl.read_csv(args.labels))

    if args.task == "classification":
        eval_classification(pred_df, lab_df, args.join_col, args.threshold)
    else:
        eval_regression(pred_df, lab_df, args.join_col, args.truth_prefix)


if __name__ == "__main__":
    main()
