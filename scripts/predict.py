#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/predict.py
==================
端到端预测：变异效应预测 + PlantCAD2 零样本 LLR + 可解释扰动谱 +
LLM 机制解释 + 育种排序。

对每个变异：
  1. 协方差探针      -> 变异效应分数（致病性概率 或 多性状效应）
  2. PlantCAD2 LLR   -> 零样本掩码似然比（互补的方向性参考，仅 SNV）
  3. 注释探针        -> 扰动谱（Top-10 被扰动的功能注释）
  4. LLM 合成        -> 自然语言育种机制解释
并把同一材料的多个变异聚合为材料级育种值排序。

用法：
    python scripts/predict.py \
        --variants data/variants.parquet \
        --activations data/activations \
        --probe assets/probe_covariance.safetensors \
        --out results/predictions.csv \
        [--annotation-probe assets/annotation_probe.safetensors] \
        [--model assets/plantcad2] [--explain]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probes.covariance_probe import CovarianceProbe
from probes.annotation_probe import (
    AnnotationProbe, disruption_profile, format_disruption_table,
)
from probes.llm_synthesis import explain_variant
from breeding.genomic_value import DEFAULT_TRAITS


def load_probe(path: Path) -> tuple[CovarianceProbe, dict]:
    import safetensors.torch
    meta = json.loads(path.with_suffix(".json").read_text())
    probe = CovarianceProbe(
        d_model=meta["d_model"], d_hidden=meta["d_hidden"],
        d_probe=meta["d_probe"], n_outputs=meta["n_outputs"],
    )
    safetensors.torch.load_model(probe, str(path))
    probe.eval()
    return probe, meta


def load_activation(act_dir: Path, vid: str) -> torch.Tensor | None:
    import safetensors.torch
    f = act_dir / f"{vid.replace(':', '_').replace('>', '_')}.safetensors"
    if not f.exists():
        return None
    return safetensors.torch.load_file(str(f))["activations"].float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="变异 parquet")
    ap.add_argument("--activations", required=True, help="激活缓存目录")
    ap.add_argument("--probe", required=True, help="协方差探针 .safetensors")
    ap.add_argument("--annotation-probe", default=None,
                    help="注释探针 .safetensors（提供则输出扰动谱）")
    ap.add_argument("--model", default="assets/plantcad2",
                    help="PlantCAD2 权重（注释探针需要它提供参考嵌入）")
    ap.add_argument("--out", required=True, help="预测结果 CSV")
    ap.add_argument("--explain", action="store_true",
                    help="调用 LLM 合成机制解释（需 ANTHROPIC_API_KEY）")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    import polars as pl

    df = pl.read_parquet(args.variants)
    act_dir = Path(args.activations)
    probe, meta = load_probe(Path(args.probe))
    task = meta["task"]
    print(f"[探针] task={task}  n_outputs={meta['n_outputs']}")

    # ---- 可选：读取预计算的零样本 LLR ----
    llr_map = {}
    llr_file = act_dir / "llr_scores.json"
    if llr_file.exists():
        llr_map = json.loads(llr_file.read_text())
        print(f"[LLR] 载入 {len(llr_map)} 条零样本掩码似然比。")

    # ---- 注释探针（可选）----
    ann_probe = None
    encoder = None
    ann_regression = False
    if args.annotation_probe and Path(args.annotation_probe).exists():
        import safetensors.torch
        # 读注释探针自带的配置 sidecar（类别列表 + d_model），据此重建结构，
        # 以兼容“结构注释”探针(17 类)与“实验轨道扩展”探针(任意类集)。
        ann_path = Path(args.annotation_probe)
        ann_meta = {}
        if ann_path.with_suffix(".json").exists():
            ann_meta = json.loads(ann_path.with_suffix(".json").read_text())
        ann_d_model = int(ann_meta.get("d_model", meta["d_model"]))
        ann_list = ann_meta.get("annotations")     # None -> 用默认全panel
        ann_probe = AnnotationProbe(d_model=ann_d_model, annotations=ann_list)
        safetensors.torch.load_model(ann_probe, str(ann_path))
        ann_probe.eval()
        ann_regression = (ann_meta.get("task") == "regression")
        from meloncad.encoder import MelonCADEncoder, build_mock_encoder
        if args.smoke_test:
            encoder = build_mock_encoder(d_model=ann_d_model)
        else:
            encoder = MelonCADEncoder(model_path=args.model)
        ann_probe = ann_probe.to(encoder.device)
        print(f"[注释探针] 已加载（{len(ann_probe.annotations)} 类），将输出扰动谱。")

    # ---- 逐变异预测 ----
    rows = []
    for r in df.iter_rows(named=True):
        vid = r["variant_id"]
        act = load_activation(act_dir, vid)
        if act is None:
            continue
        act_b = act.unsqueeze(0)                       # [1,2,2,K,d]

        out_row = {
            "variant_id": vid, "gene": r.get("gene", ""),
            "consequence": r.get("consequence", ""),
        }
        if vid in llr_map:
            out_row["llr"] = round(float(llr_map[vid]), 4)

        with torch.no_grad():
            if task == "classification":
                score = float(probe.predict_proba(act_b)[0])
                out_row["effect_score"] = round(score, 4)
                out_row["prediction"] = "不利" if score > 0.5 else "有利/中性"
                effect_summary = f"不利效应概率 = {score:.4f}"
            else:
                traits = probe.predict_traits(act_b)[0]
                signed = (traits - 0.5) * 2.0
                for t, v in zip(DEFAULT_TRAITS, signed.tolist()):
                    out_row[f"effect_{t}"] = round(v, 4)
                effect_summary = "; ".join(
                    f"{t}:{v:+.3f}" for t, v in
                    zip(DEFAULT_TRAITS, signed.tolist()))
        if vid in llr_map:
            effect_summary += f"; 零样本LLR={llr_map[vid]:+.3f}"

        # ---- 扰动谱 + LLM 解释 ----
        if ann_probe is not None and encoder is not None:
            emb_ref = encoder.encode_reference(r["ref_seq"])
            emb_alt = encoder.encode_reference(r["alt_seq"])
            prof = disruption_profile(
                ann_probe, emb_ref, emb_alt,
                var_pos=int(r.get("var_index", 0)),
                regression=ann_regression,
            )
            table = format_disruption_table(prof)
            out_row["top_disruptions"] = "|".join(prof["top_annotations"][:5])
            if args.explain:
                meta_str = (f"基因={r.get('gene','')}, 坐标={vid}, "
                            f"后果类型={r.get('consequence','')}")
                out_row["explanation"] = explain_variant(
                    meta_str, effect_summary, table,
                ).replace("\n", " ⏎ ")
        rows.append(out_row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = pl.DataFrame(rows)
    res.write_csv(out)
    print(f"[完成] 预测结果 -> {out}  （{res.height} 条）")
    print(res.head(8))

    # ---- 材料级育种排序：按 gene 聚合做演示 ----
    if task == "regression" and res.height:
        print("\n[育种排序] 按基因聚合的选择指数（演示）：")
        eff_cols = [c for c in res.columns if c.startswith("effect_")]
        agg = (res.group_by("gene")
                  .agg([pl.col(c).sum() for c in eff_cols])
                  .with_columns(
                      pl.sum_horizontal(eff_cols).alias("selection_index"))
                  .sort("selection_index", descending=True))
        print(agg.head(10))


if __name__ == "__main__":
    main()
