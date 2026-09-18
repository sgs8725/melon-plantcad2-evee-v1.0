#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/predict_annotation_batch.py v2
=======================================
批处理注释探针推理（优化版）。

优化点：
  1. 批量 tokenize：tokenizer(seqs, padding=True) 一次处理整批
  2. 变异数据用 dict 索引：O(1) 查 ref_seq/alt_seq/var_index
  3. 激活文件批量加载

V1→V2 优化：预计速度提升 3-5x，30K 变异从 3-4 小时降至 ~30 分钟。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probes.annotation_probe import AnnotationProbe, disruption_profile


def load_activation(act_dir: Path, vid: str) -> torch.Tensor | None:
    import safetensors.torch
    f = act_dir / f"{vid.replace(chr(58), chr(95)).replace(chr(62), chr(95))}.safetensors"
    if not f.exists():
        return None
    return safetensors.torch.load_file(str(f))["activations"].float()


@torch.no_grad()
def batch_encode_references_v2(
    tokenizer,
    model,
    layer: int,
    sequences: list[str],
    device: torch.device,
) -> list[torch.Tensor]:
    """批量编码：一次 tokenize + 一次前向，替代逐条编码。"""
    # 批量 tokenize（所有序列等长 8192，但可加 padding）
    tokens = tokenizer(
        [s.strip().upper() for s in sequences],
        return_tensors="pt",
        padding=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    batch_ids = tokens["input_ids"].to(device).long()  # [B, L]

    # 单次前向
    out = model(input_ids=batch_ids, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states[layer].float()  # [B, L, 2d]
    d = hs.shape[-1] // 2

    results = []
    for i in range(len(sequences)):
        fwd = hs[i, :, :d]
        rev = hs[i, :, d:].flip(0)
        emb = (fwd + rev) / 2.0
        results.append(emb)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--annotation-probe", required=True)
    ap.add_argument("--model", default="assets/plantcad2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # ---- 加载数据 ----
    pred = pl.read_csv(args.predictions)
    variants = pl.read_parquet(args.variants)
    variants = variants.select(["variant_id", "ref_seq", "alt_seq", "var_index"])

    df = pred.select("variant_id").join(variants, on="variant_id", how="inner")
    if args.limit > 0:
        df = df.head(args.limit)

    # ---- 建 dict 索引 (O(1) 查找) ----
    var_map = {}
    for row in variants.iter_rows(named=True):
        var_map[row["variant_id"]] = {
            "ref_seq": row["ref_seq"],
            "alt_seq": row["alt_seq"],
            "var_index": row["var_index"],
        }

    total = df.height
    print(f"[数据] 待处理: {total} 变异")

    act_dir = Path(args.activations)

    # ---- 加载注释探针 ----
    import safetensors.torch
    ann_path = Path(args.annotation_probe)
    ann_meta = json.loads(ann_path.with_suffix(".json").read_text()) if ann_path.with_suffix(".json").exists() else {}
    ann_d_model = int(ann_meta.get("d_model", 768))
    ann_list = ann_meta.get("annotations")
    ann_probe = AnnotationProbe(d_model=ann_d_model, annotations=ann_list)
    safetensors.torch.load_model(ann_probe, str(ann_path))
    ann_probe.eval()
    ann_regression = (ann_meta.get("task") == "regression")
    print(f"[注释探针] {len(ann_probe.annotations)} 类")

    # ---- 加载编码器（只需模型 + tokenizer） ----
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layer = -1  # 用最后一层
    d_model = int(model.config.d_model)
    ann_probe = ann_probe.to(device)
    print(f"[模型] {d_model}d, device={device}")

    # ---- 逐批处理 ----
    bs = args.batch_size
    rows_later = []
    t0 = time.time()

    for start in range(0, total, bs):
        end = min(start + bs, total)
        batch_vids = df[start:end]["variant_id"].to_list()

        # 筛选有激活文件的
        valid_vids = [v for v in batch_vids if load_activation(act_dir, v) is not None]
        if not valid_vids:
            continue

        # 从 dict 查 seq（O(1) per lookup）
        ref_seqs = [var_map[v]["ref_seq"] for v in valid_vids]
        alt_seqs = [var_map[v]["alt_seq"] for v in valid_vids]

        t1 = time.time()

        # 批处理编码（ref + alt 各一次前向）
        ref_embs = batch_encode_references_v2(tokenizer, model, layer, ref_seqs, device)
        alt_embs = batch_encode_references_v2(tokenizer, model, layer, alt_seqs, device)

        t2 = time.time()

        # 逐变异计算扰动谱
        for i, vid in enumerate(valid_vids):
            var_idx = var_map[vid]["var_index"]
            prof = disruption_profile(
                ann_probe, ref_embs[i], alt_embs[i],
                var_pos=int(var_idx),
                regression=ann_regression,
            )
            rows_later.append({
                "variant_id": vid,
                "top_disruptions": "|".join(prof["top_annotations"][:5]),
            })

        elapsed = time.time() - t0
        done = end
        rate = max(1, done / elapsed)
        eta = (total - done) / rate
        enc_time = t2 - t1
        print(f"  [{done}/{total}]  编码={enc_time:.1f}s  累计={elapsed:.0f}s  "
              f"速率={rate:.0f}项/s  ETA={eta:.0f}s")

    # ---- 合并 ----
    ann_df = pl.DataFrame(rows_later)
    print(f"\n[合并] 注释探针结果: {ann_df.height} 条")
    result = pred.join(ann_df, on="variant_id", how="left") if ann_df.height > 0 else pred
    result.write_csv(args.out)
    print(f"[完成] -> {args.out}  ({result.height} 条)")

    if ann_df.height > 0:
        print("\n样本:")
        print(ann_df.head(5))


if __name__ == "__main__":
    main()
