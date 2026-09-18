#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/extract_activations.py
==============================
用 PlantCAD2 为每个变异提取 EVEE 风格激活张量并缓存到磁盘。

对 data/variants.parquet 中的每条变异，调用 MelonCADEncoder.encode_variant
得到 [2, 2, K, d_model] 张量，按 variant_id 存为单独的 safetensors 文件
（与 EVEE demo notebook 的 artifacts/samples/*.safetensors 布局一致）。
可选地同时缓存 PlantCAD2 零样本掩码似然比（LLR），供 predict.py 输出。

激活一次性算好缓存，后续训练 / 调参探针时反复读取，避免重复跑大模型。

用法：
    python scripts/extract_activations.py \
        --variants data/variants.parquet \
        --model assets/plantcad2 \
        --out data/activations \
        --topk 256 --layer -1 --llr

冒烟测试（无权重，mock 模型）：
    python scripts/extract_activations.py --variants data/variants.parquet \
        --out data/activations --smoke-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meloncad.encoder import MelonCADEncoder, build_mock_encoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="变异 parquet")
    ap.add_argument("--model", default="assets/plantcad2",
                    help="PlantCAD2 权重目录或 HF 仓库名")
    ap.add_argument("--out", required=True, help="激活缓存输出目录")
    ap.add_argument("--topk", type=int, default=256, help="Top-K 发散位置数")
    ap.add_argument("--layer", type=int, default=-1, help="取第几层隐藏状态")
    ap.add_argument("--device", default=None)
    ap.add_argument("--llr", action="store_true",
                    help="同时计算并缓存零样本掩码似然比（仅 SNV）")
    ap.add_argument("--smoke-test", action="store_true",
                    help="用 mock PlantCAD2 跑通流程（无需下载权重）")
    args = ap.parse_args()

    import polars as pl
    import safetensors.torch

    df = pl.read_parquet(args.variants)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 构造编码器 ----
    if args.smoke_test:
        print("[冒烟测试] 使用 mock PlantCAD2（d_model=256）。")
        enc = build_mock_encoder(d_model=256)
    else:
        print(f"[加载] PlantCAD2 权重: {args.model}")
        enc = MelonCADEncoder(
            model_path=args.model, layer=args.layer, device=args.device,
        )
    print(f"[信息] d_model = {enc.d_model}")

    # ---- 逐变异提取 ----
    n = df.height
    llr_map: dict[str, float] = {}
    for i, row in enumerate(df.iter_rows(named=True)):
        vid = row["variant_id"]
        fname = out_dir / f"{vid.replace(':', '_').replace('>', '_')}.safetensors"
        if not fname.exists():
            try:
                act = enc.encode_variant(
                    ref_seq=row["ref_seq"], alt_seq=row["alt_seq"], topk=args.topk,
                )
            except Exception as e:
                print(f"  ⚠️ 跳过 {vid}: {e}")
                continue                                          # [2, 2, K, d]
            safetensors.torch.save_file(
                {"activations": act.contiguous()}, str(fname))
        if args.llr and row.get("consequence") == "SNV":
            try:
                llr_map[vid] = enc.score_variant_llr(
                    row["ref_seq"], row["alt_seq"], int(row.get("var_index", 0)))
            except Exception:
                pass
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}  最新: {fname.name}")

    if args.llr and llr_map:
        (out_dir / "llr_scores.json").write_text(json.dumps(llr_map, indent=2))
        print(f"[完成] LLR 分数 -> {out_dir / 'llr_scores.json'}（{len(llr_map)} 条）")

    print(f"[完成] 激活已缓存到 {out_dir}")


if __name__ == "__main__":
    main()
