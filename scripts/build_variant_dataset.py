#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_variant_dataset.py
================================
从 VCF + 参考基因组 + GFF 注释，构建甜瓜变异-序列数据集（parquet）。

用法：
    python scripts/build_variant_dataset.py \
        --vcf assets/variants/melon_variants.vcf.gz \
        --genome assets/genome/melon_DHL92.fa \
        --gff assets/genome/melon_DHL92.gff3 \
        --out data/variants.parquet \
        --window 8192 --max-variants 50000

产出 parquet 列：variant_id, chrom, pos, ref_allele, alt_allele,
ref_seq, alt_seq, var_index, gene, consequence, label, trait_label。
其中 label/trait_label 初始为占位（-1 / ""），需后续用甜瓜 GWAS / eQTL /
突变群体 / 代谢组（糖谱 Brix、香气酯类）表型回填——可用
scripts/build_phenotype_labels.py 按 accession 或 chrom+pos 对齐生成
（见技术方案 4.3 节）。

窗口长度上限 8192bp（PlantCAD2 硬上限），脚本内部会夹断。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.variant_dataset import build_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True, help="变异 VCF（可 .gz）")
    ap.add_argument("--genome", required=True, help="参考基因组 FASTA")
    ap.add_argument("--gff", default=None, help="GFF3 注释（用于分配基因名）")
    ap.add_argument("--out", required=True, help="输出 parquet 路径")
    ap.add_argument("--window", type=int, default=8192,
                    help="变异窗口长度(bp)，上限 8192")
    ap.add_argument("--max-variants", type=int, default=None,
                    help="最多处理多少变异（调试用）")
    args = ap.parse_args()

    print("[1/3] 读取基因组与注释 ...")
    records = build_dataset(
        vcf_path=args.vcf,
        genome_path=args.genome,
        gff_path=args.gff,
        window=args.window,
        max_variants=args.max_variants,
    )
    print(f"[2/3] 构建变异记录：{len(records)} 条")

    import polars as pl
    df = pl.DataFrame(records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)

    print(f"[3/3] 已写出 {out}")
    print(df.select(["variant_id", "gene", "consequence", "var_index"]).head(5))
    print("\n变异类型分布：")
    print(df["consequence"].value_counts().sort("count", descending=True))


if __name__ == "__main__":
    main()
