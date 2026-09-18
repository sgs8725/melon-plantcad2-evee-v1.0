#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_phenotype_labels.py
=================================
把甜瓜公开表型 / GWAS 结果整理成训练标签，喂给协方差探针与育种模型。

背景：Zhao et al. 2019（1175 份变异图谱，16 性状 GWAS）、Liu et al. 2020
（297 份重测序，果实大小/肉厚/香气 GWAS）、NPGS 核心集（Akter et al. 2023，
性别表达/果肉 SSC 等）、Esteras/Leida 177 份多样性面板等，表型多以论文
**补充材料表（Excel/CSV，按材料编号 accession 组织）** 形式发布，需要按
accession 与基因型/变异对齐后才能作监督信号。本脚本提供两种模式：

  --mode panel  （默认）
    输入：按 accession 组织的表型表（每行一份材料，列为各性状）。
    处理：把原始性状列名映射到本仓库的 DEFAULT_TRAITS；数值清洗；可选
          z-score 标准化（回归）或中位数二分（分类）。
    产出：data/phenotype_labels.parquet —— 每份材料一行的整洁标签表，
          供 (a) 育种模型材料级育种值的验证 / 回归监督，
              (b) 按 accession 把表型贴给携带该等位的变异。

  --mode gwas
    输入：GWAS 显著位点表（含 chrom,pos[,trait,effect/beta,pvalue]）。
    处理：按 chrom+pos 连接到已构建的变异 parquet（build_variant_dataset.py
          产出），据效应方向与显著性回填 label / trait_label。
    产出：带标签的变异 parquet（label: 1=不利 0=有利/中性 -1=未知；
          trait_label: 命中的性状，逗号分隔），直接供 train_probe.py 使用。
          （对应技术方案 4.3 节“GWAS 显著位点 → 有利/不利效应标签”。）

用法：
    # 面板表型 -> 整洁标签表
    python scripts/build_phenotype_labels.py --mode panel \\
        --pheno data/phenotype/Zhao2019_traits.csv \\
        --accession-col accession --out data/phenotype_labels.parquet \\
        --binarize median

    # GWAS 命中 -> 回填变异 parquet 标签
    python scripts/build_phenotype_labels.py --mode gwas \\
        --gwas data/phenotype/Zhao2019_gwas_hits.csv \\
        --variants data/variants.parquet \\
        --out data/variants_labeled.parquet

    # 离线冒烟测试（合成小表，无需任何外部文件 / 网络）
    python scripts/build_phenotype_labels.py --smoke-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from breeding.genomic_value import DEFAULT_TRAITS

# ------------------------------------------------------------------ #
# 原始性状列名 -> 本仓库 DEFAULT_TRAITS 的默认映射（不区分大小写、可被
# --trait-map 覆盖）。覆盖甜瓜公开数据常见的列名写法。
# ------------------------------------------------------------------ #
DEFAULT_TRAIT_MAP = {
    # 产量 / 单果重
    "yield": "yield", "fruit_yield": "yield", "plot_yield": "yield",
    "fruit_weight": "fruit_weight", "fruit_mass": "fruit_weight",
    "fw": "fruit_weight", "single_fruit_weight": "fruit_weight",
    "fruit_size": "fruit_weight",
    # 糖度 / 含糖量
    "sugar_content": "sugar_content", "ssc": "sugar_content",
    "brix": "sugar_content", "tss": "sugar_content",
    "soluble_solids": "sugar_content", "soluble_solid_content": "sugar_content",
    "sugar": "sugar_content", "sucrose": "sugar_content",
    # 果肉色
    "flesh_color": "flesh_color", "flesh_colour": "flesh_color",
    "beta_carotene": "flesh_color", "carotenoid": "flesh_color",
    "flesh_color_intensity": "flesh_color",
    # 香气
    "aroma": "aroma", "volatiles": "aroma", "ester": "aroma",
    "esters": "aroma", "aroma_intensity": "aroma",
    # 耐贮 / 货架期 / 成熟
    "shelf_life": "shelf_life", "storability": "shelf_life",
    "firmness": "shelf_life", "fruit_firmness": "shelf_life",
    "climacteric": "shelf_life", "ripening": "shelf_life",
    # 抗病
    "disease_resist": "disease_resist", "disease_resistance": "disease_resist",
    "powdery_mildew": "disease_resist", "fusarium": "disease_resist",
    "fom": "disease_resist", "vat": "disease_resist", "virus": "disease_resist",
    # 抗逆
    "stress_tolerance": "stress_tolerance", "cracking": "stress_tolerance",
    "heat_tolerance": "stress_tolerance", "salt_tolerance": "stress_tolerance",
    "abiotic": "stress_tolerance",
}

# 性状期望方向：True 表示“数值越高越有利”（甜瓜品质/抗性多为越高越好）。
FAVORABLE_HIGH = {t: True for t in DEFAULT_TRAITS}


# ------------------------------------------------------------------ #
# 工具
# ------------------------------------------------------------------ #
def _read_table(path: str):
    """读 CSV/TSV/XLSX 为 polars.DataFrame。xlsx 走 pandas+openpyxl 兜底。"""
    import polars as pl

    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".csv", ".txt"):
        return pl.read_csv(path, infer_schema_length=2000)
    if suf in (".tsv", ".tab"):
        return pl.read_csv(path, separator="\t", infer_schema_length=2000)
    if suf in (".xlsx", ".xls"):
        try:
            return pl.read_excel(path)          # 需 fastexcel
        except Exception:
            import pandas as pd                  # 兜底
            return pl.from_pandas(pd.read_excel(path))
    raise ValueError(f"不支持的表格格式：{suf}（请用 csv/tsv/xlsx）")


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _resolve_trait_map(df_cols, user_map: dict | None) -> dict:
    """返回 {原始列名: 标准性状名}，仅保留能映射到 DEFAULT_TRAITS 的列。"""
    table = dict(DEFAULT_TRAIT_MAP)
    if user_map:
        table.update({_norm(k): v for k, v in user_map.items()})
    resolved = {}
    for c in df_cols:
        std = table.get(_norm(c))
        if std in DEFAULT_TRAITS:
            resolved[c] = std
    return resolved


# ------------------------------------------------------------------ #
# 模式一：面板表型 -> 整洁标签表
# ------------------------------------------------------------------ #
def run_panel(args) -> None:
    import polars as pl

    df = _read_table(args.pheno)
    if args.accession_col not in df.columns:
        raise SystemExit(
            f"[错误] 表中找不到材料编号列 '{args.accession_col}'；"
            f"现有列：{df.columns}")

    user_map = json.loads(Path(args.trait_map).read_text()) if args.trait_map else None
    tmap = _resolve_trait_map([c for c in df.columns if c != args.accession_col],
                              user_map)
    if not tmap:
        raise SystemExit("[错误] 没有任何性状列能映射到 DEFAULT_TRAITS；"
                         "请用 --trait-map 提供映射（JSON：原始列名->标准性状名）。")
    print(f"[映射] {len(tmap)} 个性状列：")
    for raw, std in tmap.items():
        print(f"    {raw}  ->  {std}")

    # 选列 + 数值化（同一标准性状若有多列，取均值）
    std2raw: dict[str, list[str]] = {}
    for raw, std in tmap.items():
        std2raw.setdefault(std, []).append(raw)

    exprs = [pl.col(args.accession_col).alias("accession")]
    for std, raws in std2raw.items():
        cols = [pl.col(r).cast(pl.Float64, strict=False) for r in raws]
        exprs.append((sum(cols) / len(cols)).alias(std))
    out = df.select(exprs)

    trait_cols = list(std2raw.keys())

    # z-score 标准化（回归用）
    if args.normalize:
        for t in trait_cols:
            mean = out[t].mean()
            std = out[t].std()
            if std and std > 0:
                out = out.with_columns(((pl.col(t) - mean) / std).alias(t))
        print("[标准化] 已对各性状做 z-score（回归监督用）。")

    # 二分（分类用）：median 中位数分割 / 也支持给定阈值
    if args.binarize:
        for t in trait_cols:
            if args.binarize == "median":
                thr = out[t].median()
            else:
                thr = float(args.binarize)
            # 期望越高越好：高于阈值 -> 1（有利），否则 0
            hi = FAVORABLE_HIGH.get(t, True)
            expr = (pl.col(t) >= thr) if hi else (pl.col(t) <= thr)
            out = out.with_columns(expr.cast(pl.Int8).alias(f"{t}_cls"))
        print(f"[二分] 已按 {args.binarize} 生成 *_cls 分类标签"
              f"（1=有利方向）。")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(outp)
    print(f"[完成] 材料级表型标签 -> {outp}  "
          f"（{out.height} 份材料 × {len(trait_cols)} 性状）")
    print("[用途] (a) 育种模型 rank_candidates 的真值验证 / 回归监督；"
          "(b) 按 accession 贴给携带该等位的变异。")


# ------------------------------------------------------------------ #
# 模式二：GWAS 命中 -> 回填变异 parquet 标签
# ------------------------------------------------------------------ #
def run_gwas(args) -> None:
    import polars as pl

    hits = _read_table(args.gwas)
    var = pl.read_parquet(args.variants)
    for c in ("chrom", "pos"):
        if c not in var.columns:
            raise SystemExit(f"[错误] 变异 parquet 缺少列 '{c}'。")

    # GWAS 表列名归一
    cols = {_norm(c): c for c in hits.columns}
    chrom_c = cols.get("chrom") or cols.get("chr") or cols.get("chromosome")
    pos_c = cols.get("pos") or cols.get("position") or cols.get("bp")
    if not (chrom_c and pos_c):
        raise SystemExit("[错误] GWAS 表需含 chrom/chr 与 pos/position 列。")
    trait_c = cols.get("trait") or cols.get("phenotype")
    eff_c = cols.get("effect") or cols.get("beta") or cols.get("slope")
    p_c = cols.get("pvalue") or cols.get("p") or cols.get("p_value")

    user_map = json.loads(Path(args.trait_map).read_text()) if args.trait_map else None

    h = hits.rename({chrom_c: "chrom", pos_c: "pos"})
    h = h.with_columns([pl.col("chrom").cast(pl.Utf8),
                        pl.col("pos").cast(pl.Int64, strict=False)])
    # 显著性过滤
    if p_c and args.pmax is not None:
        h = h.filter(pl.col(p_c).cast(pl.Float64, strict=False) <= args.pmax)

    # 标准化 trait 名
    def std_trait(name):
        table = dict(DEFAULT_TRAIT_MAP)
        if user_map:
            table.update({_norm(k): v for k, v in user_map.items()})
        return table.get(_norm(str(name)), str(name))

    # 逐命中位点 -> (chrom,pos) -> (label, trait)
    label_map: dict[tuple, int] = {}
    trait_map: dict[tuple, set] = {}
    for row in h.iter_rows(named=True):
        key = (str(row["chrom"]), int(row["pos"]))
        tr = std_trait(row[trait_c]) if trait_c else ""
        if tr:
            trait_map.setdefault(key, set()).add(tr)
        if eff_c is not None and row.get(eff_c) is not None:
            try:
                eff = float(row[eff_c])
            except (TypeError, ValueError):
                eff = None
            if eff is not None:
                hi = FAVORABLE_HIGH.get(tr, True)
                favorable = (eff > 0) if hi else (eff < 0)
                label_map[key] = 0 if favorable else 1   # 0=有利,1=不利

    var = var.with_columns([pl.col("chrom").cast(pl.Utf8),
                            pl.col("pos").cast(pl.Int64)])

    def lab(c, p):
        return label_map.get((str(c), int(p)), None)

    def trl(c, p):
        s = trait_map.get((str(c), int(p)))
        return ",".join(sorted(s)) if s else None

    new_label, new_trait = [], []
    for c, p, old_lab, old_trl in zip(
        var["chrom"], var["pos"],
        var["label"] if "label" in var.columns else [None] * var.height,
        var["trait_label"] if "trait_label" in var.columns else [""] * var.height,
    ):
        nl = lab(c, p)
        new_label.append(nl if nl is not None else (old_lab if old_lab is not None else -1))
        nt = trl(c, p)
        new_trait.append(nt if nt else (old_trl or ""))

    var = var.with_columns([pl.Series("label", new_label, dtype=pl.Int64),
                            pl.Series("trait_label", new_trait, dtype=pl.Utf8)])
    n_hit = sum(1 for x in new_label if x in (0, 1))
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    var.write_parquet(outp)
    print(f"[完成] 已回填标签的变异数据集 -> {outp}")
    print(f"       命中并赋有利/不利标签的变异：{n_hit} / {var.height}")
    print("[用途] 直接作 train_probe.py 的 --labels（分类模式）。")


# ------------------------------------------------------------------ #
# 冒烟测试：合成小表跑通两种模式
# ------------------------------------------------------------------ #
def run_smoke() -> None:
    import polars as pl
    import tempfile, os

    tmp = Path(tempfile.mkdtemp(prefix="phlab_"))
    print(f"[冒烟测试] 工作目录 {tmp}")

    # 1) 合成面板表型表（列名故意用 SSC/Brix/fruit_mass 等别名）
    pheno = pl.DataFrame({
        "accession": [f"M{i:03d}" for i in range(6)],
        "SSC": [8.1, 12.4, 10.0, 14.2, 9.3, 11.1],
        "fruit_mass": [900, 1500, 1200, 1800, 1000, 1350],
        "aroma_intensity": [2.0, 4.5, 3.1, 5.0, 2.4, 3.8],
        "powdery_mildew": [1, 3, 2, 4, 1, 3],
    })
    pcsv = tmp / "pheno.csv"; pheno.write_csv(pcsv)

    class A:  # 简易 args
        pass
    a = A(); a.pheno = str(pcsv); a.accession_col = "accession"
    a.trait_map = None; a.normalize = False; a.binarize = "median"
    a.out = str(tmp / "phenotype_labels.parquet")
    print("\n--- 模式 panel ---")
    run_panel(a)
    lab = pl.read_parquet(a.out)
    print(lab)

    # 2) 合成一个小的变异 parquet + GWAS 命中表
    var = pl.DataFrame({
        "variant_id": ["v1", "v2", "v3"],
        "chrom": ["1", "1", "2"],
        "pos": [1000, 2000, 3000],
        "label": [-1, -1, -1],
        "trait_label": ["", "", ""],
    })
    vp = tmp / "variants.parquet"; var.write_parquet(vp)
    gwas = pl.DataFrame({
        "chrom": ["1", "2"],
        "pos": [1000, 3000],
        "trait": ["Brix", "fruit_mass"],
        "beta": [0.8, -0.5],
        "pvalue": [1e-9, 1e-7],
    })
    gcsv = tmp / "gwas.csv"; gwas.write_csv(gcsv)
    b = A(); b.gwas = str(gcsv); b.variants = str(vp); b.trait_map = None
    b.pmax = 1e-5; b.out = str(tmp / "variants_labeled.parquet")
    print("\n--- 模式 gwas ---")
    run_gwas(b)
    print(pl.read_parquet(b.out))
    print("\n[冒烟测试] 通过：panel + gwas 两种模式均跑通。")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="甜瓜表型 / GWAS -> 训练标签",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["panel", "gwas"], default="panel")
    # panel
    ap.add_argument("--pheno", help="按 accession 组织的表型表 csv/tsv/xlsx")
    ap.add_argument("--accession-col", default="accession",
                    help="材料编号列名（默认 accession）")
    ap.add_argument("--normalize", action="store_true",
                    help="对各性状做 z-score 标准化（回归监督用）")
    ap.add_argument("--binarize", default=None,
                    help="二分阈值：'median' 或一个数值；生成 *_cls 分类标签")
    # gwas
    ap.add_argument("--gwas", help="GWAS 命中表 csv/tsv/xlsx（chrom,pos[,trait,effect,pvalue]）")
    ap.add_argument("--variants", help="待回填标签的变异 parquet")
    ap.add_argument("--pmax", type=float, default=1e-5,
                    help="GWAS 显著性 p 值上限（默认 1e-5）")
    # 通用
    ap.add_argument("--trait-map", default=None,
                    help="JSON 文件：自定义{原始列名/性状名->标准性状名}映射")
    ap.add_argument("--out", help="输出路径")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    if args.smoke_test:
        run_smoke()
        return
    if not args.out:
        raise SystemExit("[错误] 需提供 --out。")
    if args.mode == "panel":
        if not args.pheno:
            raise SystemExit("[错误] panel 模式需 --pheno。")
        run_panel(args)
    else:
        if not (args.gwas and args.variants):
            raise SystemExit("[错误] gwas 模式需 --gwas 与 --variants。")
        run_gwas(args)


if __name__ == "__main__":
    main()
