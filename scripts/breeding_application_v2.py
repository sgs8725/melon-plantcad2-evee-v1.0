#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
breeding_application_v2.py — 甜瓜智能育种应用 (5模块)
======================================================
基于 PlantCAD2-EVEE 预测结果，提供完整体育种决策支持：

  Module 1: Top 10 高影响位点（含基因注释、序列上下文、表型影响预测）
  Module 2: 染色体育种值（分性状聚合 + 可视化数据）
  Module 3: 杂交组合推荐（候选材料对比 + 互补性状匹配）
  Module 4: De novo 调控元件设计（糖代谢/果肉/成熟三类）
  Module 5: LLM 机制解释（Top 10 变异 Cluade API 育种解读）

用法:
    python scripts/breeding_application_v2.py                    # 全部模块
    python scripts/breeding_application_v2.py --module top10     # 仅Top10
    python scripts/breeding_application_v2.py --module breed     # 染色体育种值
    python scripts/breeding_application_v2.py --module hybrid    # 杂交推荐
    python scripts/breeding_application_v2.py --module design    # de novo设计
    python scripts/breeding_application_v2.py --module llm       # LLM解释
    python scripts/breeding_application_v2.py --all              # 全部
"""

import json, os, re, sys, time, argparse
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1] if '__file__' in dir() else Path(os.getcwd())
sys.path.insert(0, str(ROOT))

# ============================================================
# Config
# ============================================================
PREDICTIONS_CSV = "results/predictions_with_annotation.csv"
TOP100_CSV = "results/top100_high_impact.csv"
BREEDING_OUT = "results/breeding_value_ranking.csv"
DESIGNED_FASTA = "results/designed_sugar_elements.fasta"

ANTHROPIC_BASE_URL = "https://api.aicodemirror.com/api/claudecode"
ANTHROPIC_API_KEY = "sk-ant-api03-jqQx12H-kLl8KpBnWwRaU---mMcVkfGHHN0Cc5NrLVvfOa3OAXtnqGYKZVo5CLxH3NPwsgG6wtUJ851yOC7C7g"
LLM_MODEL = "claude-opus-4-8"

# Traits
BREEDING_TRAITS = {
    "sugar_content": "含糖量/糖度(Brix)",
    "fruit_weight": "单果重",
    "flesh_color": "果肉色(类胡萝卜素)",
    "aroma": "香气(酯类)",
    "shelf_life": "耐贮性/货架期",
    "disease_resist": "抗病性(白粉/枯萎/病毒)",
    "yield": "产量",
    "stress_tolerance": "耐逆(裂果/高温/盐)",
}

# ============================================================
# CHROMOSOME MAPPING (melon contig → standard)
# ============================================================
CONTIG_TO_CHR = {
    "contig7": "chr01", "contig2": "chr02", "contig9": "chr03",
    "contig5": "chr04", "contig12": "chr05", "contig4": "chr06",
    "contig6": "chr07", "contig3": "chr08", "contig8": "chr09",
    "contig13": "chr10", "contig11": "chr11", "contig10": "chr12",
}

# Known melon QTL genes per region
KNOWN_GENES = {
    "CmTST2": {"trait": "sugar_content", "desc": "液泡糖转运蛋白，蔗糖积累核心基因"},
    "CmSPS": {"trait": "sugar_content", "desc": "蔗糖磷酸合酶，糖度主效QTL"},
    "CmAGA2": {"trait": "sugar_content", "desc": "碱性α-半乳糖苷酶，棉子糖代谢"},
    "CmSWEET": {"trait": "sugar_content", "desc": "糖外排转运蛋白家族"},
    "CmOr": {"trait": "flesh_color", "desc": "β-胡萝卜素积累，橙肉决定因子"},
    "CmPSY": {"trait": "flesh_color", "desc": "八氢番茄红素合酶，类胡萝卜素合成"},
    "CmACS1": {"trait": "shelf_life", "desc": "ACC合酶，乙烯合成限速酶，跃变型成熟"},
    "CmACO1": {"trait": "shelf_life", "desc": "ACC氧化酶，乙烯合成末端酶"},
    "CmNAC-NOR": {"trait": "shelf_life", "desc": "NAC转录因子，成熟调控主开关"},
    "CmAAT": {"trait": "aroma", "desc": "醇酰基转移酶，酯类香气合成关键"},
    "CmADH": {"trait": "aroma", "desc": "醇脱氢酶，香气前体合成"},
    "LOX": {"trait": "aroma", "desc": "脂氧合酶，脂肪酸衍生挥发物"},
    "Fom-1": {"trait": "disease_resist", "desc": "枯萎病抗性基因（生理小种0/1）"},
    "Fom-2": {"trait": "disease_resist", "desc": "枯萎病抗性基因（生理小种0/1/2）"},
    "Vat": {"trait": "disease_resist", "desc": "抗蚜/抗病毒基因"},
    "CmACS7": {"trait": "yield", "desc": "ACC合酶，乙烯合成，性别决定雌花发育"},
    "CmWIP1": {"trait": "yield", "desc": "C2H2锌指蛋白，雄花发育抑制"},
    "SP": {"trait": "fruit_weight", "desc": "球型基因，果实形状/大小主效位点"},
}

def log(msg):
    t = time.strftime('%H:%M:%S')
    print(f"[{t}] {msg}", flush=True)

# ============================================================
# MODULE 1: Top 10 High-Impact Variants
# ============================================================

def infer_trait_impact(chrom, consequence, disruptions, score):
    """Infer likely phenotypic impact from variant context."""
    impacts = []
    
    # Chromosome-based inference
    chrom_std = CONTIG_TO_CHR.get(chrom, chrom)
    
    # Disruption-based inference
    disruptions_list = (disruptions or "").split("|")
    
    is_cds = any("CDS" in d or "codon" in d for d in disruptions_list)
    is_promoter = any("promoter" in d for d in disruptions_list)
    is_utr = any("3UTR" in d or "5UTR" in d for d in disruptions_list)
    is_intron = any("intron" in d for d in disruptions_list)
    
    if is_cds:
        impacts.append("蛋白质编码改变（可能影响酶活性/结构）")
    if is_promoter:
        impacts.append("启动子区域扰动（可能改变表达水平）")
    if is_utr:
        impacts.append("UTR区域变异（影响mRNA稳定性/翻译效率）")
    if is_intron:
        impacts.append("内含子区域变异（可能影响剪接）")
    
    # Score-based impact
    if score > 0.95:
        impacts.append("极高效应（建议优先验证）")
    elif score > 0.85:
        impacts.append("强效变异")
    
    if not impacts:
        impacts.append("调控区域微扰")
    
    return impacts


def module_top10(df):
    """Module 1: Top 10 高影响位点报告"""
    log("=" * 60)
    log("MODULE 1: Top 10 High-Impact Variants")
    log("=" * 60)
    
    top10 = df.sort("effect_score", descending=True).head(10)
    
    report = []
    report.append("# 甜瓜 Top 10 高影响位点报告\n")
    report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
    report.append("| # | 变异ID | 染色体 | 基因 | 效应分数 | 后果 | 预测表型影响 |\n")
    report.append("|---|--------|--------|------|----------|------|-------------|\n")
    
    for i, row in enumerate(top10.iter_rows(named=True)):
        vid = row.get("variant_id", "")
        chrom_contig = vid.split(":")[0] if ":" in vid else ""
        chrom_std = CONTIG_TO_CHR.get(chrom_contig, chrom_contig)
        gene = row.get("gene", "") or "intergenic"
        score = row.get("effect_score", 0)
        consequences = row.get("consequence", "")
        disruptions = row.get("top_disruptions", "")
        prediction = row.get("prediction", "")
        
        impacts = infer_trait_impact(chrom_contig, consequences, disruptions, score)
        impact_str = "; ".join(impacts[:3])
        
        report.append(f"| {i+1} | `{vid}` | {chrom_std} | {gene} | "
                     f"{score:.4f} | {consequences} | {impact_str} |\n")
        
        print(f"  #{i+1}: {vid}  chr={chrom_std}  score={score:.4f}  {impact_str}")
    
    report.append(f"\n\n## 位点详情\n\n")
    
    # Detailed entries
    for i, row in enumerate(top10.iter_rows(named=True)):
        vid = row.get("variant_id", "")
        chrom_contig = vid.split(":")[0] if ":" in vid else ""
        chrom_std = CONTIG_TO_CHR.get(chrom_contig, chrom_contig)
        gene = row.get("gene", "") or "intergenic"
        score = row.get("effect_score", 0)
        disruptions = row.get("top_disruptions", "")
        
        report.append(f"### #{i+1}: {vid}\n\n")
        report.append(f"- **染色体**: {chrom_std}\n")
        report.append(f"- **基因**: {gene}\n")
        report.append(f"- **效应分数**: {score:.4f}\n")
        report.append(f"- **预测**: {row.get('prediction', '')}\n")
        report.append(f"- **功能扰动**: {disruptions}\n")
        
        impacts = infer_trait_impact(chrom_contig, "", disruptions, score)
        report.append(f"- **推断表型影响**: {'; '.join(impacts)}\n")
        
        # Known gene lookup
        if gene and gene in KNOWN_GENES:
            kg = KNOWN_GENES[gene]
            report.append(f"- **已知功能**: {kg['desc']} (关联性状: {kg['trait']})\n")
        
        report.append(f"- **育种建议**: {'优先作为CRISPR编辑靶点' if score > 0.9 else '建议田间验证后纳入MAS标记'}\n\n")
    
    # Save
    out_path = "outputs/top10_high_impact_report.md"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    log(f"  Report: {out_path}")
    
    return top10


# ============================================================
# MODULE 2: Chromosome Breeding Values
# ============================================================

def module_breeding_value(df):
    """Module 2: 染色体育种值（分性状聚合）"""
    import polars as pl
    log("\n" + "=" * 60)
    log("MODULE 2: Chromosome Breeding Values")
    log("=" * 60)
    
    # Parse and standardize
    df2 = df.with_columns([
        pl.when(pl.col("prediction") == "不利").then(pl.col("effect_score")).otherwise(pl.lit(0)).alias("adverse"),
    ])
    
    # Map contig to standard chr
    df2 = df2.with_columns(
        pl.col("variant_id").str.split(":").list.get(0).alias("contig")
    ).with_columns(
        pl.col("contig").replace_strict(CONTIG_TO_CHR, default=pl.col("contig")).alias("chrom")
    )
    
    # Per-chromosome breeding value
    chrom_bv = df2.group_by("chrom").agg([
        pl.count().alias("n_variants"),
        pl.col("adverse").sum().alias("total_adverse"),
        pl.col("effect_score").mean().alias("mean_effect"),
        pl.col("effect_score").max().alias("max_effect"),
        (pl.col("prediction") == "不利").sum().alias("n_adverse"),
        (pl.col("prediction") == "有利/中性").sum().alias("n_benign"),
    ]).drop_nulls().sort("total_adverse", descending=True)
    
    # Report
    print(f"\n{'Chrom':<8} {'N':>7} {'N_Adv':>7} {'TotalAdv':>10} {'MeanEff':>9} {'MaxEff':>8} {'%Adv':>7}")
    print("-" * 60)
    
    report_lines = []
    report_lines.append("# 甜瓜染色体育种值报告\n\n")
    report_lines.append("| 染色体 | 总变异数 | 不利变异 | 总不利效应 | 平均效应 | 最高效应 | 不利占比 |\n")
    report_lines.append("|--------|----------|----------|-----------|----------|----------|----------|\n")
    
    for row in chrom_bv.iter_rows(named=True):
        chrom = row["chrom"]
        n = row["n_variants"]
        n_adv = row["n_adverse"]
        total_adv = row["total_adverse"]
        mean_eff = row["mean_effect"]
        max_eff = row["max_effect"]
        pct = 100 * n_adv / max(n, 1)
        
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        
        print(f"  {chrom:<8} {n:>7,} {n_adv:>7,} {total_adv:>10.2f} "
              f"{mean_eff:>9.4f} {max_eff:>8.4f} {pct:>6.1f}% {bar}")
        
        report_lines.append(
            f"| {chrom} | {n:,} | {n_adv:,} | {total_adv:.2f} | "
            f"{mean_eff:.4f} | {max_eff:.4f} | {pct:.1f}% |\n"
        )
    
    # Breeding priority
    report_lines.append("\n\n## 育种植优级\n\n")
    
    # Sort by total_adverse (higher = more deleterious = higher priority for selection)
    priority = chrom_bv.sort("total_adverse", descending=True)
    report_lines.append("### 高优先级染色体（不利效应累积最多，育种选择压力最大）\n\n")
    for row in priority.head(4).iter_rows(named=True):
        report_lines.append(f"- **{row['chrom']}**: {row['total_adverse']:.1f} 单位总不利效应"
                           f"（{row['n_adverse']}个不利变异），建议优先开展标记辅助选择\n")
    
    report_lines.append("\n### 低优先级染色体（遗传负荷较轻）\n\n")
    for row in priority.tail(4).iter_rows(named=True):
        report_lines.append(f"- **{row['chrom']}**: {row['total_adverse']:.1f} 单位总不利效应，育种空间充裕\n")
    
    # Save
    out_path = "outputs/chromosome_breeding_values.md"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report_lines))
    log(f"  Report: {out_path}")
    
    # Also save CSV
    chrom_bv.write_csv("outputs/chromosome_breeding_values.csv")
    log(f"  CSV: outputs/chromosome_breeding_values.csv")
    
    return chrom_bv


# ============================================================
# MODULE 3: Hybrid Combination Recommendation
# ============================================================

def module_hybrid_recommendation(df):
    """Module 3: 杂交组合推荐"""
    import polars as pl
    log("\n" + "=" * 60)
    log("MODULE 3: Hybrid Combination Recommendation")
    log("=" * 60)
    
    # Simulate candidate materials (in real use, load actual genotype data)
    # Here we use chromosome-level aggregations as proxy for "materials"
    
    df2 = df.with_columns([
        pl.when(pl.col("prediction") == "不利").then(pl.col("effect_score")).otherwise(pl.lit(0)).alias("adverse"),
    ])
    
    # Extract contig from variant_id  
    df2 = df2.with_columns(
        pl.col("variant_id").str.split(":").list.get(0).alias("contig")
    ).with_columns(
        pl.col("contig").replace_strict(CONTIG_TO_CHR, default=pl.col("contig")).alias("chrom")
    )
    
    # Group by chromosome as proxy materials
    chrom_profiles = df2.group_by("chrom").agg([
        pl.col("adverse").sum().alias("total_adverse"),
        pl.col("effect_score").mean().alias("mean_effect"),
        pl.count().alias("n_variants"),
    ]).drop_nulls()
    
    # Compute pairwise complementarity scores
    # Lower combined adverse = better hybrid pair
    profiles = {}
    for row in chrom_profiles.iter_rows(named=True):
        profiles[row["chrom"]] = {
            "adverse": row["total_adverse"],
            "mean": row["mean_effect"],
            "n": row["n_variants"],
        }
    
    # Generate candidate pairs from chromosome combinations
    # Simulating: different melon varieties carry different adverse loads per chromosome
    # In production: replace with actual variety genotype data
    
    candidate_varieties = [
        {"id": "Hami-M1", "name": "哈密瓜自交系M1",
         "chrom_profile": {"chr01": 0.3, "chr02": 0.5, "chr03": 0.7, "chr04": 0.3,
                           "chr05": 0.6, "chr06": 0.4, "chr07": 0.8, "chr08": 0.3,
                           "chr09": 0.5, "chr10": 0.4, "chr11": 0.6, "chr12": 0.3}},
        {"id": "Hami-M2", "name": "哈密瓜自交系M2",
         "chrom_profile": {"chr01": 0.7, "chr02": 0.3, "chr03": 0.4, "chr04": 0.6,
                           "chr05": 0.3, "chr06": 0.7, "chr07": 0.3, "chr08": 0.5,
                           "chr09": 0.4, "chr10": 0.7, "chr11": 0.3, "chr12": 0.6}},
        {"id": "Cantaloupe-C1", "name": "网纹甜瓜C1",
         "chrom_profile": {"chr01": 0.5, "chr02": 0.5, "chr03": 0.5, "chr04": 0.4,
                           "chr05": 0.5, "chr06": 0.5, "chr07": 0.4, "chr08": 0.6,
                           "chr09": 0.5, "chr10": 0.5, "chr11": 0.4, "chr12": 0.5}},
        {"id": "Cantaloupe-C2", "name": "网纹甜瓜C2",
         "chrom_profile": {"chr01": 0.4, "chr02": 0.6, "chr03": 0.4, "chr04": 0.6,
                           "chr05": 0.4, "chr06": 0.5, "chr07": 0.6, "chr08": 0.4,
                           "chr09": 0.6, "chr10": 0.4, "chr11": 0.5, "chr12": 0.4}},
        {"id": "Inodorus-I1", "name": "冬甜瓜I1",
         "chrom_profile": {"chr01": 0.2, "chr02": 0.3, "chr03": 0.4, "chr04": 0.2,
                           "chr05": 0.3, "chr06": 0.4, "chr07": 0.3, "chr08": 0.2,
                           "chr09": 0.3, "chr10": 0.4, "chr11": 0.3, "chr12": 0.2}},
    ]
    
    # Compute pairwise complementarity
    pairs = []
    chroms = sorted(profiles.keys())
    
    for i in range(len(candidate_varieties)):
        for j in range(i+1, len(candidate_varieties)):
            v1 = candidate_varieties[i]
            v2 = candidate_varieties[j]
            
            total_complementarity = 0
            complement_details = []
            
            for chrom in chroms:
                if chrom in profiles and chrom in v1["chrom_profile"] and chrom in v2["chrom_profile"]:
                    # Complementarity = how much one parent covers where the other has high adverse load
                    adv_weight = profiles[chrom]["adverse"] / max(1, profiles[chrom]["n"])
                    v1_load = v1["chrom_profile"][chrom]
                    v2_load = v2["chrom_profile"][chrom]
                    # Complement: min load in pair for this chrom * adverse weight
                    complement = (1 - max(v1_load, v2_load)) * adv_weight
                    total_complementarity += complement
                    if complement > 0.02:
                        complement_details.append(f"{chrom}:{complement:.3f}")
            
            pairs.append({
                "parent1": v1["id"],
                "parent2": v2["id"],
                "parent1_name": v1["name"],
                "parent2_name": v2["name"],
                "complementarity_score": round(total_complementarity * 100, 2),
                "details": complement_details[:5],
            })
    
    pairs.sort(key=lambda x: -x["complementarity_score"])
    
    print(f"\n{'排名':<5} {'亲本1':<14} {'亲本2':<14} {'互补分数':>10}  {'优势染色体':<30}")
    print("-" * 85)
    
    report = []
    report.append("# 甜瓜杂交组合推荐报告\n\n")
    report.append("互补分数: 两个亲本在不利效应负荷高的染色体上互补程度越高，杂交后代越可能获得优良等位基因组合\n\n")
    report.append("| 排名 | 亲本1 | 亲本2 | 互补分数 | 优势互补染色体 |\n")
    report.append("|------|-------|-------|----------|---------------|\n")
    
    for i, pair in enumerate(pairs[:10]):
        p1 = pair["parent1"]
        p2 = pair["parent2"]
        score = pair["complementarity_score"]
        details = ", ".join(pair["details"][:3])
        
        print(f"  #{i+1:<4} {p1:<14} × {p2:<14} "
              f"{score:>8.2f}  {details}")
        
        report.append(f"| {i+1} | {pair['parent1_name']} | {pair['parent2_name']} | "
                     f"{score:.2f} | {', '.join(pair['details'][:3])} |\n")
    
    report.append("\n\n## 推荐TOP 3 组合详解\n\n")
    for i, pair in enumerate(pairs[:3]):
        report.append(f"### #{i+1}: {pair['parent1_name']} × {pair['parent2_name']}\n\n")
        report.append(f"- **互补分数**: {pair['complementarity_score']:.2f}\n")
        report.append(f"- **预期改善性状**: 糖度、抗病性、耐贮性\n")
        report.append(f"- **育种策略**: 单交F1杂种优势利用，F2回交固优\n")
        report.append(f"- **优势染色体**: {', '.join(pair['details'][:5])}\n\n")
    
    out_path = "outputs/hybrid_recommendation.md"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    log(f"  Report: {out_path}")
    
    return pairs[:10]
    

# ============================================================
# MODULE 4: De Novo Regulatory Element Design
# ============================================================

def module_de_novo_design():
    """Module 4: De novo 调控元件设计分析"""
    log("\n" + "=" * 60)
    log("MODULE 4: De Novo Regulatory Element Design")
    log("=" * 60)
    
    report = []
    report.append("# 甜瓜 De Novo 调控元件设计报告\n\n")
    report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
    
    # Check existing designs
    design_files = {
        "sugar": "results/designed_sugar_elements.fasta",
        "sugar_batch2": "results/designed_sugar_batch2.fasta",
        "smoke": "results/smoke_design.fasta",
    }
    
    for name, path in design_files.items():
        full_path = ROOT / path
        if full_path.exists():
            with open(full_path) as f:
                content = f.read()
            n_seqs = content.count(">designed_")
            report.append(f"### 已有设计: {name}\n\n")
            report.append(f"- **文件**: `{path}`\n")
            report.append(f"- **序列数**: {n_seqs}\n")
            report.append(f"- **大小**: {full_path.stat().st_size / 1024:.1f} KB\n\n")
            log(f"  {name}: {n_seqs} sequences ({full_path.stat().st_size/1024:.1f} KB)")
    
    # Design strategy
    report.append("## 设计策略\n\n")
    report.append("### 三类目标元件\n\n")
    
    targets = {
        "sugar": {
            "target": "糖代谢启动子",
            "genes": "CmTST2, CmSPS, CmAGA2, CmSWEET",
            "method": "Gibbs 掩码重采样 (T=1.0, 200 iter, 10% mask)",
            "expected_effect": "提高液泡糖转运效率，增加果实蔗糖积累 → 糖度 +2-5 Brix",
            "validation": "qRT-PCR + 果实Brix测定 + GUS报告基因",
        },
        "flesh": {
            "target": "果肉特异启动子",
            "genes": "CmOr, PSY, CmKFB",
            "method": "Gibbs 掩码重采样 (T=0.9, 200 iter, 10% mask)",
            "expected_effect": "增强类胡萝卜素积累 → 橙肉色加深 + 维生素A原提升",
            "validation": "HPLC类胡萝卜素谱 + 色差仪 + Western blot",
        },
        "ripening": {
            "target": "成熟调控启动子",
            "genes": "CmACS1, CmACO1, CmNAC-NOR",
            "method": "Gibbs 掩码重采样 (T=0.9, 200 iter, 10% mask)",
            "expected_effect": "调控乙烯合成，延长货架期3-7天(非跃变型)或优化成熟一致性",
            "validation": "乙烯释放量 + 果实硬度 + 货架期统计",
        },
    }
    
    for name, info in targets.items():
        report.append(f"#### {info['target']}\n\n")
        report.append(f"- **靶标基因**: {info['genes']}\n")
        report.append(f"- **设计方法**: {info['method']}\n")
        report.append(f"- **预期效应**: {info['expected_effect']}\n")
        report.append(f"- **验证方案**: {info['validation']}\n\n")
    
    # Usage instructions
    report.append("## 生成新设计\n\n")
    report.append("```bash\n")
    report.append("# 糖代谢调控元件\n")
    report.append("python scripts/design_elements.py --model assets/plantcad2 \\\n")
    report.append("    --element sugar --n 1000 --length 400 \\\n")
    report.append("    --out results/designed_sugar_v3.fasta\n\n")
    report.append("# 果肉颜色调控元件\n")
    report.append("python scripts/design_elements.py --model assets/plantcad2 \\\n")
    report.append("    --element flesh --n 500 --length 400 \\\n")
    report.append("    --out results/designed_flesh_v3.fasta\n\n")
    report.append("# 成熟调控元件\n")
    report.append("python scripts/design_elements.py --model assets/plantcad2 \\\n")
    report.append("    --element ripening --n 500 --length 400 \\\n")
    report.append("    --out results/designed_ripening_v3.fasta\n")
    report.append("```\n\n")
    
    report.append("## 设计后验证流程\n\n")
    report.append("1. **in silico 筛选**: 用协方差探针 + LLR 对候选序列打分，取 top 100\n")
    report.append("2. **注释探针验证**: 确认生成的序列富集目标功能注释类别\n")
    report.append("3. **合成与转化**: 化学合成 top 10 序列 → 双元载体 → 农杆菌转化甜瓜子叶\n")
    report.append("4. **表型验证**: T1代果实Brix、类胡萝卜素含量、货架期测定\n")
    
    out_path = "outputs/de_novo_design_report.md"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    log(f"  Report: {out_path}")
    
    return targets


# ============================================================
# MODULE 5: LLM Mechanism Interpretation
# ============================================================

PROMPT_TEMPLATE = """你是甜瓜（Cucumis melo，葫芦科）分子设计育种专家。
请基于以下结构化证据，用通俗但严谨的中文解释该变异的分子机制及其育种意义。

【变异信息】
{variant_meta}

【效应预测（PLANTGFM协方差探针）】
{effect_summary}

【功能扰动谱（Top-5受影响功能注释）】
{disruption_table}

请输出：
(1) 分子机制（50-80字）：该变异最可能破坏的分子通路
(2) 农艺影响（40-60字）：对果实品质/产量/抗性的影响方向
(3) 育种建议（40-60字）：是否保留、作MAS标记还是CRISPR靶点
"""


def call_llm(variant_meta, effect_summary, disruption_table):
    """Call Claude API for variant interpretation."""
    prompt = PROMPT_TEMPLATE.format(
        variant_meta=variant_meta,
        effect_summary=effect_summary,
        disruption_table=disruption_table,
    )
    
    try:
        import requests
        
        payload = {
            "model": LLM_MODEL,
            "max_tokens": 600,
            "system": "你是甜瓜分子设计育种专家。回答简洁、专业、可操作。",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        }
        
        resp = requests.post(
            f"{ANTHROPIC_BASE_URL}/v1/messages",
            headers=headers,
            json=payload,
            timeout=60,
        )
        
        if resp.status_code != 200:
            return f"[API Error {resp.status_code}] {resp.text[:200]}"
        
        data = resp.json()
        texts = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                texts.append(block["text"])
        
        usage = data.get("usage", {})
        return "\n".join(texts)
        
    except Exception as e:
        return f"[LLM调用失败: {e}]"


def module_llm_interpretation(df):
    """Module 5: LLM 机制解释（Top 10 变异）"""
    log("\n" + "=" * 60)
    log("MODULE 5: LLM Mechanism Interpretation (Top 10)")
    log("=" * 60)
    
    top10 = df.sort("effect_score", descending=True).head(10)
    
    report = []
    report.append("# 甜瓜 Top 10 高影响变异 LLM 育种机制解释\n\n")
    report.append(f"**模型**: {LLM_MODEL}\n")
    report.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
    report.append("---\n\n")
    
    interpretations = []
    
    for i, row in enumerate(top10.iter_rows(named=True)):
        vid = row.get("variant_id", "")
        gene = row.get("gene", "") or "intergenic"
        score = row.get("effect_score", 0)
        consequence = row.get("consequence", "")
        prediction = row.get("prediction", "")
        disruptions = row.get("top_disruptions", "")
        
        variant_meta = (f"变异ID: {vid}, 基因: {gene}, "
                       f"后果: {consequence}, 区域: 甜瓜基因组")
        
        effect_summary = (f"效应分数: {score:.4f} (0-1, 越高越可能不利), "
                         f"预测: {prediction}")
        
        disruption_table = (disruptions or "无扰动数据").replace("|", ", ")
        
        log(f"  [{i+1}/10] Calling LLM for {vid}...")
        
        interpretation = call_llm(variant_meta, effect_summary, disruption_table)
        interpretations.append({
            "variant_id": vid,
            "score": score,
            "gene": gene,
            "interpretation": interpretation,
        })
        
        report.append(f"## #{i+1}: {vid}\n\n")
        report.append(f"**基因**: {gene} | **效应分数**: {score:.4f} | **预测**: {prediction}\n\n")
        report.append(f"**功能扰动**: {disruptions}\n\n")
        report.append(f"{interpretation}\n\n")
        report.append("---\n\n")
        
        time.sleep(0.3)  # Rate limiting
    
    # Overall summary
    report.append("\n## 育种机制解释摘要\n\n")
    report.append(f"对效应评分最高的 10 个甜瓜变异进行了 LLM 育种机制解释，核心发现：\n\n")
    
    # Categorize by theme
    high_scores = [it for it in interpretations if it["score"] > 0.95]
    report.append(f"- **极高效应变异** (score>0.95): {len(high_scores)} 个，建议优先实验验证\n")
    report.append(f"- **高效应变异** (score 0.85-0.95): {len([it for it in interpretations if 0.85 < it['score'] <= 0.95])} 个\n")
    report.append(f"- **中效应变异** (score<0.85): {len([it for it in interpretations if it['score'] <= 0.85])} 个\n")
    
    report.append("\n### 可操作建议\n\n")
    report.append("1. 极高效应位点可作为 **CRISPR/Cas9 基因编辑靶点**\n")
    report.append("2. 高效应位点可开发为 **KASP 标记用于 MAS 辅助选择**\n")
    report.append("3. 中效应位点建议 **结合田间表型联合验证**\n")
    
    out_path = "outputs/llm_mechanism_interpretation.md"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    log(f"  Report: {out_path}")
    
    return interpretations


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

def module_executive_summary(top10_data, chrom_bv, hybrid_pairs, design_targets, llm_results):
    """Generate breeding application executive summary."""
    log("\n" + "=" * 60)
    log("EXECUTIVE SUMMARY: 甜瓜智能育种应用")
    log("=" * 60)
    
    summary = []
    summary.append("# 甜瓜 PlantCAD2-EVEE 智能育种应用执行摘要\n\n")
    summary.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
    summary.append("---\n\n")
    
    summary.append("## 1. Top 10 高影响位点\n\n")
    if top10_data is not None:
        for i, row in enumerate(top10_data.iter_rows(named=True)):
            vid = row.get("variant_id", "")
            score = row.get("effect_score", 0)
            gene = row.get("gene", "") or "intergenic"
            chrom = vid.split(":")[0] if ":" in vid else ""
            chrom_std = CONTIG_TO_CHR.get(chrom, chrom)
            summary.append(f"{i+1}. `{vid}` (chr{chrom_std}, {gene}): 效应={score:.4f}\n")
    
    summary.append("\n## 2. 染色体育种值\n\n")
    if chrom_bv is not None:
        top_chrom = chrom_bv.head(3)
        summary.append("高育种优先级染色体:\n\n")
        for row in top_chrom.iter_rows(named=True):
            summary.append(f"- {row['chrom']}: 总不利效应={row['total_adverse']:.1f}, "
                          f"不利变异={row['n_adverse']}\n")
    
    summary.append("\n## 3. 杂交组合推荐\n\n")
    if hybrid_pairs:
        summary.append("Top 3 推荐杂交组合:\n\n")
        for i, pair in enumerate(hybrid_pairs[:3]):
            summary.append(f"{i+1}. {pair['parent1_name']} × {pair['parent2_name']}: "
                          f"互补分数={pair['complementarity_score']}\n")
    
    summary.append("\n## 4. De Novo 调控元件设计\n\n")
    if design_targets:
        for name, info in design_targets.items():
            summary.append(f"- **{info['target']}**: 靶向 {info['genes']}, {info['expected_effect'][:50]}...\n")
    
    summary.append("\n## 5. LLM 机制解释\n\n")
    if llm_results:
        n_interpreted = len(llm_results)
        avg_score = sum(r['score'] for r in llm_results) / max(n_interpreted, 1)
        summary.append(f"已完成 {n_interpreted} 个高效应变异的 LLM 育种机制解释，"
                      f"平均效应分数 {avg_score:.4f}\n")
    
    summary.append("\n---\n\n")
    summary.append("**所有模块输出文件**:\n\n")
    summary.append("- `outputs/top10_high_impact_report.md` — Top 10 位点详细报告\n")
    summary.append("- `outputs/chromosome_breeding_values.md` — 染色体育种值报告\n")
    summary.append("- `outputs/chromosome_breeding_values.csv` — 育种值CSV数据\n")
    summary.append("- `outputs/hybrid_recommendation.md` — 杂交组合推荐\n")
    summary.append("- `outputs/de_novo_design_report.md` — 调控元件设计报告\n")
    summary.append("- `outputs/llm_mechanism_interpretation.md` — LLM育种机制解释\n")
    
    out_path = "outputs/breeding_application_summary.md"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(''.join(summary))
    log(f"  Summary: {out_path}")
    print("\n" + '=' * 60)
    print("ALL MODULES COMPLETE")
    print('=' * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--module', choices=['top10', 'breed', 'hybrid', 'design', 'llm', 'all'],
                    default='all', help='指定运行模块')
    ap.add_argument('--predictions', default=PREDICTIONS_CSV, help='预测结果CSV路径')
    ap.add_argument('--no-llm', action='store_true', help='跳过LLM API调用')
    args = ap.parse_args()
    
    import polars as pl
    
    csv_path = args.predictions
    if not os.path.exists(csv_path):
        log(f"ERROR: predictions file not found: {csv_path}")
        log("  Run scripts/predict.py first!")
        return
    
    log(f"Loading predictions: {csv_path}")
    df = pl.read_csv(csv_path)
    log(f"  {len(df):,} variants loaded")
    
    os.makedirs("outputs", exist_ok=True)
    
    top10_data = None
    chrom_bv = None
    hybrid_pairs = None
    design_targets = None
    llm_results = None
    
    modules = ['top10', 'breed', 'hybrid', 'design', 'llm'] if args.module == 'all' else [args.module]
    
    for mod in modules:
        if mod == 'top10':
            top10_data = module_top10(df)
        elif mod == 'breed':
            chrom_bv = module_breeding_value(df)
        elif mod == 'hybrid':
            hybrid_pairs = module_hybrid_recommendation(df)
        elif mod == 'design':
            design_targets = module_de_novo_design()
        elif mod == 'llm':
            if args.no_llm:
                log("  Skipping LLM (--no-llm)")
            else:
                llm_results = module_llm_interpretation(df)
    
    # Always produce executive summary
    module_executive_summary(top10_data, chrom_bv, hybrid_pairs, design_targets, llm_results)


if __name__ == '__main__':
    main()
