import csv, os, sys
from collections import defaultdict

os.chdir("/data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee")

# ========================
# 方案A: 基因坐标验证
# ========================
print("="*60)
print("方案A: 候选基因区域效应分数验证")
print("="*60)

# 已知候选基因及其在DHL92基因组上的位置
# 来源: Ensembl Plants / 文献报道
CANDIDATE_GENES = {
    "LOC01_CmOr": {
        "genes": ["MELO3C005449"],
        "contig": "contig8",  # chr9 → contig8
        "region": (20540000, 20560000),  # ~20.54Mb, CmOr golden SNP
        "trait": "Flesh color (orange)",
        "known_effect": "major (~70% PVE)"
    },
    "LOC03_CmACS7": {
        "genes": ["MELO3C005122"],
        "contig": "contig2",  # chr2 → contig2
        "region": (16340000, 16343000),  # CmACS7 approximate
        "trait": "Fruit length/shape",
        "known_effect": "major QTL"
    },
    "LOC04_SSC": {
        "genes": ["MELO3C014519"],
        "contig": "contig12",  # chr5 → contig12
        "region": (1500000, 2500000),  # SSC QTL region
        "trait": "Soluble solids / sucrose",
        "known_effect": "major QTL"
    },
    "LOC05_CmTST2": {
        "genes": ["MELO3C009291"],
        "contig": "contig10",  # chr12 → contig10
        "region": (15500000, 15700000),  # Tonoplast sugar transporter
        "trait": "Sugar accumulation",
        "known_effect": "confirmed gene"
    },
    "LOC15_CmCLV3": {
        "genes": ["MELO3C016409"],
        "contig": "contig10",  # chr12:15.4Mb
        "region": (15450000, 15460000),  # CmCLV3: S12_15,455,020
        "trait": "Carpel number / fruit shape",
        "known_effect": "pleiotropic"
    },
    "LOC16_CmAAT": {
        "genes": ["MELO3C018167", "MELO3C018168"],
        "contig": "contig11",  # chr11
        "region": (7900000, 8100000),  # CmAAT1/2: S11_7.9-8.0Mb
        "trait": "Aroma",
        "known_effect": "selection signal"
    },
    "LOC17_CmBt": {
        "genes": ["MELO3C005611"],
        "contig": "contig8",  # chr9:21.8Mb
        "region": (21780000, 21790000),  # CmBt
        "trait": "Flesh bitterness",
        "known_effect": "subspecies-divergent"
    },
    "LOC17_CmBi": {
        "genes": ["MELO3C022374"],
        "contig": "contig11",  # chr11:30.2Mb
        "region": (30230000, 30240000),  # CmBi
        "trait": "Flesh bitterness",
        "known_effect": "subspecies-divergent"
    }
}

# Load predictions
predictions = {}
with open("results/predictions_with_annotation.csv") as f:
    for row in csv.DictReader(f):
        vid = row["variant_id"]
        try:
            effect = float(row.get("effect_score", row.get("effect", 0)))
            predictions[vid] = {
                "effect": effect,
                "gene": row.get("gene", ""),
                "consequence": row.get("consequence", ""),
                "chrom": vid.split(":")[0],
                "pos": int(vid.split(":")[1])
            }
        except: pass

for locus_name, info in CANDIDATE_GENES.items():
    contig = info["contig"]
    start, end = info["region"]
    
    # Find variants in this region
    hits = []
    for vid, pred in predictions.items():
        if pred["chrom"] == contig and start <= pred["pos"] <= end:
            hits.append((pred["pos"], pred["effect"], vid, pred.get("gene",""), pred.get("consequence","")))
    
    hits.sort()
    n_hits = len(hits)
    if n_hits > 0:
        mean_eff = sum(h[1] for h in hits) / n_hits
        high = sum(1 for h in hits if h[1] >= 0.5)
        low = sum(1 for h in hits if h[1] == 0)
        print(f"\n{locus_name} ({info['trait']})")
        print(f"  区域: {contig}:{start}-{end}")
        print(f"  变异数: {n_hits}  |  平均效应: {mean_eff:.4f}")
        print(f"  高效应(>=0.5): {high}  |  零效应: {low}")
        for p, e, v, g, c in hits[:5]:
            print(f"    {v:40s}  effect={e:.4f}  {g[:25]:25s} {c}")
        if n_hits > 5:
            print(f"    ... 还有 {n_hits - 5} 个变异")
    else:
        print(f"\n{locus_name} ({info['trait']})")
        print(f"  区域: {contig}:{start}-{end}  →  ❌ 无变异")
        # Try searching broader
        broader_hits = [(p["pos"], p["effect"], vid) for vid, p in predictions.items() 
                       if p["chrom"] == contig and abs(p["pos"] - (start+end)//2) <= 500000]
        if broader_hits:
            mean_b = sum(h[1] for h in broader_hits) / len(broader_hits)
            print(f"    附近500Kb内有 {len(broader_hits)} 个变异, 平均效应={mean_b:.4f}")

# ========================
# 方案B: 染色体富集验证
# ========================
print("\n" + "="*60)
print("方案B: 染色体效应富集验证")
print("="*60)

# Per-chromosome statistics
chrom_data = defaultdict(lambda: {"total": 0, "unfavorable": 0, "sum_effect": 0.0, "zero_effect": 0})
for vid, pred in predictions.items():
    c = pred["chrom"]
    e = pred["effect"]
    chrom_data[c]["total"] += 1
    chrom_data[c]["sum_effect"] += e
    if e >= 0.5:
        chrom_data[c]["unfavorable"] += 1
    if e == 0 or e < 0.001:
        chrom_data[c]["zero_effect"] += 1

# Chromosome mapping
CHR_MAP_REV = {
    "contig1": "chr01", "contig2": "chr02", "contig3": "chr08",
    "contig4": "chr06", "contig5": "chr04", "contig6": "chr07",
    "contig7": "chr01", "contig8": "chr09", "contig9": "chr03",
    "contig10": "chr12", "contig11": "chr11", "contig12": "chr05",
    "contig13": "chr10"
}

print(f"\n{'染色体':>8s} {'标准名':>6s} {'总数':>6s} {'不利':>6s} {'不利%':>7s} {'平均效应':>9s} {'零效应%':>8s}")
print("-"*55)
sorted_chroms = sorted(chrom_data.keys(), key=lambda x: chrom_data[x]["unfavorable"]/max(chrom_data[x]["total"],1), reverse=True)
for c in sorted_chroms:
    d = chrom_data[c]
    pct_unfav = d["unfavorable"] / d["total"] * 100
    pct_zero = d["zero_effect"] / d["total"] * 100
    mean_eff = d["sum_effect"] / d["total"]
    std_name = CHR_MAP_REV.get(c, c)
    print(f"{c:>8s} {std_name:>6s} {d['total']:>6d} {d['unfavorable']:>6d} {pct_unfav:>6.1f}% {mean_eff:>8.4f} {pct_zero:>6.1f}%")

# Summary
print("\n染色体不利变异排名:")
for i, c in enumerate(sorted_chroms, 1):
    d = chrom_data[c]
    pct = d["unfavorable"] / d["total"] * 100
    print(f"  {i:2d}. {c:>8s} ({CHR_MAP_REV.get(c,'?'):>5s}): {pct:.1f}% unfavorable, mean effect={d['sum_effect']/d['total']:.4f}")
