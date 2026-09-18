#!/usr/bin/env python3
"""
prepare_zhao_labels.py
将 Zhao 2019 多来源区间合并为统一的变异标签:
  Tier 1: 分化区间 (DR, 289个, Fst top 10%)
  Tier 2: 驯化选择区间 (WM 154 + WA 191)
  Tier 3: QTL重叠区间 (Supplementary 10 + 14)

染色体命名映射: chr1-chr12 → contig1-contig13 (VCF命名)
"""

import csv, os, sys
from pathlib import Path

# Go up one level from scripts/ to project root
PROJ = Path(__file__).resolve().parent.parent
os.chdir(str(PROJ))
print(f"Working dir: {os.getcwd()}")

# ========== 染色体命名映射 (VCF contig命名) ==========
CHR_MAP = {
    'chr1': 'contig7', 'chr2': 'contig2', 'chr3': 'contig9',
    'chr4': 'contig5', 'chr5': 'contig12', 'chr6': 'contig4',
    'chr7': 'contig6', 'chr8': 'contig3', 'chr9': 'contig8',
    'chr10': 'contig13', 'chr11': 'contig11', 'chr12': 'contig10',
}

def map_chr(c):
    c = c.strip().lower().replace('chromosome', 'chr').replace('chromsome', 'chr')
    if not c.startswith('chr'):
        c = 'chr' + c
    return CHR_MAP.get(c, c)

# ========== 1. 加载 Zhao 2019 分化区间 (DR) ==========
def load_intervals(path, source_name):
    intervals = []
    with open(path) as f:
        reader = csv.DictReader(f)
        next(reader)  # skip title
        for row in reader:
            try:
                chrom = map_chr(row['chromosome'])
                start = int(row['start'])
                end = int(row['end'])
                if start < end:
                    intervals.append((chrom, start, end, source_name))
            except:
                continue
    return intervals

# ========== 2. 加载 Zhao 2019 选择区间 (WM/WA) ==========
def load_sweeps(path):
    intervals = []
    with open(path) as f:
        reader = csv.DictReader(f)
        next(reader)
        for row in reader:
            try:
                chrom = map_chr(row['chromosome'])
                start = int(row['start'])
                end = int(row['end'])
                sid = row.get('region_id', '')
                if start < end and chrom and sid:
                    src = 'melo_sweep' if sid.startswith('WM') else 'agrestis_sweep'
                    intervals.append((chrom, start, end, src))
            except:
                continue
    return intervals

# ========== 3. QTL 区间 ==========
# Supplementary 10: QTLs overlapping with domestication sweeps
QTL10 = [
    ('contig2', 1200803, 1962709, 'qtl_flqmz2.1'),
    ('contig9', 22800812, 22804774, 'qtl_CmHMGR'),
    ('contig9', 24754248, 25639776, 'qtl_flqaz3.1'),
    ('contig9', 24932671, 25728751, 'qtl_fwqm3.1'),
    ('contig5', 15252998, 26456049, 'qtl_ftqaz4.1'),
    ('contig12', 23403959, 24186650, 'qtl_ftqmz5.1'),
    ('contig12', 23403959, 24186650, 'qtl_flqmz5.1'),
    ('contig12', 23403959, 23937721, 'qtl_fwqm5.1'),
    ('contig4', 31866468, 33580170, 'qtl_fdqmz6.1'),
    ('contig4', 32243557, 33580170, 'qtl_ftqmz6.1'),
    ('contig4', 32836635, 33406981, 'qtl_fwqm6.1'),
    ('contig6', 11795792, 22515291, 'qtl_fwqaz7.1'),
    ('contig3', 2949482, 4768320, 'qtl_fdqaz8.1'),
    ('contig3', 2949482, 4768320, 'qtl_ftqaz8.1'),
    ('contig3', 3609238, 4128008, 'qtl_fwqaz8.1'),
    ('contig3', 25385932, 26041972, 'qtl_CmPH'),
    ('contig8', 9486455, 18823423, 'qtl_scdqmz9.1'),
    ('contig8', 19120963, 20737938, 'qtl_flqaz9.1'),
    ('contig8', 21787268, 21787681, 'qtl_CmBt'),
    ('contig11', 20703791, 26478808, 'qtl_fdqaz11.1'),
    ('contig11', 20925335, 26571074, 'qtl_ftqaz11.1'),
    ('contig11', 27871190, 29074792, 'qtl_sscqmz11.1'),
    ('contig11', 30234349, 30239671, 'qtl_CmBi'),
]

# Supplementary 14: QTLs overlapping with divergent regions
QTL14 = [
    ('contig5', 10589815, 20498359, 'qtl_FCONVQF4.3'),
    ('contig5', 1244873, 10589815, 'qtl_FFPQF4.1'),
    ('contig12', 19871767, 28181106, 'qtl_FFPQF5.3'),
    ('contig6', 21903935, 25409863, 'qtl_FFPQF7.2'),
    ('contig3', 1149022, 4743709, 'qtl_FFPQF8.1'),
    ('contig3', 26427230, 31855757, 'qtl_FFPQF8.4'),
    ('contig3', 26427230, 29681431, 'qtl_SSCQA8.1'),
    ('contig9', 15078876, 24683733, 'qtl_SSCQA3.1'),
    ('contig12', 1508958, 2409548, 'qtl_SSCQA5.1'),
    ('contig9', 22582712, 27248392, 'qtl_SSCQC3.5'),
    ('contig6', 21903935, 25409863, 'qtl_SSCQC7.2'),
    ('contig8', 22070369, 23380898, 'qtl_SSCQC9.3'),
    ('contig12', 19871767, 28181106, 'qtl_TSUGQH5.3'),
    ('contig3', 4743709, 8994256, 'qtl_TSUGQH8.1'),
    ('contig3', 26427230, 31855757, 'qtl_TSUGQH8.4'),
    ('contig8', 22070369, 23380898, 'qtl_TSUGQH9.3'),
    ('contig11', 29526751, 31127208, 'qtl_EayQL11.1'),
]

# ========== 主流程 ==========
def main():
    print("="*60)
    print("Prepare Zhao 2019 Interval Labels")
    print("="*60)

    # 加载所有区间
    all_intervals = []
    
    # Tier 1: 分化区间
    dr = load_intervals('assets/phenotypes/Zhao2019_DifferentiatedRegions.csv', 'DR')
    all_intervals.extend(dr)
    print(f"\n[Tier 1] Differentiated Regions (DR): {len(dr)}")
    
    # Tier 2: 选择区间
    sw = load_sweeps('assets/phenotypes/Zhao2019_SweepRegions.csv')
    all_intervals.extend(sw)
    print(f"[Tier 2] Domestication Sweeps: {len(sw)}")
    
    # Tier 3: QTL
    all_intervals.extend(QTL10)
    all_intervals.extend(QTL14)
    print(f"[Tier 3] QTL intervals: {len(QTL10)} + {len(QTL14)} = {len(QTL10)+len(QTL14)}")
    
    print(f"\nTotal intervals: {len(all_intervals)}")
    
    # 去重: 相同(chrom,start,end)的只保留一个
    unique = list(set((c,s,e) for c,s,e,_ in all_intervals))
    unique.sort()
    print(f"Unique intervals: {len(unique)}")
    
    # 统计每染色体
    chrom_counts = {}
    for c,s,e,_ in all_intervals:
        chrom_counts[c] = chrom_counts.get(c, 0) + 1
    print("\nPer-chromosome interval counts:")
    for c in sorted(chrom_counts):
        print(f"  {c}: {chrom_counts[c]}")
    
    # 加载变异数据集
    print("\n[Loading] variants.parquet ...")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile('data/variants.parquet')
    table = pf.read()
    
    # 构建区间索引: {chrom: [(start, end), ...]}
    from collections import defaultdict
    idx = defaultdict(list)
    for c, s, e, _ in all_intervals:
        idx[c].append((s, e))
    
    # 对每条变异打标签
    print(f"\n[Labeling] {len(table)} variants ...")
    labels = []
    n_hit = 0
    for i in range(len(table)):
        vid = table.column('variant_id')[i].as_py()
        if not vid:
            labels.append(0)
            continue
        parts = vid.split(':')
        chrom = parts[0]
        try:
            pos = int(parts[1])
        except:
            labels.append(0)
            continue
        
        hit = 0
        if chrom in idx:
            for s, e in idx[chrom]:
                if s <= pos <= e:
                    hit = 1
                    n_hit += 1
                    break
        labels.append(hit)
        
        if (i+1) % 10000 == 0:
            print(f"  [{i+1}/{len(table)}] hits so far: {n_hit}")
    
    print(f"\nTotal positive (hit interval): {n_hit} / {len(table)} ({100*n_hit/len(table):.1f}%)")
    
    # 添加label列
    import pyarrow as pa
    label_col = pa.array(labels, type=pa.int32())
    table = table.append_column('label', label_col)
    
    # 写回
    out_path = 'data/variants_labeled_zhao.parquet'
    pq.write_table(table, out_path)
    print(f"\n[Saved] {out_path}")
    
    # 统计
    pos = sum(labels)
    neg = len(labels) - pos
    print(f"  Positive (labeled): {pos}")
    print(f"  Negative (unlabeled): {neg}")
    print(f"  Ratio: {pos/neg:.4f}" if neg > 0 else "  All positive")
    print("\nDone!")

if __name__ == '__main__':
    main()
