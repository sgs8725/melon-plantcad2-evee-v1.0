#!/usr/bin/env python3
"""统一增强标签：Zhao 2019区间 + Liu 2020 GWAS SNP + Liu 2020区间"""
import csv, os, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))

# 染色体映射: chr编号→VCF contig名
CHR_MAP = {'1':'contig7','2':'contig2','3':'contig9','4':'contig5','5':'contig12',
           '6':'contig4','7':'contig6','8':'contig3','9':'contig8','10':'contig13',
           '11':'contig11','12':'contig10'}
# 也支持"chr1"格式
for k in list(CHR_MAP.keys()):
    CHR_MAP['chr'+k] = CHR_MAP[k]

def map_chr(c):
    c = str(c).strip().lower().replace('chromosome','chr').replace('chromsome','chr')
    return CHR_MAP.get(c, c)

# ====== 1. Zhao 2019 区间标签（已有） ======
intervals = defaultdict(list)
with open('assets/phenotypes/Zhao2019_DifferentiatedRegions.csv') as f:
    r = csv.DictReader(f); next(r)
    for row in r:
        try:
            c,s,e = map_chr(row['chromosome']), int(row['start']), int(row['end'])
            if s<e: intervals[c].append((s,e,'zhao_DR'))
        except: pass

with open('assets/phenotypes/Zhao2019_SweepRegions.csv') as f:
    r = csv.DictReader(f); next(r)
    for row in r:
        try:
            c,s,e = map_chr(row['chromosome']), int(row['start']), int(row['end'])
            if s<e and c and row.get('region_id','').startswith('WM'): intervals[c].append((s,e,'zhao_WM'))
            elif s<e and c: intervals[c].append((s,e,'zhao_WA'))
        except: pass

# Zhao QTL intervals (from Supplementary 10+14)
zhao_qtls = [
    ('contig2',1200803,1962709),('contig9',22800812,22804774),('contig9',24754248,25639776),
    ('contig9',24932671,25728751),('contig5',15252998,26456049),('contig12',23403959,24186650),
    ('contig12',23403959,23937721),('contig4',31866468,33580170),('contig4',32243557,33580170),
    ('contig4',32836635,33406981),('contig6',11795792,22515291),('contig3',2949482,4768320),
    ('contig3',3609238,4128008),('contig3',25385932,26041972),('contig8',9486455,18823423),
    ('contig8',19120963,20737938),('contig8',21787268,21787681),('contig11',20703791,26478808),
    ('contig11',20925335,26571074),('contig11',27871190,29074792),('contig11',30234349,30239671),
    ('contig5',10589815,20498359),('contig5',1244873,10589815),('contig12',19871767,28181106),
    ('contig6',21903935,25409863),('contig3',1149022,4743709),('contig3',26427230,31855757),
    ('contig3',26427230,29681431),('contig9',15078876,24683733),('contig12',1508958,2409548),
    ('contig9',22582712,27248392),('contig8',22070369,23380898),('contig3',4743709,8994256),
    ('contig11',29526751,31127208),
]
for c,s,e in zhao_qtls:
    intervals[c].append((s,e,'zhao_qtl'))

print(f'[Zhao] 区间标签: {sum(len(v) for v in intervals.values())}')

# ====== 2. Liu 2020 GWAS SNPs ======
gwas_list = []
with open('assets/phenotypes/Liu2020_GWAS_SNPs.csv') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            c = map_chr(row['chromosome'])
            pos = int(row['snp_position'])
            logp = float(row['neg_log10_p'])
            trait = row['trait']
            gwas_list.append((c, pos, logp, trait))
        except: pass
print(f'[Liu GWAS] {len(gwas_list)} SNPs')

# ====== 3. Liu 2020 区间标签 ======
for fname, src in [('Sweeps_Domestication','liu_dom'),('Sweeps_Improvement','liu_imp'),
                   ('FST_Domestication','liu_fstd'),('FST_Improvement','liu_fsti')]:
    path = f'assets/phenotypes/Liu2020_{fname}.csv'
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                c = map_chr(row['chromosome'])
                s,e = int(row['start']), int(row['end'])
                if s<e: intervals[c].append((s,e,src))
            except: pass

total_int = sum(len(v) for v in intervals.values())
print(f'[Liu Intervals] {total_int - len(zhao_qtls) - 477} (总区间: {total_int})')

# ====== 4. Akter 2023 GWAS SNPs (8个, 20Kb窗口) ======
akter_list = []
with open('assets/phenotypes/Akter_2023_TableS6_significant_SNPs.csv') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            c = map_chr(row['chromosome'])
            pos = int(row['position'])
            pval = float(row['p_value'])
            pheno = row['phenotype']
            akter_list.append((c, pos, pval, pheno))
        except: pass
print(f'[Akter GWAS] {len(akter_list)} SNPs ({", ".join(set(a[3] for a in akter_list))})')

# 合并GWAS SNP标签（20Kb窗口）
import pyarrow.parquet as pq, pyarrow as pa
table = pq.read_table('data/variants.parquet')

# 先标区间标签
labels = []
for i in range(len(table)):
    vid = table.column('variant_id')[i].as_py()
    if not vid:
        labels.append(0); continue
    parts = vid.split(':')
    chrom, pos = parts[0], int(parts[1])
    label = 0
    # 检查区间
    if chrom in intervals:
        for s,e,_ in intervals[chrom]:
            if s <= pos <= e:
                label = 1
                break
    # 如果没命中区间，检查GWAS SNP窗口
    if label == 0:
        for c, p, logp, trait in gwas_list:
            if c == chrom and abs(pos - p) <= 20000:
                label = 1
                break
    # 如果没命中区间和Liu GWAS，检查Akter GWAS SNP窗口
    if label == 0:
        for c, p, pval, pheno in akter_list:
            if c == chrom and abs(pos - p) <= 20000:
                label = 1
                break
    labels.append(label)
    if (i+1) % 10000 == 0:
        print(f'  标注 {i+1}/{len(table)}', flush=True)

pos = sum(labels)
print(f'\n[结果] 正={pos}/{len(labels)} ({100*pos/len(labels):.1f}%) 负={len(labels)-pos}')

# 写出
cols = {n: table.column(i) for i,n in enumerate(table.schema.names) if n not in ['label','trait_label','var_index']}
cols['label'] = pa.array(labels, type=pa.int32())
t = pa.table(cols)
pq.write_table(t, 'data/variants_labeled_unified.parquet')
print(f'[保存] data/variants_labeled_unified.parquet')
