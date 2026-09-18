#!/usr/bin/env python3
"""Regenerate Zhao labels as a clean 2-column label file (variant_id, label)"""
import csv
from collections import defaultdict

# Chromosome mapping
CHR_MAP = {
    'chr1': 'contig7', 'chr2': 'contig2', 'chr3': 'contig9',
    'chr4': 'contig5', 'chr5': 'contig12', 'chr6': 'contig4',
    'chr7': 'contig6', 'chr8': 'contig3', 'chr9': 'contig8',
    'chr10': 'contig13', 'chr11': 'contig11', 'chr12': 'contig10',
}

def map_chr(c):
    c = c.strip().lower().replace('chromosome', 'chr').replace('chromsome', 'chr')
    if not c.startswith('chr'): c = 'chr' + c
    return CHR_MAP.get(c, c)

# Load DR intervals
intervals = defaultdict(list)
with open('assets/phenotypes/Zhao2019_DifferentiatedRegions.csv') as f:
    reader = csv.DictReader(f)
    next(reader)
    for row in reader:
        try:
            c, s, e = map_chr(row['chromosome']), int(row['start']), int(row['end'])
            if s < e: intervals[c].append((s, e, 'DR'))
        except: pass

# Load Sweep intervals
with open('assets/phenotypes/Zhao2019_SweepRegions.csv') as f:
    reader = csv.DictReader(f)
    next(reader)
    for row in reader:
        try:
            c, s, e = map_chr(row['chromosome']), int(row['start']), int(row['end'])
            if s < e and c: intervals[c].append((s, e, 'sweep'))
        except: pass

# QTL10
qtls = [
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
for c,s,e in qtls:
    intervals[c].append((s,e,'qtl'))

# Sort intervals per chromosome
for c in intervals:
    intervals[c].sort()

import pyarrow.parquet as pq
table = pq.read_table('data/variants.parquet')
variants = []
for i in range(len(table)):
    vid = table.column('variant_id')[i].as_py()
    if not vid: continue
    parts = vid.split(':')
    chrom, pos = parts[0], int(parts[1])
    label = 0
    if chrom in intervals:
        for s,e,_ in intervals[chrom]:
            if s <= pos <= e:
                label = 1
                break
    variants.append((vid, label))

# Write as CSV
with open('data/zhao_labels.csv', 'w') as f:
    f.write('variant_id,label\n')
    for vid, lbl in variants:
        f.write(f'{vid},{lbl}\n')

pos = sum(l for _, l in variants)
print(f'Total: {len(variants)}, Positive: {pos} ({100*pos/len(variants):.1f}%)')
print('Saved: data/zhao_labels.csv')
