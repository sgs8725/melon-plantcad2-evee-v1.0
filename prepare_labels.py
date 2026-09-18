#!/usr/bin/env python3
import polars as pl
PROJ = '/data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee'
CM = {'chr00':'contig1','chr01':'contig7','chr02':'contig2','chr03':'contig9','chr04':'contig5',
      'chr05':'contig12','chr06':'contig4','chr07':'contig6','chr08':'contig3',
      'chr09':'contig8','chr10':'contig13','chr11':'contig11','chr12':'contig10'}

l = pl.read_csv(f'{PROJ}/data/probe_labels/unified_probe_training_labels.csv',
    schema_overrides={c:pl.Utf8 for c in ['chromosome','position','gene','start','end','trait','p_value','effect_size','source','label_type']})
l = l.filter(pl.col('position').str.contains(r'^[0-9]+$'))
def nc(c):
    c=c.strip().lower().replace('chromsome','').replace('chr','')
    return f'chr{int(c):02d}' if c.isdigit() and 1<=int(c)<=12 else None
l = l.with_columns(pl.col('chromosome').map_elements(nc, return_dtype=pl.Utf8).alias('cn'))
l = l.filter(pl.col('cn').is_not_null())
l = l.with_columns(pl.col('position').cast(pl.Int64))
l = l.with_columns([
    (pl.col('position')-20000).alias('rs'),
    (pl.col('position')+20000).alias('re')])
l = l.with_columns(pl.col('cn').replace_strict(CM).alias('chrom'))
print(f'Labels: {len(l)}')

v = pl.read_parquet(f'{PROJ}/data/variants.parquet')
v = v.with_columns([
    pl.col('variant_id').str.split(':').list.get(0).alias('chrom'),
    pl.col('variant_id').str.split(':').list.get(1).cast(pl.Int64).alias('pos')])
print(f'Variants: {len(v)}')

m = v.join_where(l.select(['chrom','rs','re','trait']),
    (pl.col('chrom')==pl.col('chrom_right'))&(pl.col('pos')>=pl.col('rs'))&(pl.col('pos')<=pl.col('re')))
m = m.select(['variant_id','trait']).unique()
m = m.group_by('variant_id').agg([pl.col('trait').str.join(','), pl.lit(1).alias('label')])
print(f'Matched: {len(m)} labels on {len(v)} variants')

r = v.with_columns(pl.lit(-1).alias('label'), pl.lit('').alias('trait_label'))
r = r.join(m, on='variant_id', how='left', suffix='_g')
r = r.with_columns([pl.col('label_g').fill_null(-1).alias('label'), pl.col('trait').fill_null('').alias('trait_label')])
n = r.filter(pl.col('label')==1).shape[0]
print(f'Positive: {n}')
r.write_parquet(f'{PROJ}/data/variants_labeled.parquet')
print('Done')
