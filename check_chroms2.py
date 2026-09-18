import csv, gzip

# Check DR chroms
with open('assets/phenotypes/Zhao2019_DifferentiatedRegions.csv') as f:
    reader = csv.DictReader(f)
    next(reader)  # skip title row
    chroms = set()
    for row in reader:
        chroms.add(row['chromosome'])
print('DR chroms:', sorted(chroms))

# Check Sweep chroms
with open('assets/phenotypes/Zhao2019_SweepRegions.csv') as f:
    reader = csv.DictReader(f)
    next(reader)
    chroms = set()
    sources = set()
    for row in reader:
        chroms.add(row['chromosome'])
        sources.add(row['source'])
print('Sweep chroms:', sorted(chroms))
print('Sweep sources:', sorted(sources))

# Check variant chroms
import pyarrow.parquet as pq
pf = pq.ParquetFile('data/variants.parquet')
table = pf.read()
vids = set()
for i in range(len(table.column('variant_id'))):
    v = table.column('variant_id')[i].as_py()
    if v:
        vids.add(v.split(':')[0])
print('VCF chroms:', sorted(vids))
