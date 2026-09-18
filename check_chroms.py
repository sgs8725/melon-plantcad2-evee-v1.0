import pandas as pd
d = pd.read_csv('assets/phenotypes/Zhao2019_DifferentiatedRegions.csv', skiprows=1)
print('DR chroms:', sorted(d['chromosome'].unique()))
s = pd.read_csv('assets/phenotypes/Zhao2019_SweepRegions.csv', skiprows=1)
print('Sweep chroms:', sorted(s['chromosome'].unique()))
print('Sweep sources:', sorted(s['source'].unique()))
v = pd.read_parquet('data/variants.parquet')
vids = v['variant_id'].str.split(':').str[0].unique()
print('VCF chroms:', sorted(vids))
