#!/usr/bin/env python3
import polars as pl
from collections import Counter
import os

PROJ = "/data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee"
os.chdir(PROJ)

df = pl.read_csv("results/predictions_with_annotation.csv")
total = len(df)

print("="*60)
print("[1] 功能类别统计")
print("="*60)
all_ann = []
for row in df["top_disruptions"].to_list():
    if row:
        all_ann.extend([a.strip() for a in row.split("|")])

cat_counts = Counter(all_ann)
print("\n{:<25} {:>8} {:>8}".format("功能类别","数量","占比"))
print("-"*45)
for cat, cnt in cat_counts.most_common():
    print("{:<25} {:>8} {:>7.1f}%".format(cat, cnt, cnt/total*100))
print("\n")

print("="*60)
print("[2] 高影响变异排序 Top 20")
print("="*60)
sorted_df = df.sort("effect_score", descending=True)
top20 = sorted_df.head(20).with_columns([
    pl.col("variant_id").str.split(":").list.get(0).alias("chrom"),
    pl.col("variant_id").str.split(":").list.get(1).alias("pos"),
])
print("\n{:<5} {:<28} {:<8} {:<10} {:<10} {:<8} {}".format(
    "排名","变异ID","染色体","位置","效应分","预测","功能扰动"))
print("-"*110)
for i, row in enumerate(top20.iter_rows(named=True)):
    d = (row["top_disruptions"][:35] if row["top_disruptions"] else "-")
    print("{:<5} {:<28} {:<8} {:<10} {:<10.4f} {:<8} {}".format(
        i+1, row["variant_id"], row["chrom"], row["pos"],
        row["effect_score"], row["prediction"], d))

sorted_df.head(100).select(["variant_id","effect_score","prediction","top_disruptions"]).write_csv("results/top100_high_impact.csv")

print("\n")
print("="*60)
print("[3] 染色体分布统计")
print("="*60)
df2 = df.with_columns([
    pl.col("variant_id").str.split(":").list.get(0).alias("chrom"),
    pl.col("effect_score").cast(pl.Float64),
])
chrom_stats = df2.group_by("chrom").agg([
    pl.count().alias("total"),
    pl.col("effect_score").mean().alias("avg_effect"),
    pl.col("prediction").str.contains("不利").sum().alias("bad"),
]).sort("chrom")

print("\n{:<10} {:>8} {:>8} {:>8} {:>8} {:>10}".format(
    "染色体","总数","不利","有利","不利%","平均效应"))
print("-"*55)
for row in chrom_stats.iter_rows(named=True):
    good = row["total"] - row["bad"]
    bad_pct = row["bad"]/row["total"]*100
    print("{:<10} {:>8} {:>8} {:>8} {:>7.1f}% {:>10.4f}".format(
        row["chrom"], row["total"], row["bad"], good, bad_pct, row["avg_effect"]))

chrom_stats.write_csv("results/chrom_distribution.csv")

print("\n"+ "="*60)
print("[额外] 效应分分布区间")
print("="*60)
bins = [0, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
labels = ["0-0.001","0.001-0.01","0.01-0.1","0.1-0.3","0.3-0.5","0.5-0.7","0.7-0.9","0.9-1.0"]
df2 = df2.with_columns(
    pl.cut(pl.col("effect_score"), breaks=bins, labels=labels).alias("score_bin")
)
hist = df2["score_bin"].value_counts().sort("score_bin")
print("\n{:<15} {:>8} {:>8}".format("效应区间","数量","占比"))
print("-"*33)
for row in hist.iter_rows(named=True):
    print("{:<15} {:>8} {:>7.1f}%".format(row["score_bin"], row["count"], row["count"]/total*100))

print("\n Done!")
