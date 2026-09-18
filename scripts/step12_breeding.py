import polars as pl

df = pl.read_csv("results/predictions_with_annotation.csv")
df2 = df.with_columns([
    pl.col("variant_id").str.split(":").list.get(0).alias("chrom"),
    pl.col("effect_score").cast(pl.Float64),
])

print("="*60)
print("Step 12: Breeding Value Ranking")
print("="*60)

top50 = df2.sort("effect_score", descending=True).head(50).select(["variant_id","chrom","effect_score","top_disruptions"])
print("\nTop 10 High-Impact Variants:")
for i, row in enumerate(top50.head(10).iter_rows(named=True)):
    ds = (row["top_disruptions"] or "").split("|")[:3]
    info = "{}: {:<30} chr={:<8} score={:.4f}  {}".format(
        i+1, row["variant_id"], row["chrom"], row["effect_score"], ", ".join(ds))
    print("  " + info)

df_score = df2.with_columns(
    pl.when(pl.col("prediction") == "不利").then(pl.col("effect_score")).otherwise(pl.lit(0)).alias("adverse")
)

chrom_bv = df_score.group_by("chrom").agg([
    pl.count().alias("n"),
    pl.col("adverse").sum().alias("total_adv"),
    pl.col("effect_score").mean().alias("mean"),
    pl.col("effect_score").max().alias("max"),
]).sort("total_adv", descending=True)

print("\n\nChromosome Breeding Value Ranking:")
hdr = "{:<10} {:>8} {:>10} {:>8} {:>8}".format("Chrom","N","TotalAdv","Mean","Max")
print(hdr)
print("-"*48)
for row in chrom_bv.iter_rows(named=True):
    line = "{:<10} {:>8} {:>10.2f} {:>8.4f} {:>8.4f}".format(
        row["chrom"], row["n"], row["total_adv"], row["mean"], row["max"])
    print(line)

# Standard chromosome names
cmap = {"contig7":"chr01","contig2":"chr02","contig9":"chr03","contig5":"chr04",
        "contig12":"chr05","contig4":"chr06","contig6":"chr07","contig3":"chr08",
        "contig8":"chr09","contig13":"chr10","contig11":"chr11","contig10":"chr12"}

df_score = df_score.with_columns(pl.col("chrom").replace_strict(cmap).alias("chr_name"))

chr_s = df_score.group_by("chr_name").agg([
    pl.col("effect_score").mean().alias("mean_eff"),
    pl.col("adverse").sum().alias("total_adv"),
    pl.col("prediction").str.contains("不利").sum().alias("n_adverse"),
]).drop_nulls().sort("chr_name")

print("\n\nPer-Chromosome Summary:")
print("{:<6} {:>10} {:>12} {:>10}".format("Chr","N_Adverse","TotalAdv","MeanEff"))
print("-"*40)
for row in chr_s.iter_rows(named=True):
    print("{:<6} {:>10} {:>12.2f} {:>10.4f}".format(
        row["chr_name"], row["n_adverse"], row["total_adv"], row["mean_eff"]))

chrom_bv.write_csv("results/breeding_value_ranking.csv")
print("\n\nSaved: results/breeding_value_ranking.csv")
print("Done!")
