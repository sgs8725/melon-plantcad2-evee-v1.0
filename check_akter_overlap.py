import csv, os
os.chdir("/data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee")

# Read Akter 2023 GWAS SNPs
akter_headers = None
akter_rows = []
with open("assets/phenotypes/Akter_2023_TableS6_significant_SNPs.csv") as f:
    reader = csv.DictReader(f)
    akter_headers = reader.fieldnames
    for row in reader:
        akter_rows.append(dict(row))
print(f"Akter columns: {akter_headers}")
print(f"Akter SNPs: {len(akter_rows)}")
for r in akter_rows[:5]:
    print(f"  {r}")

# Read Liu 2020 GWAS SNPs  
with open("assets/phenotypes/Liu2020_GWAS_SNPs.csv") as f:
    liu_rows = list(csv.DictReader(f))
liu_set = set((r["chromosome"], r["snp_position"]) for r in liu_rows if r.get("chromosome"))

# Also read existing unified intervals/overlap check
# Check overlap between Akter SNPs and existing labels
overlap_count = 0
for r in akter_rows:
    ch = r.get("Chromosome", r.get("chromosome", ""))
    pos = r.get("Position", r.get("position", r.get("pos", "")))
    for lr in liu_rows:
        if lr["chromosome"] == ch and abs(int(lr["snp_position"]) - int(pos)) <= 20000:
            overlap_count += 1
            break

print(f"\nAkter SNPs overlapping with Liu 2020 (20Kb): {overlap_count}/{len(akter_rows)}")
print(f"Unique Akter SNPs: {len(akter_rows) - overlap_count}")
