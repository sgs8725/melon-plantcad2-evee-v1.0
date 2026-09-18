import pyarrow.parquet as pq, os
os.chdir("/data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee")

LOC_CHROM = "contig8"
LOC_POS = 20550439
WINDOW = 50000

for fname, label_type in [
    ("data/variants_labeled_zhao_clean.parquet", "Zhao 2019"),
    ("data/variants_labeled_unified.parquet", "Unified"),
]:
    t = pq.read_table(fname)
    nearby = []
    for i in range(len(t)):
        vid = t.column("variant_id")[i].as_py()
        if not vid: continue
        parts = vid.split(":")
        if parts[0] != LOC_CHROM: continue
        pos = int(parts[1])
        if abs(pos - LOC_POS) <= WINDOW:
            label = t.column("label")[i].as_py()
            nearby.append((pos, label, vid))
    
    print(f"=== {fname} ({label_type}) ===")
    print(f"  Variants within {WINDOW}bp: {len(nearby)}")
    if nearby:
        pos_l = sum(1 for _,l,_ in nearby if l == 1)
        neg_l = sum(1 for _,l,_ in nearby if l == 0)
        print(f"  Labeled 1: {pos_l}")
        print(f"  Labeled 0: {neg_l}")
        print(f"  Sample: {nearby[:5]}")
    print()

# Also check original predictions
import csv
print("=== Checking predictions_with_annotation.csv ===")
with open("results/predictions_with_annotation.csv") as f:
    reader = csv.DictReader(f)
    nearby_pred = []
    for row in reader:
        vid = row.get("variant_id", "")
        if not vid: continue
        parts = vid.split(":")
        if parts[0] != LOC_CHROM: continue
        pos = int(parts[1])
        if abs(pos - LOC_POS) <= WINDOW:
            effect = row.get("effect_score", row.get("effect", "N/A"))
            nearby_pred.append((pos, effect, vid))
    print(f"  Predictions near LOC01: {len(nearby_pred)}")
    for p, e, v in sorted(nearby_pred)[:10]:
        print(f"    {v}  effect={e}")
