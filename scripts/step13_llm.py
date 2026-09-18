import polars as pl
import sys; sys.path.insert(0, ".")
from probes.llm_synthesis import explain_variant

df = pl.read_csv("results/predictions_with_annotation.csv")
top5 = df.sort("effect_score", descending=True).head(5)

for i, row in enumerate(top5.iter_rows(named=True)):
    print("="*60)
    print("[" + str(i+1) + "] " + row["variant_id"])
    print("  gene:", row["gene"] or "intergenic")
    print("  score:", round(row["effect_score"], 4))
    print("  prediction:", row["prediction"])
    
    disruptions = (row["top_disruptions"].split("|")[:8] if row["top_disruptions"] else [])
    disp_str = "Disruptions: " + ", ".join(disruptions)
    
    meta = "variant: " + row["variant_id"] + ", gene: " + (row["gene"] or "intergenic") + ", consequence: " + row["consequence"] + ", score: " + str(round(row["effect_score"], 4))
    effect = "effect_score=" + str(round(row["effect_score"], 4)) + ", prediction=" + row["prediction"]
    
    expl = explain_variant(
        variant_meta=meta,
        effect_summary=effect,
        disruption_table=disp_str,
        topk=min(8, len(disruptions))
    )
    print(expl)
    print()

