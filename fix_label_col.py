#!/usr/bin/env python3
"""Fix duplicate label column in zhao labeled parquet"""
import pyarrow.parquet as pq, pyarrow as pa

table = pq.read_table("data/variants_labeled_zhao.parquet")
col_names = table.schema_arrow.names

# Rename columns to avoid duplicates
new_names = []
label_count = 0
for name in col_names:
    if name == "label":
        label_count += 1
        if label_count == 1:
            new_names.append("label_orig")
        else:
            new_names.append("label")
    else:
        new_names.append(name)

new_table = pa.table({new_names[i]: table.column(i) for i in range(len(col_names))})
pq.write_table(new_table, "data/variants_labeled_zhao.parquet")
print("Fixed columns:", new_table.schema_arrow.names)
print(f"Label distribution: {new_table.column('label').value_counts().to_pylist()}")
