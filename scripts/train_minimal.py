#!/usr/bin/env python3
"""最小化GPU训练 - 快速验证Zhao 2019标签"""
import polars as pl, torch, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))
from probes.covariance_probe import CovarianceProbe

device = torch.device("cuda")
print(f"Device: {device}", flush=True)

# 采样5000条（2500正+2500负）
df = pl.read_parquet("data/variants_labeled_zhao_clean.parquet")
pos = df.filter(pl.col("label") == 1).sample(2500, seed=2024)
neg = df.filter(pl.col("label") == 0).sample(2500, seed=2024)
df = pl.concat([pos, neg]).sample(fraction=1.0, seed=2024)

import safetensors.torch
act_dir = Path("data/activations")

# 加载激活
acts, labs = [], []
for row in df.iter_rows(named=True):
    fname = act_dir / f"{row['variant_id'].replace(':', '_').replace('>', '_')}.safetensors"
    if fname.exists():
        acts.append(safetensors.torch.load_file(str(fname))["activations"].float())
        labs.append(int(row["label"]))
print(f"Loaded {len(acts)} samples", flush=True)

# 取前2000条放GPU
X = torch.stack(acts[:2000]).to(device)
Y = torch.tensor(labs[:2000], device=device)

# 划分
n_val = 400
X_val, Y_val = X[:n_val], Y[:n_val]
X_train, Y_train = X[n_val:], Y[n_val:]

probe = CovarianceProbe(d_model=768).to(device)
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)

bs = 128
for ep in range(1, 16):
    probe.train()
    perm = torch.randperm(len(X_train))
    total = 0.0
    for i in range(0, len(X_train), bs):
        idx = perm[i:i+bs]
        loss = torch.nn.CrossEntropyLoss()(probe(X_train[idx]), Y_train[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        total += float(loss)
    
    probe.eval()
    with torch.no_grad():
        p = torch.softmax(probe(X_val), dim=-1)[:, 1]
    
    # AUROC
    pairs = sorted(zip(p.cpu().tolist(), Y_val.cpu().tolist()))
    n_pos = sum(Y_val.cpu().tolist())
    n_neg = len(Y_val) - n_pos
    if n_pos > 0 and n_neg > 0:
        ranks = {}
        i = 0
        while i < len(pairs):
            j = i
            while j < len(pairs) and pairs[j][0] == pairs[i][0]:
                j += 1
            avg = (i + j - 1) / 2 + 1
            for k in range(i, j):
                ranks[k] = avg
            i = j
        sum_pos = sum(ranks[idx] for idx, (_, t) in enumerate(pairs) if t == 1)
        auroc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    else:
        auroc = float("nan")
    
    print(f"Epoch {ep:2d}  loss={total:.4f}  AUROC={auroc:.4f}", flush=True)
