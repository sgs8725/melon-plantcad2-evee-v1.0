#!/usr/bin/env python3
"""GPU训练 - 批量加载+逐批GPU计算，充分等待"""
import polars as pl, torch, sys, os, gc
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))
from probes.covariance_probe import CovarianceProbe

device = torch.device("cuda")
print(f"[设备] {device}", flush=True)

# 加载标签
df = pl.read_parquet("data/variants_labeled_zhao_clean.parquet")
pos_df = df.filter(pl.col("label") == 1)
neg_df = df.filter(pl.col("label") == 0)
n = min(2500, len(pos_df), len(neg_df))
pos = pos_df.sample(n, seed=2024)
neg = neg_df.sample(n, seed=2024)
df = pl.concat([pos, neg])
sample_limit = n * 2

import safetensors.torch
act_dir = Path("data/activations")

# 加载平衡样本
acts, labs = [], []
for row in df.iter_rows(named=True):
    fname = act_dir / f"{row['variant_id'].replace(':', '_').replace('>', '_')}.safetensors"
    if fname.exists():
        t = safetensors.torch.load_file(str(fname))["activations"].float()
        acts.append(t)
        labs.append(int(row["label"]))
    if len(acts) >= sample_limit:
        break

print(f"[数据] {len(acts)} 样本 ({sum(labs)}正/{len(acts)-sum(labs)}负)", flush=True)

# 打乱顺序
import random
tmp = list(zip(acts, labs))
random.shuffle(tmp)
acts, labs = zip(*tmp) if tmp else ([], [])
acts, labs = list(acts), list(labs)

# 一次性全部搬到GPU（1000个×3MB=3GB，GPU有48GB，足够）
print("[GPU] 堆叠并搬运...", flush=True)
X = torch.stack(acts).to(device)
Y = torch.tensor(labs, device=device)
del acts, labs; gc.collect()
print(f"  X.shape={X.shape}  X.device={X.device}", flush=True)

n_val = max(1, int(len(X) * 0.2))
X_val, Y_val = X[:n_val], Y[:n_val]
X_train, Y_train = X[n_val:], Y[n_val:]

probe = CovarianceProbe(d_model=768).to(device)
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
bs = 128

for ep in range(1, 31):
    probe.train()
    perm = torch.randperm(len(X_train), device=device)
    total_loss = 0.0
    for i in range(0, len(X_train), bs):
        idx = perm[i:i+bs]
        loss = torch.nn.CrossEntropyLoss()(probe(X_train[idx]), Y_train[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += float(loss)
    
    probe.eval()
    with torch.no_grad():
        p = torch.softmax(probe(X_val), dim=-1)[:, 1]
    
    # AUROC
    y_true = Y_val.cpu().tolist()
    y_score = p.cpu().tolist()
    pairs = sorted(zip(y_score, y_true))
    i = 0; n_pos = sum(y_true); n_neg = len(y_true) - n_pos
    if n_pos > 0 and n_neg > 0:
        while i < len(pairs):
            j = i
            while j < len(pairs) and pairs[j][0] == pairs[i][0]:
                j += 1
            avg = (i + j - 1) / 2 + 1
            for k in range(i, j):
                pairs[k] = (pairs[k][0], pairs[k][1], avg)
            i = j
        sum_pos = sum(v[2] for v in pairs if v[1] == 1)
        auroc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        print(f"E{ep:2d}  loss={total_loss:.4f}  AUROC={auroc:.4f}  v({n_pos}/{n_neg})", flush=True)
    else:
        print(f"E{ep:2d}  loss={total_loss:.4f}  AUROC=NaN (n_pos={n_pos})", flush=True)
