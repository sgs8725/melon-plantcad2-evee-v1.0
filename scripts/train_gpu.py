#!/usr/bin/env python3
"""GPU训练v3 - 修复标签划分问题"""
import polars as pl, torch, sys, os, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))
from probes.covariance_probe import CovarianceProbe

device = torch.device("cuda")
print(f"[设备] {device}", flush=True)

# 加载并分层采样
df = pl.read_parquet("data/variants_labeled_zhao_clean.parquet")
pos = df.filter(pl.col("label") == 1).sample(3000, seed=2024)
neg = df.filter(pl.col("label") == 0).sample(3000, seed=2024)

import safetensors.torch
act_dir = Path("data/activations")

# 分别加载正负样本激活
def load_samples(df_sub):
    acts, labs = [], []
    for row in df_sub.iter_rows(named=True):
        fname = act_dir / f"{row['variant_id'].replace(':', '_').replace('>', '_')}.safetensors"
        if fname.exists():
            acts.append(safetensors.torch.load_file(str(fname))["activations"].float())
            labs.append(int(row["label"]))
    return acts, labs

print("[加载] 正样本...", flush=True)
pos_acts, pos_labs = load_samples(pos)
print(f"  正样本: {len(pos_acts)}", flush=True)

print("[加载] 负样本...", flush=True)
neg_acts, neg_labs = load_samples(neg)
print(f"  负样本: {len(neg_acts)}", flush=True)

# 各取相同数量，交错混合
n = min(len(pos_acts), len(neg_acts)) // 2 * 2
pos_acts, pos_labs = pos_acts[:n], pos_labs[:n]
neg_acts, neg_labs = neg_acts[:n], neg_labs[:n]

# 交错排列: [pos1, neg1, pos2, neg2, ...]
acts = []
labs = []
for i in range(n):
    acts.append(pos_acts[i])
    labs.append(pos_labs[i])
    acts.append(neg_acts[i])
    labs.append(neg_labs[i])

total = len(acts)
n_train = int(total * 0.8)
n_val = total - n_train

print(f"[数据] 总={total} 训练={n_train} 验证={n_val}", flush=True)

# 放GPU
print("[GPU] 堆叠张量...", flush=True)
X = torch.stack(acts).to(device)
Y = torch.tensor(labs, device=device)
print(f"[GPU] X.shape={X.shape}", flush=True)

X_train, Y_train = X[:n_train], Y[:n_train]
X_val, Y_val = X[n_train:], Y[n_train:]

probe = CovarianceProbe(d_model=768).to(device)
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
bs = 128

for ep in range(1, 16):
    probe.train()
    perm = torch.randperm(n_train, device=device)
    total_loss = 0.0
    for i in range(0, n_train, bs):
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
    i = 0
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos > 0 and n_neg > 0:
        ranks = {}
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
    
    print(f"Epoch {ep:2d}  loss={total_loss:.4f}  AUROC={auroc:.4f}  val_pos={n_pos}/{n_neg}", flush=True)
