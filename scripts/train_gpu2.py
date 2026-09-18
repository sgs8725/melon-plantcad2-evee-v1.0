#!/usr/bin/env python3
"""GPU训练v4 - 逐批堆叠（避免大tensor堆叠）"""
import polars as pl, torch, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))
from probes.covariance_probe import CovarianceProbe

device = torch.device("cuda")
print(f"[设备] {device}", flush=True)

# 加载分层采样数据
df = pl.read_parquet("data/variants_labeled_zhao_clean.parquet")
pos = df.filter(pl.col("label") == 1).sample(5000, seed=2024)
neg = df.filter(pl.col("label") == 0).sample(5000, seed=2024)

import safetensors.torch
act_dir = Path("data/activations")

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
print(f"  正={len(pos_acts)}", flush=True)
print("[加载] 负样本...", flush=True)
neg_acts, neg_labs = load_samples(neg)
print(f"  负={len(neg_acts)}", flush=True)

n = min(len(pos_acts), len(neg_acts))
# 交错混合 pos/neg
all_acts, all_labs = [], []
for i in range(n):
    all_acts.append(pos_acts[i]); all_labs.append(pos_labs[i])
    all_acts.append(neg_acts[i]); all_labs.append(neg_labs[i])

total = len(all_acts)
n_train = int(total * 0.8)
n_val = total - n_train
print(f"[数据] 总={total} 训练={n_train} 验证={n_val}", flush=True)

train_acts, train_labs = all_acts[:n_train], all_labs[:n_train]
val_acts, val_labs = all_acts[n_train:], all_labs[n_train:]

probe = CovarianceProbe(d_model=768).to(device)
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
bs = 128

def batch(acts, labs, bs, shuffle=True):
    n = len(acts)
    idx = list(range(n))
    if shuffle:
        import random
        random.shuffle(idx)
    for i in range(0, n, bs):
        ids = idx[i:i+bs]
        x = torch.stack([acts[j] for j in ids]).to(device)
        y = torch.tensor([labs[j] for j in ids], device=device)
        yield x, y

for ep in range(1, 16):
    probe.train()
    total_loss = 0.0
    for x, y in batch(train_acts, train_labs, bs):
        loss = torch.nn.CrossEntropyLoss()(probe(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += float(loss)
    
    probe.eval()
    with torch.no_grad():
        val_scores, val_true = [], []
        for x, y in batch(val_acts, val_labs, bs, shuffle=False):
            p = torch.softmax(probe(x), dim=-1)[:, 1]
            val_scores.extend(p.cpu().tolist())
            val_true.extend(y.cpu().tolist())
    
    # AUROC
    pairs = sorted(zip(val_scores, val_true))
    i = 0
    n_pos = sum(val_true); n_neg = len(val_true) - n_pos
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
    else:
        auroc = float("nan")
    
    print(f"E{ep:2d}  loss={total_loss:.2f}  AUROC={auroc:.4f}  val({n_pos}/{n_neg})", flush=True)
