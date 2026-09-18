#!/usr/bin/env python3
"""快速训练协方差探针 - 子集+GPU加速版"""
import polars as pl, torch, random, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

from probes.covariance_probe import CovarianceProbe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[设备] {device}", flush=True)

# 加载标签
df = pl.read_parquet("data/variants_labeled_zhao_clean.parquet")
print(f"[数据] {len(df)} 变异", flush=True)

# 平衡采样（5K正+5K负 = 10K）
pos = df.filter(pl.col("label") == 1)
neg = df.filter(pl.col("label") == 0)
n = min(5000, len(pos), len(neg))
pos = pos.sample(n, seed=2024)
neg = neg.sample(n, seed=2024)
df_sub = pl.concat([pos, neg]).sample(fraction=1.0, seed=2024)
print(f"[子集] 正={n} 负={n} 共={2*n}", flush=True)

act_dir = Path("data/activations")

def load_act(vid):
    import safetensors.torch
    fname = act_dir / f"{vid.replace(':', '_').replace('>', '_')}.safetensors"
    if not fname.exists():
        return None
    return safetensors.torch.load_file(str(fname))["activations"].float()

# 加载激活到CPU
samples = []
for row in df_sub.iter_rows(named=True):
    act = load_act(row["variant_id"])
    if act is not None:
        y = torch.tensor(int(row["label"]))
        samples.append((act.cpu(), y))
    if len(samples) % 2000 == 0 and len(samples) > 0:
        print(f"  加载 {len(samples)}/{len(df_sub)}", flush=True)

print(f"[样本] {len(samples)} 有效", flush=True)

random.Random(2024).shuffle(samples)
n_val = max(1, int(len(samples) * 0.2))
val, train = samples[:n_val], samples[n_val:]
print(f"[划分] 训练 {len(train)} / 验证 {len(val)}", flush=True)

# 探针
probe = CovarianceProbe(d_model=768, d_hidden=64, d_probe=128, n_outputs=2).to(device)
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)

def batches(data, bs=128):
    for i in range(0, len(data), bs):
        chunk = data[i:i+bs]
        x = torch.stack([c[0] for c in chunk]).to(device)
        y = torch.stack([c[1] for c in chunk]).to(device)
        yield x, y

# 训练
for ep in range(1, 31):
    probe.train()
    tot = 0.0
    for x, y in batches(train):
        loss = torch.nn.CrossEntropyLoss()(probe(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss)
    
    # 验证
    probe.eval()
    with torch.no_grad():
        ys, ss = [], []
        for x, y in batches(val):
            p = probe.predict_proba(x)
            ys.extend(y.cpu().tolist())
            ss.extend(p[:, 1].cpu().tolist())
    
    # AUROC
    pairs = sorted(zip(ss, ys))
    n_pos = sum(ys)
    n_neg = len(ys) - n_pos
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
    
    print(f"  epoch {ep:2d}  train_loss={tot:.4f}  val_AUROC={auroc:.4f}", flush=True)

print("\n[完成]", flush=True)
