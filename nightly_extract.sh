#!/bin/bash
PROJECT=/data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee
LOG=$PROJECT/logs/nightly_extract.log

export PATH=/data/zhangcw/miniconda3/envs/plantcad/bin:$PATH
export LD_LIBRARY_PATH=/data/zhangcw/cuda-11.8/lib64:/data/zhangcw/app/libstdc++:$LD_LIBRARY_PATH
export CUDA_HOME=/data/zhangcw/cuda-11.8
export CUDA_VISIBLE_DEVICES=0

cd $PROJECT
mkdir -p data/activations

echo "[$(date)] ====== 夜间批处理启动 ======" >> $LOG

# 统计已完成数量
DONE=$(ls data/activations/*.safetensors 2>/dev/null | wc -l)
TOTAL=32268
echo "[$(date)] 已处理: $DONE/$TOTAL" >> $LOG

if [ $DONE -ge $TOTAL ]; then
    echo "[$(date)] ✅ 全部已完成，进入Step 3" >> $LOG
    python $PROJECT/prepare_labels.py >> $LOG 2>&1
    trap '' EXIT
    python scripts/train_probe.py --activations data/activations --labels data/variants_labeled.parquet --out assets/probe_covariance.safetensors --task classification --d-model 768 --epochs 30 >> $LOG 2>&1
    echo "[$(date)] === 🎉 探针训练完成! ===" >> $LOG
    exit 0
fi

# 创建过滤后的变异列表（跳过已处理的）
python -c "
import polars as pl, os
PROJ = '$PROJECT'
df = pl.read_parquet(f'{PROJ}/data/variants.parquet')
act_dir = f'{PROJ}/data/activations'
os.makedirs(act_dir, exist_ok=True)
existing = set(os.path.splitext(f)[0] for f in os.listdir(act_dir) if f.endswith('.safetensors'))
# 过滤未处理的
remaining = df.filter(~pl.col('variant_id').map_elements(
    lambda v: v.replace(':', '_').replace('>', '_') in existing, return_dtype=pl.Boolean
))
print(f'剩余变异: {len(remaining)}')
remaining.write_parquet(f'{PROJ}/data/variants_remaining.parquet')
" >> $LOG 2>&1

REMAIN=$(python -c "import polars as pl; print(pl.read_parquet('$PROJECT/data/variants_remaining.parquet').shape[0])" 2>/dev/null)
echo "[$(date)] 本批将处理: $REMAIN 个变异" >> $LOG

if [ $REMAIN -le 0 ]; then
    echo "[$(date)] ⚠️ 无剩余变异" >> $LOG
    exit 0
fi

# 运行激活提取（只处理剩余变异）
python scripts/extract_activations.py     --variants data/variants_remaining.parquet     --model assets/plantcad2     --out data/activations     --topk 256 --layer -1     --llr >> $LOG 2>&1

EXIT_CODE=$?
DONE_NOW=$(ls data/activations/*.safetensors 2>/dev/null | wc -l)
echo "[$(date)] 批处理结束，退出码: $EXIT_CODE, 累计完成: $DONE_NOW/$TOTAL" >> $LOG

# 如果全部完成，自动进入Step 3
if [ $DONE_NOW -ge $TOTAL ]; then
    echo "[$(date)] ✅ 全部完成! 进入Step 3..." >> $LOG
    python $PROJECT/prepare_labels.py >> $LOG 2>&1
    python scripts/train_probe.py --activations data/activations --labels data/variants_labeled.parquet --out assets/probe_covariance.safetensors --task classification --d-model 768 --epochs 30 >> $LOG 2>&1
    echo "[$(date)] === 🎉 全部完成! ===" >> $LOG
fi
