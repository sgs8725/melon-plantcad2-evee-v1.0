#!/bin/bash
PROJECT=/data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee
LOG=$PROJECT/logs/wait_and_train.log

export PATH=/data/zhangcw/miniconda3/envs/plantcad/bin:$PATH
export LD_LIBRARY_PATH=/data/zhangcw/cuda-11.8/lib64:$LD_LIBRARY_PATH

cd $PROJECT

# 等待提取进程结束
EXTRACT_PID=$(ps aux | grep extract_activations | grep -v grep | awk '{print $2}')
if [ -n "$EXTRACT_PID" ]; then
    echo "[$(date)] 等待提取进程 $EXTRACT_PID 完成..." >> $LOG
    while kill -0 $EXTRACT_PID 2>/dev/null; do
        sleep 60
    done
fi

echo "[$(date)] 提取完成，开始Step 3..." >> $LOG

# 准备标签
python $PROJECT/prepare_labels.py >> $LOG 2>&1

# 训练探针
CUDA_VISIBLE_DEVICES=0 python scripts/train_probe.py     --activations data/activations     --labels data/variants_labeled.parquet     --out assets/probe_covariance.safetensors     --task classification --d-model 768 --epochs 30 >> $LOG 2>&1

echo "[$(date)] === 🎉 探针训练完成! ===" >> $LOG
