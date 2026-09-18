#!/bin/bash
PROJECT=/data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee
LOG=$PROJECT/logs/pipeline_monitor.log

export PATH=/data/zhangcw/miniconda3/envs/plantcad/bin:$PATH
export LD_LIBRARY_PATH=/data/zhangcw/cuda-11.8/lib64:/data/zhangcw/app/libstdc++:$LD_LIBRARY_PATH
export CUDA_HOME=/data/zhangcw/cuda-11.8

cd $PROJECT
echo "[$(date)] 监控脚本启动" >> $LOG

TOTAL=32268

while true; do
    PROC=$(ps aux | grep extract_activations | grep -v grep | wc -l)
    COUNT=$(ls $PROJECT/data/activations/*.safetensors 2>/dev/null | wc -l)
    echo "[$(date)] 已处理: $COUNT / $TOTAL, 进程: $PROC" >> $LOG

    if [ $PROC -eq 0 ] && [ $COUNT -ge $TOTAL ]; then
        echo "[$(date)] ✅ Step 2 完成! 共 $COUNT 个" >> $LOG
        break
    elif [ $PROC -eq 0 ] && [ $COUNT -lt $TOTAL ]; then
        echo "[$(date)] ⚠️ 异常退出: $COUNT/$TOTAL" >> $LOG
        tail -20 $PROJECT/logs/extract_activations.log >> $LOG
        break
    fi
    sleep 180
done

echo "[$(date)] === Step 3a: 准备标签 ===" >> $LOG
python $PROJECT/prepare_labels.py >> $LOG 2>&1

echo "[$(date)] === Step 3b: 训练探针 ===" >> $LOG
python scripts/train_probe.py     --activations data/activations     --labels data/variants_labeled.parquet     --out assets/probe_covariance.safetensors     --task classification     --d-model 768     --epochs 30 >> $LOG 2>&1

echo "[$(date)] === 🎉 探针训练完成! ===" >> $LOG
