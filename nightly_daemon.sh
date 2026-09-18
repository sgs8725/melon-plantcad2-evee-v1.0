#!/bin/bash
# 后台守护进程：23:00启动，8:00停止
PROJECT=/data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee
LOG=$PROJECT/logs/nightly_daemon.log
PID_FILE=$PROJECT/.extract_pid

export PATH=/data/zhangcw/miniconda3/envs/plantcad/bin:$PATH
export LD_LIBRARY_PATH=/data/zhangcw/cuda-11.8/lib64:/data/zhangcw/app/libstdc++:$LD_LIBRARY_PATH
export CUDA_HOME=/data/zhangcw/cuda-11.8

cd $PROJECT

echo "[$(date)] 守护进程启动" >> $LOG

while true; do
    HOUR=$(date +%H)
    
    # 23:00-7:59 = 夜间窗口，启动处理
    if [ $HOUR -ge 23 ] || [ $HOUR -lt 8 ]; then
        
        # 检查是否已经在运行
        if [ -f $PID_FILE ] && kill -0 $(cat $PID_FILE) 2>/dev/null; then
            # 进程已在运行，等待
            sleep 1800
            continue
        fi
        
        # 统计已完成数量
        DONE=$(ls data/activations/*.safetensors 2>/dev/null | wc -l)
        TOTAL=32268
        
        echo "[$(date)] ====== 批处理启动 (已处理 $DONE/$TOTAL) ======" >> $LOG
        
        if [ $DONE -ge $TOTAL ]; then
            echo "[$(date)] ✅ 全部完成，进入Step 3" >> $LOG
            python $PROJECT/prepare_labels.py >> $LOG 2>&1
            CUDA_VISIBLE_DEVICES=0 python scripts/train_probe.py                 --activations data/activations                 --labels data/variants_labeled.parquet                 --out assets/probe_covariance.safetensors                 --task classification --d-model 768 --epochs 30 >> $LOG 2>&1
            echo "[$(date)] === 🎉 全部完成! ===" >> $LOG
            rm -f $PID_FILE
            exit 0
        fi
        
        # 创建过滤列表（跳过已处理的）
        python -c "
import polars as pl, os
act_dir = 'data/activations'
if not os.path.exists(act_dir): os.makedirs(act_dir)
existing = set(os.path.splitext(f)[0] for f in os.listdir(act_dir) if f.endswith('.safetensors'))
df = pl.read_parquet('data/variants.parquet')
remaining = df.filter(~pl.col('variant_id').map_elements(
    lambda v: v.replace(':','_').replace('>','_') in existing, return_dtype=pl.Boolean
))
if len(remaining) > 0:
    remaining.write_parquet('data/variants_remaining.parquet')
    print(f'剩余: {len(remaining)}')
else:
    print('0')
" >> $LOG 2>&1
        
        REMAIN=$(python -c "import polars as pl; print(pl.read_parquet('data/variants_remaining.parquet').shape[0])" 2>/dev/null)
        
        if [ "$REMAIN" = "0" ] || [ -z "$REMAIN" ]; then
            echo "[$(date)] 全部完成!" >> $LOG
            python $PROJECT/prepare_labels.py >> $LOG 2>&1
            CUDA_VISIBLE_DEVICES=0 python scripts/train_probe.py                 --activations data/activations                 --labels data/variants_labeled.parquet                 --out assets/probe_covariance.safetensors                 --task classification --d-model 768 --epochs 30 >> $LOG 2>&1
            echo "[$(date)] === 🎉 全部完成! ===" >> $LOG
            rm -f $PID_FILE
            exit 0
        fi
        
        echo "[$(date)] 处理 $REMAIN 个变异..." >> $LOG
        CUDA_VISIBLE_DEVICES=0 python scripts/extract_activations.py             --variants data/variants_remaining.parquet             --model assets/plantcad2             --out data/activations             --topk 256 --layer -1             --llr >> $LOG 2>&1
        
        echo "[$(date)] 批处理结束" >> $LOG
        
    else
        # 白天(8:00-22:59)：确保没有残留进程
        if [ -f $PID_FILE ]; then
            kill $(cat $PID_FILE) 2>/dev/null
            rm -f $PID_FILE
            echo "[$(date)] 🛑 白天已停止" >> $LOG
        fi
        # 白天每30分钟检查一次
        sleep 1800
    fi
done
