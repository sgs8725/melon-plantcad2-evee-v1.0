#!/bin/bash
# 自修复监控脚本：每60分钟检查一次，自动修复常见问题
PROJECT=/data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee
LOG=$PROJECT/logs/self_heal.log
TOTAL=32268

export PATH=/data/zhangcw/miniconda3/envs/plantcad/bin:$PATH
export LD_LIBRARY_PATH=/data/zhangcw/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/data/zhangcw/cuda-11.8

cd $PROJECT

echo "[$(date)] ====== 自修复监控启动 ======" >> $LOG

while true; do
    DONE=$(find data/activations -name '*.safetensors' 2>/dev/null | wc -l)
    REMAIN=$((TOTAL - DONE))
    
    # 检查是否有提取进程在运行
    EXTRACT_PID=$(ps aux | grep 'extract_activations' | grep -v grep | grep -v wait_and_train | awk '{print $2}')
    
    echo "[$(date)] 已处理: $DONE/$TOTAL, 进程PID: ${EXTRACT_PID:-无}" >> $LOG
    
    if [ $DONE -ge $TOTAL ]; then
        echo "[$(date)] ✅ 全部完成! 等待Step 3 (wait_and_train.sh负责)" >> $LOG
        sleep 3600
        continue
    fi
    
    if [ -z "$EXTRACT_PID" ]; then
        echo "[$(date)] ⚠️ 进程已退出! 尝试修复并重启..." >> $LOG
        
        # 检查日志中的错误类型
        LOGFILE=$(ls -t logs/*batch*.log logs/final_batch.log 2>/dev/null | head -1)
        if [ -n "$LOGFILE" ] && grep -q 'embedding\|FloatTensor' $LOGFILE 2>/dev/null; then
            echo "[$(date)] 🔧 修复 embedding 类型错误..." >> $LOG
            sed -i 's/return ids.to(.*)/return ids.to(self.device).long()/' meloncad/encoder.py
        fi
        
        # 如果剩余很多未处理，重新生成过滤列表并重启
        if [ $REMAIN -gt 0 ]; then
            echo "[$(date)] 🔄 重新生成剩余变异列表 ($REMAIN 个)..." >> $LOG
            python -c "
import polars as pl, os
act_dir = 'data/activations'
existing = set(os.path.splitext(f)[0] for f in os.listdir(act_dir) if f.endswith('.safetensors'))
df = pl.read_parquet('data/variants.parquet')
remaining = df.filter(~pl.col('variant_id').map_elements(
    lambda v: v.replace(':','_').replace('>','_') in existing, return_dtype=pl.Boolean
))
remaining.write_parquet('data/variants_remaining.parquet')
print(f'剩余: {len(remaining)}')
" >> $LOG 2>&1
            
            echo "[$(date)] 🚀 启动提取进程 (剩余 $REMAIN 个)..." >> $LOG
            nohup bash -c '. /data/zhangcw/miniconda3/etc/profile.d/conda.sh && conda activate plantcad && export LD_LIBRARY_PATH=/data/zhangcw/cuda-11.8/lib64:/data/zhangcw/app/libstdc++ && export CUDA_HOME=/data/zhangcw/cuda-11.8 && export CUDA_VISIBLE_DEVICES=0 && cd /data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee && python scripts/extract_activations.py --variants data/variants_remaining.parquet --model assets/plantcad2 --out data/activations --topk 256 --layer -1 --llr' > logs/auto_restart.log 2>&1 &
            echo "[$(date)] 重启PID=$!" >> $LOG
        fi
    fi
    
    # 每60分钟检查一次
    sleep 3600
done
