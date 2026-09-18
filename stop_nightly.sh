#!/bin/bash
# 08:00 停止夜间批处理
PROJECT=/data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee
LOG=$PROJECT/logs/nightly_extract.log

# 杀掉所有相关进程
pkill -f extract_activations 2>/dev/null
pkill -f nightly_extract 2>/dev/null
echo "[$(date)] 🛑 夜间批处理已停止" >> $LOG
