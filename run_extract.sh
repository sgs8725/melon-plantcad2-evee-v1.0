#!/bin/bash
export PATH=/data/zhangcw/miniconda3/envs/plantcad/bin:$PATH
export LD_LIBRARY_PATH=/data/zhangcw/cuda-11.8/lib64:/data/zhangcw/app/libstdc++:$LD_LIBRARY_PATH
export CUDA_HOME=/data/zhangcw/cuda-11.8
export CUDA_VISIBLE_DEVICES=0
cd /data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee
python scripts/extract_activations.py     --variants data/variants.parquet     --model assets/plantcad2     --out data/activations     --topk 256 --layer -1     --llr 2>&1
