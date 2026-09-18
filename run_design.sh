#!/bin/bash
. /data/zhangcw/miniconda3/etc/profile.d/conda.sh
conda activate plantcad
export LD_LIBRARY_PATH=/data/zhangcw/cuda-11.8/lib64:/data/zhangcw/app/libstdc++
export CUDA_HOME=/data/zhangcw/cuda-11.8
export CUDA_VISIBLE_DEVICES=0
cd /data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee
python scripts/design_elements.py --model assets/plantcad2 --element sugar --n 100 --length 400 --out results/designed_sugar_elements.fasta
