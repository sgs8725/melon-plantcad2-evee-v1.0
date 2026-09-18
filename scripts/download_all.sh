#!/usr/bin/env bash
# scripts/download_all.sh
# ===========================================================================
# Melon-PlantCAD2-EVEE 一键下载：PlantCAD2 权重 + 甜瓜基因组/注释/变异/表型
#
# 用法：
#   bash scripts/download_all.sh             # 默认下载全部主源
#   bash scripts/download_all.sh check       # 只检查链接连通性
#   bash scripts/download_all.sh models      # 仅模型权重（含 Medium/Large）
#   bash scripts/download_all.sh genome      # 仅基因组 + 注释
#   bash scripts/download_all.sh variants    # 仅群体变异 / 入口
#   bash scripts/download_all.sh phenotype   # 仅表型数据 / 入口
# ===========================================================================
set -e
cd "$(dirname "$0")/.."

CMD=${1:-all}

echo ">>> Melon-PlantCAD2-EVEE 资源下载"
echo ">>> 目标目录: $(pwd)/assets, $(pwd)/data"
echo

case "$CMD" in
  check)
    python scripts/download_data.py --check
    ;;
  models)
    python scripts/download_data.py --models --large
    ;;
  genome)
    python scripts/download_data.py --genome
    ;;
  variants)
    python scripts/download_data.py --variants
    ;;
  phenotype)
    python scripts/download_data.py --phenotype
    ;;
  all|"")
    python scripts/download_data.py --all
    ;;
  *)
    echo "未知命令: $CMD"
    echo "可选: all | check | models | genome | variants | phenotype"
    exit 1
    ;;
esac

echo
echo ">>> 完成。各资源落地目录："
echo "    assets/plantcad2*/"
echo "    assets/genome/"
echo "    assets/variants/"
echo "    data/phenotype/"
ls -lh assets/ 2>/dev/null || true
ls -lh assets/genome/ 2>/dev/null || true
ls -lh assets/variants/ 2>/dev/null || true
ls -lh data/phenotype/ 2>/dev/null || true
