#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/design_elements.py
==========================
基于 PlantCAD2（掩码语言模型）生成能力的 de novo 甜瓜调控元件设计。

与 PlantGFM 不同，PlantCAD2（Caduceus + Mamba2）是**双向掩码语言模型**，
没有自回归 ``generate``；正确的 de novo 设计方式是「迭代掩码重采样 /
Gibbs 采样」：从模板（或随机）序列出发，每轮随机掩掉一部分位置，由 MLM
预测分布重采样填回，多轮后收敛到符合甜瓜序列语法的候选元件
（``MelonCADEncoder.design_by_gibbs``）。

随后用已训练的协方差探针 / 注释探针 / 零样本 LLR 对生成序列做效应打分，
形成「生成-评估」闭环，再交育种家复核。

三类目标元件：
  1. flesh    —— 果肉特异启动子（驱动糖代谢 CmTST2/CmSPS 与色素 CmOr 等基因）
  2. ripening —— 果实成熟相关启动子（乙烯 CmACS1/CmACO1、CmNAC-NOR 调控）
  3. sugar    —— 糖度/蔗糖积累通路（CmTST2/CmSPS/CmAGA2）调控元件

用法：
    python scripts/design_elements.py \
        --model assets/plantcad2 \
        --element sugar --n 1000 --length 400 \
        --out results/designed_sugar_elements.fasta

冒烟测试（CPU，无需权重）：
    python scripts/design_elements.py --smoke-test --n 5 --length 60 --out /tmp/d.fasta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meloncad.encoder import MelonCADEncoder, build_mock_encoder


# 三类元件的设计先验：起始模板骨架（None 表示随机起始）与采样温度
ELEMENT_PRESETS = {
    # 果肉特异启动子：富 AT 的 TATA 近端 + CAAT，偏保守起始
    "flesh": {
        "template": None,
        "temperature": 0.9,
        "desc": "果肉特异启动子（驱动 CmTST2/CmSPS 糖代谢与 CmOr 色素等基因）",
    },
    # 果实成熟启动子：与乙烯 / CmNAC-NOR 跃变型成熟相关
    "ripening": {
        "template": None,
        "temperature": 0.9,
        "desc": "果实成熟相关启动子（乙烯 CmACS1/CmACO1、CmNAC-NOR 调控）",
    },
    # 糖度/蔗糖积累调控元件：CmTST2/CmSPS/CmAGA2 上游顺式元件
    "sugar": {
        "template": None,
        "temperature": 1.0,
        "desc": "糖度/蔗糖积累通路（CmTST2/CmSPS/CmAGA2）调控元件",
    },
}


def read_first_fasta(path: str) -> str:
    seq: list[str] = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if seq:
                    break
                continue
            seq.append(line.strip())
    return "".join(seq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="assets/plantcad2",
                    help="PlantCAD2 权重目录或 HF 名（trust_remote_code）")
    ap.add_argument("--element", choices=sorted(ELEMENT_PRESETS),
                    default="sugar", help="目标调控元件类别")
    ap.add_argument("--template-fasta", default=None,
                    help="可选：已知元件骨架 FASTA（取第一条作为 Gibbs 起始模板）")
    ap.add_argument("--n", type=int, default=100, help="生成候选数")
    ap.add_argument("--length", type=int, default=400,
                    help="生成长度（无模板时使用；上限受 8192bp 上下文约束）")
    ap.add_argument("--n-iters", type=int, default=200, help="Gibbs 迭代轮数")
    ap.add_argument("--mask-frac", type=float, default=0.1, help="每轮掩码比例")
    ap.add_argument("--temperature", type=float, default=None,
                    help="采样温度（默认取元件预设）")
    ap.add_argument("--out", required=True, help="输出 FASTA")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    preset = ELEMENT_PRESETS[args.element]
    temperature = args.temperature if args.temperature is not None \
        else preset["temperature"]

    if args.smoke_test:
        print("[冒烟测试] 随机初始化 mock PlantCAD2（纯 CPU）。")
        enc = build_mock_encoder(d_model=256)
    else:
        enc = MelonCADEncoder(model_path=args.model)

    template = (read_first_fasta(args.template_fasta)
                if args.template_fasta else preset["template"])
    eff_len = len(template) if template else args.length
    print(f"[设计] 类别={args.element}（{preset['desc']}）")
    print(f"       模板长度={eff_len}  目标 {args.n} 条  "
          f"Gibbs {args.n_iters} 轮 / 掩码 {args.mask_frac:.0%} / T={temperature}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fout:
        done = 0
        batch = 8
        while done < args.n:
            k = min(batch, args.n - done)
            seqs = enc.design_by_gibbs(
                length=args.length,
                template=template,
                n_iters=args.n_iters,
                mask_frac=args.mask_frac,
                temperature=temperature,
                num_return_sequences=k,
                seed=done,  # 不同批次不同种子
            )
            for s in seqs:
                done += 1
                fout.write(f">designed_{args.element}_{done}\n{s}\n")
            print(f"  已生成 {done}/{args.n}")

    print(f"[完成] de novo 甜瓜调控元件 -> {out}")
    print("[提示] 下一步：用 scripts/extract_activations.py + predict.py "
          "对这些序列做效应/LLR 筛选，再交育种家复核。")


if __name__ == "__main__":
    main()
