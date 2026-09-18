#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/smoke_test.py
=====================
端到端冒烟测试 —— 无需下载任何权重/数据、无需 mamba-ssm / GPU，用 mock
PlantCAD2（纯 PyTorch 复刻其对外接口）和合成数据，验证整条流水线
（编码器 -> 协方差探针 -> 注释探针 -> 扰动谱 -> 零样本 LLR ->
Gibbs 元件设计 -> 育种聚合）可以跑通。

用法：
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meloncad.encoder import build_mock_encoder
from probes.covariance_probe import CovarianceProbe
from probes.annotation_probe import (
    AnnotationProbe, disruption_profile, format_disruption_table,
)
from breeding.genomic_value import (
    Variant, Material, rank_candidates, DEFAULT_TRAITS,
)

D_MODEL = 256          # mock 用较小隐藏维以加速（真实 PlantCAD2-Small 为 768）


def rand_seq(n: int, seed: int = 0) -> str:
    import random
    r = random.Random(seed)
    return "".join(r.choice("ACGT") for _ in range(n))


def main() -> None:
    print("=" * 64)
    print("Melon-PlantCAD2-EVEE 端到端冒烟测试（mock PlantCAD2）")
    print("=" * 64)

    # ---- 1. 编码器（mock PlantCAD2：RC 双通道掩码 LM）----
    print("\n[1] 构造 mock PlantCAD2 编码器 ...")
    enc = build_mock_encoder(d_model=D_MODEL)
    print(f"    d_model(hidden_size) = {enc.d_model}")

    # ---- 2. 编码一个 SNV ----
    print("\n[2] 编码一个合成 SNV（300bp 窗口）...")
    ref = rand_seq(300, seed=1)
    alt = ref[:150] + ("A" if ref[150] != "A" else "C") + ref[151:]
    act = enc.encode_variant(ref, alt, topk=64)
    print(f"    激活张量 shape = {tuple(act.shape)}  （期望 [2,2,64,{D_MODEL}]）")
    assert tuple(act.shape) == (2, 2, 64, D_MODEL), "激活张量形状不符！"

    # ---- 2b. PlantCAD2 原生零样本掩码似然比（LLR）----
    print("\n[2b] PlantCAD2 零样本掩码似然比（LLR）...")
    llr = enc.score_variant_llr(ref, alt, var_index=150)
    print(f"    LLR = logP(alt) - logP(ref) = {llr:+.4f}")

    # ---- 3. 协方差探针前向 ----
    print("\n[3] 协方差探针前向（二分类）...")
    probe = CovarianceProbe(d_model=D_MODEL, n_outputs=2)
    with torch.no_grad():
        score = probe.predict_proba(act.unsqueeze(0))
    print(f"    P(不利) = {float(score[0]):.4f}")

    print("\n[3b] 协方差探针前向（多性状回归，8 个甜瓜性状）...")
    mt_probe = CovarianceProbe(d_model=D_MODEL, n_outputs=len(DEFAULT_TRAITS))
    with torch.no_grad():
        traits = mt_probe.predict_traits(act.unsqueeze(0))[0]
    for t, v in zip(DEFAULT_TRAITS, traits.tolist()):
        print(f"    {t:18s} = {v:.4f}")

    # ---- 4. 注释探针 + 扰动谱 ----
    print("\n[4] 注释探针 + 扰动谱 ...")
    ann = AnnotationProbe(d_model=D_MODEL)
    emb_ref = enc.encode_reference(ref)
    emb_alt = enc.encode_reference(alt)
    prof = disruption_profile(ann, emb_ref, emb_alt, var_pos=150)
    print("    扰动谱 Top-10：")
    print("    " + format_disruption_table(prof).replace("\n", "\n    "))

    # ---- 5. de novo 元件设计（掩码 Gibbs 重采样，少量迭代）----
    print("\n[5] de novo 调控元件设计（Gibbs，演示 3 轮）...")
    designs = enc.design_by_gibbs(length=60, n_iters=3, num_return_sequences=2)
    for i, s in enumerate(designs, 1):
        print(f"    设计 #{i}（{len(s)}bp）: {s[:40]}...")

    # ---- 6. 育种值聚合与排序 ----
    print("\n[6] 育种值聚合与候选排序 ...")
    materials = []
    for m in range(3):
        variants = []
        for v in range(4):
            r = rand_seq(300, seed=10 * m + v)
            a = r[:150] + ("G" if r[150] != "G" else "T") + r[151:]
            variants.append(Variant(
                variant_id=f"M{m}_V{v}",
                activations=enc.encode_variant(r, a, topk=64),
            ))
        materials.append(Material(material_id=f"Material_{m}", variants=variants))

    ranking = rank_candidates(materials, mt_probe)
    print("    候选材料排名（选择指数降序）：")
    for rank, item in enumerate(ranking, 1):
        print(f"    #{rank}  {item['material_id']}  "
              f"selection_index={item['selection_index']:+.4f}  "
              f"(n_variants={item['n_variants']})")

    print("\n" + "=" * 64)
    print("✓ 全部环节跑通。流水线结构正确。")
    print("  正式使用请：bash scripts/download_all.sh 下载真实 PlantCAD2 权重，")
    print("  再依次运行 build_variant_dataset → extract_activations →")
    print("  train_probe → predict。")
    print("=" * 64)


if __name__ == "__main__":
    main()
