# -*- coding: utf-8 -*-
"""
breeding/genomic_value.py
=========================
甜瓜智能育种模型：多性状效应聚合与候选排序。

把协方差探针对单个变异的多性状效应，聚合为材料（基因型）级别的
基因组育种值估计，并对候选材料 / 杂交组合 / 编辑方案排序。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

# 默认目标性状（与协方差探针 n_outputs 对应）。
# 训练探针时 n_outputs 应等于 len(DEFAULT_TRAITS)。
#
# 甜瓜与黄瓜 / 辣椒的关键差异：
#  - 采收器官是果实，最具甜瓜特异性的品质性状是「含糖量（糖度）」，由蔗糖积累
#    通路（CmTST2 液泡糖转运、CmSPS 蔗糖磷酸合酶、CmAGA2 碱性 α-半乳糖苷酶、
#    CmSWEET 等）调控，存在主效 QTL，适合 GFM + 扰动谱方法；
#  - 风味维度需刻画香气（酯类，CmAAT/CmADH/LOX 通路），是甜瓜区别于多数果蔬的
#    标志性品质；
#  - 采后维度由「跃变型 / 非跃变型」乙烯成熟特性决定耐贮性（CmACS1/CmACO1/
#    CmNAC-NOR），直接关系货架期；
#  - 果肉色由 CmOr（β-胡萝卜素，橙肉）等控制；
#  - 葫芦科抗病压力大（白粉病 / 枯萎病 Fom / 病毒病，Vat 抗蚜抗病毒），抗病性单列。
# 因此性状清单覆盖产量、品质核心(糖度)、果实外观、风味、采后、抗病、抗逆等维度。
DEFAULT_TRAITS: list[str] = [
    "yield",            # 单位面积产量
    "fruit_weight",     # 单果重
    "sugar_content",    # 含糖量 / 糖度（Brix，蔗糖积累）—— 甜瓜核心品质
    "flesh_color",      # 果肉颜色（CmOr β-胡萝卜素 / 橙肉 vs 白肉/绿肉）
    "aroma",            # 香气（酯类挥发物，CmAAT 等）—— 甜瓜标志性风味
    "shelf_life",       # 耐贮性 / 货架期（跃变型乙烯成熟 vs 非跃变型）
    "disease_resist",   # 抗病性（白粉病 / 枯萎病 Fom / 病毒病聚合）
    "stress_tolerance", # 耐逆（耐裂果 / 耐高温 / 耐盐，设施栽培关键）
]


@dataclass
class Variant:
    """一个变异及其 PlantCAD2 激活张量。"""
    variant_id: str
    activations: torch.Tensor          # [2, 2, K, d_model]
    gene: str = ""
    consequence: str = ""


@dataclass
class Material:
    """一个候选材料（基因型）：相对参考基因组的变异集合。"""
    material_id: str
    variants: list[Variant] = field(default_factory=list)


@torch.no_grad()
def variant_trait_effects(probe, activations: torch.Tensor) -> torch.Tensor:
    """单变异多性状效应。

    probe 的 n_outputs == n_traits。返回 [n_traits]，
    经 (sigmoid-0.5)*2 映射到 [-1, 1]：正=有利，负=不利，绝对值=幅度。
    """
    if activations.dim() == 4:
        activations = activations.unsqueeze(0)     # -> [1, 2, 2, K, d]
    proba = probe.predict_traits(activations)[0]   # [n_traits]，已 sigmoid
    return (proba - 0.5) * 2.0                     # [-1, 1]


@torch.no_grad()
def material_breeding_value(
    material: Material,
    probe,
    trait_weights: torch.Tensor | None = None,
    traits: list[str] | None = None,
) -> dict:
    """聚合一个材料全部变异，得到多性状育种值与综合选择指数。

    Returns
    -------
    dict
        per_trait        : {性状: 累计效应}
        selection_index  : float  加权综合选择指数
        n_variants       : int
    """
    traits = traits or DEFAULT_TRAITS
    n = len(traits)
    if trait_weights is None:
        trait_weights = torch.ones(n)

    total = torch.zeros(n)
    for v in material.variants:
        total += variant_trait_effects(probe, v.activations)

    index = float((total * trait_weights).sum())
    return {
        "material_id": material.material_id,
        "per_trait": {t: round(float(s), 4) for t, s in zip(traits, total)},
        "selection_index": round(index, 4),
        "n_variants": len(material.variants),
    }


@torch.no_grad()
def rank_candidates(
    materials: list[Material],
    probe,
    trait_weights: torch.Tensor | None = None,
    traits: list[str] | None = None,
) -> list[dict]:
    """对候选材料按综合选择指数降序排名。"""
    scored = [
        material_breeding_value(m, probe, trait_weights, traits)
        for m in materials
    ]
    return sorted(scored, key=lambda x: x["selection_index"], reverse=True)
