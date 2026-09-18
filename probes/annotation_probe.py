# -*- coding: utf-8 -*-
"""
probes/annotation_probe.py
==========================
监督注释探针与扰动谱 —— 移植 EVEE 的"annotation disruption profiling"。

EVEE 在 Evo2 参考嵌入上训练一组监督探针，预测一个功能注释面板
（artifacts/heads.feather 中 category=='disruption' 的约 369 个 head）。
对每个变异，计算变异 vs 参考在这些注释上的预测差值 Δ，得到"扰动谱"
——一份"该变异破坏/改变了哪些分子功能"的清单。

本文件把该范式迁移到 PlantCAD2 + 甜瓜功能注释面板。注释探针为逐位置
线性探针（与 EVEE token-level probe 一致），输入 PlantCAD2 的 RC 不变
逐位置嵌入（meloncad/encoder.py:encode_reference 产出 [L, d_model]）。

甜瓜功能注释面板（MELON_ANNOTATIONS）参照 PlantCAD2 的下游功能注释任务
（可及染色质、基因表达、翻译起始/终止、剪接等）设计，葫芦科与甜瓜特有功能
重点纳入：糖代谢/蔗糖积累（CmTST2 液泡糖转运、CmSPS、CmAGA2、CmSWEET）、
香气酯类（CmAAT/CmADH/LOX）、果实成熟与乙烯（CmACS1/CmACO1/CmNAC-NOR，
跃变型）、果肉色与类胡萝卜素（CmOr/PSY/CmKFB）、性别决定（CmACS7/CmWIP1/
CmACS11，甜瓜经典模型性状）、果肉/果皮特异染色质可及性、葫芦科抗病通路
（Fom 枯萎病、Vat 抗蚜抗病毒）。
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

# 甜瓜功能注释面板：注释探针要预测的每位置功能标签。
# 类别参照 PlantCAD2 的功能注释任务（chromatin accessibility / gene
# expression / translation / splice）与 EVEE heads 的分组思想；
# 与辣椒面板的差异集中在糖代谢、香气、乙烯成熟、性别决定等甜瓜特异通路。
MELON_ANNOTATIONS: list[str] = [
    # --- 基因组区域身份（来自 DHL92 参考注释 v4.0，对应 PlantCAD2 注释任务）---
    "region_CDS", "region_intron", "region_5UTR", "region_3UTR",
    "region_intergenic", "region_exon_boundary",
    # --- 剪接位点 ---
    "splice_donor", "splice_acceptor",
    # --- 编码效应 ---
    "codon_pos1", "codon_pos2", "codon_pos3",
    "start_codon", "stop_codon", "frameshift_risk",
    # --- 调控元件 ---
    "promoter_core", "promoter_proximal", "terminator", "cre_high_strength",
    # --- 转录因子结合位点（葫芦科 + 果实品质/成熟相关 TF 家族）---
    "tfbs_MYB",       # 次生代谢、果实品质调控（R2R3-MYB）
    "tfbs_bHLH",      # 与 MYB 互作，次生代谢协同
    "tfbs_WRKY",      # 抗病、抗逆
    "tfbs_AP2ERF",    # 乙烯信号 / 果实成熟（跃变型核心）
    "tfbs_NAC",       # 果实成熟master调控（CmNAC-NOR 型）
    "tfbs_MADS",      # 果实发育与成熟（RIN/TAGL 型 MADS）
    "tfbs_bZIP",      # 抗逆、ABA 信号
    "tfbs_generic",
    # --- 染色质可及性（甜瓜关键组织，对应 PlantCAD2 chromatin accessibility）---
    "chromatin_accessible_leaf",        # 叶片
    "chromatin_accessible_flesh",       # 果肉（糖分积累与色素合成核心部位）
    "chromatin_accessible_rind",        # 果皮 / 外皮（网纹、果色相关）
    # --- 糖代谢 / 蔗糖积累位点：甜瓜糖度核心 ---
    "sugar_metabolism_motif",           # CmTST2、CmSPS、CmAGA2、CmSWEET 关联调控元件
    # --- 香气酯类生物合成位点（CmAAT/CmADH/LOX）---
    "aroma_ester_motif",
    # --- 乙烯 / 果实成熟位点（CmACS1/CmACO1/CmNAC-NOR，跃变型）---
    "ripening_ethylene_motif",
    # --- 果肉色 / 类胡萝卜素位点（CmOr / PSY / CmKFB）---
    "carotenoid_color_motif",
    # --- 性别决定位点（CmACS7 a 位点 / CmWIP1 g 位点 / CmACS11，甜瓜经典模型）---
    "sex_determination_motif",
    # --- 翻译 / RNA ---
    "utr5_translation_motif", "mirna_target",
    # --- 保守性 ---
    "conserved_cross_species",
]


class AnnotationProbe(nn.Module):
    """逐位置监督注释探针。

    由 PlantCAD2 逐位置嵌入 [L, d_model] 预测每个位置的一组功能注释概率。
    每个注释为二分类（存在/不存在），用 BCEWithLogits 训练。
    """

    def __init__(self, d_model: int = 768, annotations: list[str] | None = None):
        super().__init__()
        self.annotations = annotations or MELON_ANNOTATIONS
        self.n_ann = len(self.annotations)
        self.linear = nn.Linear(d_model, self.n_ann)

    def forward(self, emb: Tensor) -> Tensor:
        """emb: [L, d_model] 或 [B, L, d_model] → 注释 logit。"""
        return self.linear(emb)

    @torch.no_grad()
    def predict(self, emb: Tensor) -> Tensor:
        """返回每位置每注释的概率（sigmoid 后）。"""
        return torch.sigmoid(self.linear(emb))


@torch.no_grad()
def disruption_profile(
    probe: AnnotationProbe,
    emb_ref: Tensor,
    emb_alt: Tensor,
    var_pos: int,
    flank: int = 5,
    topk: int = 10,
    regression: bool = False,
) -> dict:
    """计算一个变异的注释扰动谱。

    Parameters
    ----------
    probe : AnnotationProbe
        已训练的注释探针。
    emb_ref, emb_alt : Tensor
        参考 / 变异序列的逐位置嵌入 [L, d_model]（来自 MelonCADEncoder）。
    var_pos : int
        变异在嵌入序列中的位置索引。
    flank : int
        除突变位点外，左右各取多少侧翼位置一并计入（EVEE 取 ±5）。
    topk : int
        返回扰动幅度最大的前 K 个注释。
    regression : bool
        False（默认）：二值注释探针，取 sigmoid 概率差作为扰动；
        True：连续信号回归探针，直接取原始预测信号差（不过 sigmoid）。

    Returns
    -------
    dict
        delta            : [L, n_ann]  全序列注释扰动
        top_annotations  : list[str]   扰动幅度最大的 K 个注释名
        top_scores       : list[float] 对应的（带符号）扰动值
    """
    if regression:
        a_ref = probe(emb_ref)                   # 原始预测信号（无 sigmoid）
        a_alt = probe(emb_alt)
    else:
        a_ref = probe.predict(emb_ref)           # [L, n_ann] sigmoid 概率
        a_alt = probe.predict(emb_alt)
    L = min(a_ref.shape[0], a_alt.shape[0])
    delta = a_alt[:L] - a_ref[:L]                # 注释扰动 Δ

    lo = max(0, var_pos - flank)
    hi = min(L, var_pos + flank + 1)
    local = delta[lo:hi]                         # [窗口, n_ann]

    # 每个注释取窗口内绝对值最大的扰动（保留符号）
    abs_local = local.abs()
    pos_of_max = abs_local.argmax(dim=0)         # [n_ann]
    signed = local[pos_of_max, torch.arange(local.shape[1])]   # [n_ann]

    order = signed.abs().topk(k=min(topk, signed.shape[0])).indices
    return {
        "delta": delta,
        "top_annotations": [probe.annotations[i] for i in order.tolist()],
        "top_scores": [round(float(signed[i]), 4) for i in order.tolist()],
    }


def format_disruption_table(profile: dict) -> str:
    """把扰动谱整理成可读文本表，供 LLM 综合解释使用。"""
    rows = ["注释项\t扰动值(变异-参考)"]
    for name, score in zip(profile["top_annotations"], profile["top_scores"]):
        arrow = "↑" if score > 0 else "↓"
        rows.append(f"{name}\t{arrow} {score:+.4f}")
    return "\n".join(rows)
