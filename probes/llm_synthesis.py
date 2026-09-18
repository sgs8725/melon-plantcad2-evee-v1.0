# -*- coding: utf-8 -*-
"""
probes/llm_synthesis.py
=======================
LLM 机制解释合成 —— 移植 EVEE 第三组件。

EVEE 取扰动谱 Top-10 + 变异元数据，送入前沿推理 LLM，合成人类可读的
致病机制解释。本文件把提示词从临床语境改写为甜瓜育种语境。

若未安装 anthropic 或无 API key，explain_variant 会回退到一个基于扰动谱的
模板式本地解释，保证流水线不中断。
"""
from __future__ import annotations

PROMPT_TEMPLATE = """你是甜瓜（Cucumis melo，葫芦科）分子设计育种专家。
请基于以下结构化证据，用通俗但严谨的中文解释该变异的分子机制及其对育种的意义。

【变异信息】
{variant_meta}

【效应预测（协方差探针 + PlantCAD2 零样本掩码似然比 LLR）】
{effect_summary}

【注释扰动谱（Top-{topk}，按扰动幅度排序）】
{disruption_table}

请输出三段：
(1) 分子机制：该变异最可能破坏或改变了什么分子功能（特别关注甜瓜的
    糖度/蔗糖积累通路 CmTST2—CmSPS—CmAGA2—CmSWEET、香气酯类通路 CmAAT/CmADH/LOX、
    果实成熟与乙烯通路 CmACS1/CmACO1—CmNAC-NOR（跃变型）、果肉色与类胡萝卜素
    CmOr/PSY/CmKFB、性别决定 CmACS7/CmWIP1/CmACS11、葫芦科抗病通路如白粉病、
    枯萎病 Fom-1/Fom-2、抗蚜抗病毒 Vat 等）；
(2) 农艺与品质影响：由此推断对单果重 / 含糖量（糖度 Brix）/ 果肉色 / 香气 /
    耐贮性（成熟期/货架期）/ 抗病性 / 抗逆性的影响方向；
(3) 育种建议：是否保留该等位、是否适合作为基因编辑靶点、推荐的利用方式
    （如：用作 MAS 标记 / CRISPR 敲除或敲入 / 杂交制种亲本选配 /
    高糖风味品系 / 耐贮长货架期品系）。
"""


def _local_fallback(variant_meta: str, effect_summary: str,
                     disruption_table: str) -> str:
    """无 LLM 时的模板式解释。"""
    return (
        "【分子机制】（本地模板，未调用 LLM）\n"
        f"根据注释扰动谱，受影响最大的功能为：\n{disruption_table}\n\n"
        "【农艺与品质影响】\n"
        f"效应预测：{effect_summary}\n"
        "扰动方向为正表示该功能被增强、为负表示被削弱，"
        "可据此结合甜瓜糖代谢—香气—乙烯成熟—果肉色—抗病等性状先验"
        "判断影响方向。\n\n"
        "【育种建议】\n"
        "建议结合田间表型与下表扰动谱由育种家复核；"
        "若效应为强不利且位于关键功能区，可作为基因编辑修正靶点。\n"
        f"\n变异信息：{variant_meta}"
    )


def explain_variant(
    variant_meta: str,
    effect_summary: str,
    disruption_table: str,
    topk: int = 10,
    model: str = "claude-opus-4-7",
    api_key: str | None = None,
) -> str:
    """合成变异的育种机制解释。

    Parameters
    ----------
    variant_meta : str
        变异元数据（基因、坐标、后果类型等）。
    effect_summary : str
        协方差探针 + LLR 的效应预测摘要。
    disruption_table : str
        format_disruption_table() 产出的扰动谱文本表。
    model : str
        Anthropic 模型名。
    api_key : str | None
        若为 None，则尝试从环境变量 ANTHROPIC_API_KEY 读取。

    Returns
    -------
    str  自然语言机制解释。
    """
    prompt = PROMPT_TEMPLATE.format(
        variant_meta=variant_meta,
        effect_summary=effect_summary,
        disruption_table=disruption_table,
        topk=topk,
    )
    try:
        import os
        import anthropic

        client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        msg = client.messages.create(
            model=model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:                       # 无 key / 无网络 / 未安装
        return _local_fallback(variant_meta, effect_summary, disruption_table) \
            + f"\n\n[注：未能调用 LLM（{type(e).__name__}），已使用本地模板。]"
