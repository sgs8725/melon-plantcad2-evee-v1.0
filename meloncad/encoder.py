# -*- coding: utf-8 -*-
"""
meloncad/encoder.py
====================
甜瓜基因组大模型编码器 —— PlantCAD2 与 EVEE 之间的桥接层。

职责：
  1. 加载 PlantCAD2 预训练权重（基础模型层，全程冻结）。
  2. 对参考/变异等位序列提取逐位置隐藏状态，并利用 PlantCAD2 的
     反向互补等变性（RC-equivariance），从**单次前向**同时拿到正义链 /
     反义链两套表征。
  3. 按 EVEE 的「余弦发散 Top-K 选点」策略，挑出变异与参考表征差异最大的
     K 个位置，组装为 EVEE 协方差探针要求的 [2, 2, K, d_model] 激活张量。
  4. 额外提供 PlantCAD2 原生的「零样本掩码似然比（LLR）」打分，作为协方差
     探针之外的互补效应预测路径。

与 PlantCAD2（PlantCaduceus / Caduceus + Mamba2）官方接口的对接点：
  - 模型类：通过 transformers 的 ``AutoModelForMaskedLM.from_pretrained(
    model_path, trust_remote_code=True)`` 加载（PlantCAD2 为掩码语言模型，
    custom_code，与 PlantGFM 的因果语言模型不同）。
  - 分词器：``AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)``，
    单核苷酸字符级分词，**直接传入序列字符串**（无需像 PlantGFM 那样空格分隔）。
  - 前向：``model(input_ids=..., output_hidden_states=True)``，
    ``outputs.hidden_states[-1]`` 形状 ``[B, L, 2*hidden_size]``——
    前 hidden_size 通道为正义链表征，后 hidden_size 通道为反义链（沿 L 反向）
    表征。PlantCAD2 README 的标准用法即对二者取平均得到 RC 不变嵌入。

物种说明：甜瓜（Cucumis melo，葫芦科 Cucurbitaceae，2n=2x=24，12 条染色体，
基因组约 375–450 Mb、转座子约占 20%）与黄瓜、西瓜、南瓜同属葫芦科，是研究果实
成熟、性别决定与韧皮部生理的模式作物。PlantCAD2 在 65 个被子植物基因组上预训练，
覆盖多个双子叶分支，跨物种 zero-shot 能力扎实，因此直接复用其权重作为甜瓜基因组
大模型骨干，无需重新预训练。本编码器架构与黄瓜 / 辣椒版本同构，差异集中在下游
注释面板与目标性状（侧重糖度/蔗糖积累、香气酯类、乙烯成熟与耐贮、果肉色、性别
决定、葫芦科抗病通路）。
"""
from __future__ import annotations

import torch

# DNA 互补表，用于反义链（仅在 LLR / 设计等少数路径需要手动 revcomp）
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    """返回 DNA 序列的反向互补。"""
    return seq.translate(_COMP)[::-1]


class MelonCADEncoder:
    """封装 PlantCAD2 的甜瓜基因组编码器。

    Parameters
    ----------
    model_path : str
        PlantCAD2 权重目录或 HuggingFace 仓库名。默认使用 PlantCAD2-Small
        （l24, d768）。可换 Medium（l48, d1024）/ Large（l48, d1536）。
        由 scripts/download_data.py 下载到 assets/plantcad2/。
    layer : int
        取第几层 hidden_states。索引 0 为 embedding 输出，末位为最后一层。
        EVEE 论文在 Evo2 上做层扫描后取中后层最佳；对 PlantCAD2 默认取
        最后一层（-1）。
    device : str
    dtype : torch.dtype
    """

    def __init__(
        self,
        model_path: str = "assets/plantcad2",
        layer: int = -1,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.layer = layer

        # PlantCAD2 为 custom_code 模型，需 trust_remote_code
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype
        )
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():           # 冻结基础模型层
            p.requires_grad_(False)

        # PlantCAD2 输出隐藏维 = 2 * hidden_size（正义 + 反义两套通道）；
        # 单链有效维度 d_model = hidden_size。
        self.d_model: int = int(self.model.config.d_model)

    # ------------------------------------------------------------------ #
    # 工具：把 DNA 序列编码为 input_ids
    # ------------------------------------------------------------------ #
    def _encode_ids(self, sequence: str) -> torch.Tensor:
        ids = self.tokenizer.encode_plus(
            sequence.strip().upper(),
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        return ids.to(self.device).long()                  # [1, L]

    # ------------------------------------------------------------------ #
    # 基础前向：单条序列 -> (正义链嵌入, 反义链嵌入)，均对齐到正义链坐标
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _embed_dual(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        """对单条 DNA 序列做一次前向，返回 (fwd, rev) 两套逐位置嵌入。

        利用 PlantCAD2 的 RC 等变性：hidden_states[layer] 形状 [1, L, 2d]，
        前 d 通道为正义链表征，后 d 通道为反义链表征（沿 L 反向存放），
        本函数把反义链沿 L 翻转回正义链坐标，便于与正义链共用同一组 Top-K
        索引、并供协方差探针做「正义/反义」双向池化。
        """
        ids = self._encode_ids(sequence)
        out = self.model(input_ids=ids, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[self.layer].squeeze(0).float()   # [L, 2d]
        d = hs.shape[-1] // 2
        fwd = hs[:, :d]                              # 正义链 [L, d]
        rev = hs[:, d:].flip(0)                      # 反义链翻回正义链坐标 [L, d]
        return fwd, rev

    # ------------------------------------------------------------------ #
    # EVEE 风格：变异 -> [2, 2, K, d_model] 激活张量
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_variant(
        self,
        ref_seq: str,
        alt_seq: str,
        topk: int = 256,
    ) -> torch.Tensor:
        """对一个变异，组装 EVEE 协方差探针所需的激活张量 [2, 2, K, d_model]。

        ====  ====  ===========================================
        dim   size  含义
        ====  ====  ===========================================
        0     2     方向：正义链 forward / 反义链 backward
        1     2     视图：变异序列 variant / 参考序列 reference
        2     K     由余弦发散挑出的 Top-K 位置
        3     d     PlantCAD2 单链隐藏维 hidden_size
        ====  ====  ===========================================

        与 PlantGFM 版本相比：PlantCAD2 的 RC 等变性让一次前向即同时得到
        正义/反义两套表征，因此每个变异仅需 **2 次前向**（ref / alt），
        而非 4 次。维度语义与 EVEE demo notebook 完全一致。

        要求 ref_seq 与 alt_seq 等长（SNV）；indel 由 data/variant_dataset.py
        以参考侧翼补齐为等长后传入。
        """
        fwd_ref, bwd_ref = self._embed_dual(ref_seq)
        fwd_alt, bwd_alt = self._embed_dual(alt_seq)

        L = min(fwd_ref.shape[0], fwd_alt.shape[0],
                bwd_ref.shape[0], bwd_alt.shape[0])
        fwd_ref, fwd_alt = fwd_ref[:L], fwd_alt[:L]
        bwd_ref, bwd_alt = bwd_ref[:L], bwd_alt[:L]

        # --- 余弦发散 Top-K 选点（正义链上计算，两链共用同一组索引）---
        cos = torch.nn.functional.cosine_similarity(fwd_alt, fwd_ref, dim=-1)  # [L]
        k = min(topk, L)
        idx = torch.topk(-cos, k=k).indices.sort().values                     # [k]

        def _sel(t: torch.Tensor) -> torch.Tensor:
            sel = t[idx]                             # [k, d]
            if k < topk:                             # 不足 K 时 pad（仅极短序列）
                pad = torch.zeros(topk - k, t.shape[-1])
                sel = torch.cat([sel, pad], dim=0)
            return sel

        # 组装 [2, 2, K, d]：dim0 方向, dim1 视图(variant, reference)
        fwd = torch.stack([_sel(fwd_alt), _sel(fwd_ref)], dim=0)   # [2, K, d]
        bwd = torch.stack([_sel(bwd_alt), _sel(bwd_ref)], dim=0)   # [2, K, d]
        return torch.stack([fwd, bwd], dim=0)                       # [2, 2, K, d]

    @torch.no_grad()
    def encode_reference(self, seq: str) -> torch.Tensor:
        """供注释探针使用：返回序列的 RC 不变逐位置嵌入 [L, d_model]。

        采用 PlantCAD2 README 推荐做法：对正义链与反义链表征取平均，
        得到反向互补不变的逐位置表征。
        """
        fwd, rev = self._embed_dual(seq)
        return (fwd + rev) / 2.0                     # [L, d]

    # ------------------------------------------------------------------ #
    # PlantCAD2 原生：零样本掩码似然比（LLR）变异效应打分
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score_variant_llr(self, ref_seq: str, alt_seq: str, var_index: int) -> float:
        """对一个 SNV，返回零样本掩码似然比 logP(alt) - logP(ref)。

        这是 PlantCAD/PlantCAD2 的标志性能力：把变异位点替换为 [MASK]，
        让掩码语言模型预测该位置的核苷酸分布，比较参考/变异等位的对数概率。
        分数显著为负通常意味着该突变破坏了进化上保守的位点（可能有害）。
        与训练好的协方差探针互补，可在无标签时直接给出方向性参考。

        Parameters
        ----------
        ref_seq, alt_seq : str
            等长的参考 / 变异窗口序列（SNV）。
        var_index : int
            变异在窗口序列中的 0-based 位置（不含分词器可能添加的特殊 token）。
        """
        ref_base = ref_seq[var_index].upper()
        alt_base = alt_seq[var_index].upper()

        ids = self._encode_ids(ref_seq)[0].clone()   # [L]
        # 定位变异位点对应的 token 下标：通过比对 ref 序列各碱基 token 与原始位置。
        # 字符级分词下，序列 token 与碱基一一对应，但可能有前导特殊 token。
        offset = self._char_token_offset(ref_seq, ids)
        pos = offset + var_index
        if not (0 <= pos < ids.shape[0]):
            return float("nan")

        mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            return float("nan")
        ids[pos] = mask_id

        out = self.model(input_ids=ids.unsqueeze(0).to(self.device), return_dict=True)
        logits = out.logits[0, pos].float()          # [vocab]
        logp = torch.log_softmax(logits, dim=-1)

        ref_id = self.tokenizer.convert_tokens_to_ids(ref_base)
        alt_id = self.tokenizer.convert_tokens_to_ids(alt_base)
        if ref_id is None or alt_id is None:
            return float("nan")
        return float(logp[alt_id] - logp[ref_id])

    def _char_token_offset(self, seq: str, ids: torch.Tensor) -> int:
        """估计第一个核苷酸 token 在 input_ids 中的下标（处理前导特殊 token）。"""
        # 取序列首碱基对应的 token id，从前向后找第一个匹配位置。
        first_id = self.tokenizer.convert_tokens_to_ids(seq[0].upper())
        ids_list = ids.tolist()
        if first_id in ids_list:
            return ids_list.index(first_id)
        # 退化：长度差即偏移
        return max(0, ids.shape[0] - len(seq))

    # ------------------------------------------------------------------ #
    # de novo 调控元件设计：基于掩码语言模型的迭代 Gibbs 重采样
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def design_by_gibbs(
        self,
        length: int = 400,
        template: str | None = None,
        n_iters: int = 200,
        mask_frac: float = 0.1,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
        seed: int = 0,
    ) -> list[str]:
        """用 PlantCAD2（掩码语言模型）做 de novo 序列设计。

        PlantCAD2 是掩码语言模型，没有 PlantGFM 那样的自回归 generate；
        正确的生成方式是「迭代掩码重采样 / Gibbs 采样」：从模板（或随机）
        序列出发，每轮随机掩掉一部分位置，由 MLM 预测分布重采样填回，
        多轮后收敛到模型偏好的、符合甜瓜序列语法的候选元件。

        Parameters
        ----------
        length : int            生成序列长度（template 为空时使用）。
        template : str | None   起始模板（如已知启动子骨架），None 则随机起始。
        n_iters : int           Gibbs 迭代轮数。
        mask_frac : float       每轮掩码比例。
        temperature : float     采样温度。
        num_return_sequences : int
        seed : int
        """
        import random
        rng = random.Random(seed)
        bases = "ACGT"
        results: list[str] = []

        # 允许采样的核苷酸 token id
        allowed_tokens = [b for b in bases]
        allowed_ids = [self.tokenizer.convert_tokens_to_ids(b) for b in allowed_tokens]
        mask_id = self.tokenizer.mask_token_id

        for s in range(num_return_sequences):
            seq = list(template.upper()) if template else \
                [rng.choice(bases) for _ in range(length)]
            ids = self._encode_ids("".join(seq))[0].clone()  # [L]
            offset = self._char_token_offset("".join(seq), ids)
            n_pos = len(seq)

            for _ in range(n_iters):
                # 随机选一批位置掩码
                n_mask = max(1, int(mask_frac * n_pos))
                positions = rng.sample(range(n_pos), n_mask)
                work = ids.clone()
                for p in positions:
                    work[offset + p] = mask_id
                out = self.model(
                    input_ids=work.unsqueeze(0).to(self.device), return_dict=True
                )
                logits = out.logits[0].float()       # [L, vocab]
                for p in positions:
                    lp = logits[offset + p] / max(temperature, 1e-6)
                    # 仅在 A/C/G/T 之间采样
                    sub = torch.tensor([lp[i] for i in allowed_ids])
                    probs = torch.softmax(sub, dim=-1)
                    choice = int(torch.multinomial(probs, 1))
                    ids[offset + p] = allowed_ids[choice]
                    seq[p] = allowed_tokens[choice]
            results.append("".join(seq))
        return results


# ---------------------------------------------------------------------- #
# 离线 / CI 用的轻量 mock 编码器：无需 mamba-ssm / CUDA / 下载权重
# ---------------------------------------------------------------------- #
def build_mock_encoder(d_model: int = 768, vocab_size: int = 16) -> "MelonCADEncoder":
    """构造一个**随机初始化**的小型 RC 等变掩码 LM stand-in，用于无权重冒烟测试。

    真实 PlantCAD2（Caduceus + Mamba2）依赖 mamba-ssm / causal-conv1d 与
    CUDA 内核，难以在纯 CPU 跑通；本 mock 用纯 PyTorch 复刻 PlantCAD2 对外的
    最小接口（hidden_states[-1] 形状 [B, L, 2d]，logits 形状 [B, L, vocab]，
    字符级分词 + [MASK]），从而在 CPU 上验证整条流水线结构。

    正式使用请走 ``MelonCADEncoder(model_path="assets/plantcad2")`` 加载
    下载好的真实权重。
    """
    from meloncad.mock_model import MockCaduceusMLM, CharTokenizer

    enc = MelonCADEncoder.__new__(MelonCADEncoder)
    enc.tokenizer = CharTokenizer()
    enc.model = MockCaduceusMLM(
        vocab_size=enc.tokenizer.vocab_size, d_model=d_model
    ).eval()
    for p in enc.model.parameters():
        p.requires_grad_(False)
    enc.device = "cpu"
    enc.dtype = torch.float32
    enc.layer = -1
    enc.d_model = d_model
    return enc
