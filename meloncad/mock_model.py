# -*- coding: utf-8 -*-
"""
meloncad/mock_model.py
=======================
离线 / CI 专用的 PlantCAD2 stand-in。

真实 PlantCAD2 是 Caduceus（Mamba2）双向、反向互补等变的掩码语言模型，
其官方实现依赖 mamba-ssm / causal-conv1d 与 CUDA 内核，难以在纯 CPU 上运行。
本文件用**纯 PyTorch** 复刻 PlantCAD2 对外暴露的最小接口，使
``scripts/smoke_test.py`` 能在无 GPU、无下载的环境下验证整条流水线结构：

  - CharTokenizer：单核苷酸字符级分词，含 [MASK]，提供 encode_plus /
    convert_tokens_to_ids / mask_token_id / batch_decode。
  - MockCaduceusMLM：forward(output_hidden_states=True) 返回
      .hidden_states  —— 元组，末项形状 [B, L, 2*hidden_size]
                         （前 hidden_size 通道=正义链，后 hidden_size=反义链）
      .logits         —— [B, L, vocab_size]（掩码语言模型头）
    并带 .config.hidden_size。

**这不是 PlantCAD2 本体**，仅用于结构性冒烟测试；真实推理请用
``MelonCADEncoder(model_path="assets/plantcad2")`` 加载真实权重。
"""
from __future__ import annotations

import torch
from torch import nn


class _Config:
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size


class CharTokenizer:
    """单核苷酸字符级分词器（mock）。

    词表：特殊 token + A/C/G/T/N。与 PlantCAD2 真实 tokenizer 的差异不影响
    结构性测试——真实使用时由 AutoTokenizer 提供。
    """

    def __init__(self):
        specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
        bases = ["A", "C", "G", "T", "N"]
        self._itos = specials + bases
        self._stoi = {t: i for i, t in enumerate(self._itos)}
        self.mask_token_id = self._stoi["[MASK]"]
        self.pad_token_id = self._stoi["[PAD]"]
        self.unk_token_id = self._stoi["[UNK]"]

    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    def convert_tokens_to_ids(self, tok: str):
        return self._stoi.get(tok.upper(), self.unk_token_id)

    def encode_plus(self, sequence: str, return_tensors="pt",
                    return_attention_mask=False, return_token_type_ids=False,
                    **kwargs) -> dict:
        ids = [self.convert_tokens_to_ids(c) for c in sequence.strip().upper()]
        t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)   # [1, L]
        return {"input_ids": t}

    def batch_decode(self, ids, skip_special_tokens=True, **kwargs) -> list[str]:
        out = []
        for row in ids:
            chars = []
            for i in row.tolist():
                tok = self._itos[i] if 0 <= i < len(self._itos) else "[UNK]"
                if skip_special_tokens and tok.startswith("["):
                    continue
                chars.append(tok)
            out.append("".join(chars))
        return out


class _MockOutput:
    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = hidden_states


class MockCaduceusMLM(nn.Module):
    """随机初始化的小型双向掩码 LM，复刻 PlantCAD2 的对外接口（仅形状一致）。"""

    def __init__(self, vocab_size: int = 12, d_model: int = 768):
        super().__init__()
        self.config = _Config(hidden_size=d_model, vocab_size=vocab_size)
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        # 简单双向上下文混合（用两层一维卷积模拟，不追求真实建模能力）
        self.mix = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
        )
        # 正义链 / 反义链两套投影头（拼成 2d，模拟 RC 等变的双通道输出）
        self.proj_fwd = nn.Linear(d_model, d_model)
        self.proj_rev = nn.Linear(d_model, d_model)
        # 掩码语言模型头
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False,
                return_dict: bool = True, **kwargs):
        x = self.embed(input_ids)                    # [B, L, d]
        h = self.mix(x.transpose(1, 2)).transpose(1, 2)   # [B, L, d]
        fwd = self.proj_fwd(h)                        # [B, L, d]
        rev = self.proj_rev(h).flip(1)               # [B, L, d] 反义链（沿 L 反向存放）
        hidden_last = torch.cat([fwd, rev], dim=-1)  # [B, L, 2d]
        logits = self.lm_head(h)                     # [B, L, vocab]
        hidden_states = (
            torch.cat([x, x], dim=-1),               # 模拟 embedding 层输出 [B,L,2d]
            hidden_last,                             # 模拟最后一层 [B,L,2d]
        )
        return _MockOutput(logits=logits, hidden_states=hidden_states)
