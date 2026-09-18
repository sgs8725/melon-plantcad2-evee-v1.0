# -*- coding: utf-8 -*-
"""
probes/covariance_probe.py
==========================
EVEE 协方差探针 —— 移植自 evee-manuscript 的
notebooks/demo_probe_inference.ipynb（CovarianceProbe 类）。

与原始 EVEE 代码的差异**仅有一处**：
  d_model 默认值 4096（Evo2-7B block 27） → 768（PlantCAD2-Small 的 hidden_size）。
  换用 PlantCAD2-Medium / Large 时分别改为 1024 / 1536。
探针的三阶段结构（非约束投影 / 协方差池化 + Newton-Schulz 谱压缩 / 因式读出）
与算法逻辑保持逐行一致。

输入张量布局（由 meloncad/encoder.py:encode_variant 产出）：
    activations: [B, 2, 2, K, d_model]
        dim1 = 方向 (forward, backward)        ← PlantCAD2 单次前向的 RC 双通道
        dim2 = 视图 (variant, reference)
        dim3 = K 个 Top-K 发散位置
        dim4 = d_model = PlantCAD2 hidden_size
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


def newton_schulz_sqrtm(matrix: Tensor, n_iters: int = 3) -> Tensor:
    """矩阵平方根的 Newton-Schulz 耦合迭代（EVEE 原始实现）。

    先用 Frobenius 范数归一把特征值压到 (0, 1]，迭代后再缩放回去。
    用于对协方差矩阵做"谱压缩"，抑制主导特征值。
    """
    dtype = matrix.dtype
    m = matrix.float()
    d = m.size(-1)

    frob = m.flatten(-2).norm(dim=-1)[..., None, None]
    scale = frob.sqrt()
    normed = m / frob

    eye = torch.eye(d, device=m.device)
    y, z = normed, eye.expand_as(m).clone()
    for _ in range(n_iters):
        t = 3 * eye - z @ y
        y = y @ t / 2
        z = t @ z / 2

    return (y * scale).to(dtype)


class CovarianceProbe(nn.Module):
    """双向基因组激活的协方差探针（EVEE）。

    把变长激活差经"非约束 left/right 投影"池化为交叉协方差矩阵，
    施加 Newton-Schulz 谱压缩，再用因式读出做分类/回归。

    Parameters
    ----------
    d_model : int
        基础模型隐藏维。**PlantCAD2-Small 为 768**（区别于 Evo2 的 4096、
        PlantGFM 的 256）；Medium=1024，Large=1536。
    d_hidden : int
        投影后维度（协方差矩阵边长），EVEE 默认 64。
    d_probe : int
        因式读出的中间维，EVEE 默认 128。
    n_outputs : int
        输出维数。2 = 二分类(有利/不利)；n = 多性状（每个性状一维 logit）。
    n_sqrtm_iters : int
        Newton-Schulz 迭代步数。
    eps : float
        协方差对角正则系数。
    """

    def __init__(
        self,
        d_model: int = 768,
        d_hidden: int = 64,
        d_probe: int = 128,
        n_outputs: int = 2,
        n_sqrtm_iters: int = 3,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.n_sqrtm_iters = n_sqrtm_iters
        self.eps = eps

        # 阶段 1：非约束（left/right 不共享）投影，每个方向一对
        self.proj_left_fwd = nn.Linear(d_model, d_hidden)
        self.proj_right_fwd = nn.Linear(d_model, d_hidden)
        self.proj_left_bwd = nn.Linear(d_model, d_hidden)
        self.proj_right_bwd = nn.Linear(d_model, d_hidden)
        self.register_buffer("_eye", torch.eye(d_hidden))

        # 阶段 3：因式读出
        self.head_left = nn.Linear(d_hidden, d_probe, bias=False)
        self.head_right = nn.Linear(d_hidden, d_probe, bias=False)
        self.out = nn.Linear(d_probe, n_outputs)

    # 阶段 2：协方差池化 + 谱压缩
    def embedding(self, activations: Tensor) -> Tensor:
        """[B, 2, 2, K, d] → [B, d_hidden, d_hidden] 压缩协方差矩阵。"""
        diff_fwd = activations[:, 0, 0] - activations[:, 0, 1]   # 变异 - 参考（正义）
        diff_bwd = activations[:, 1, 0] - activations[:, 1, 1]   # 变异 - 参考（反义）

        left_fwd = self.proj_left_fwd(diff_fwd).float()
        right_fwd = self.proj_right_fwd(diff_fwd).float()
        left_bwd = self.proj_left_bwd(diff_bwd).float()
        right_bwd = self.proj_right_bwd(diff_bwd).float()

        K = left_fwd.shape[1]
        cov = (
            torch.bmm(left_fwd.transpose(1, 2), right_fwd) / K
            + torch.bmm(left_bwd.transpose(1, 2), right_bwd) / K
            + self.eps * self._eye
        )
        return newton_schulz_sqrtm(cov, self.n_sqrtm_iters)

    def forward(self, activations: Tensor) -> Tensor:
        """[B, 2, 2, K, d] → [B, n_outputs] logits。"""
        cov = self.embedding(activations)
        feat = torch.einsum(
            "blr,hl,hr->bh", cov,
            self.head_left.weight, self.head_right.weight,
        )
        return self.out(feat)

    def predict_proba(self, activations: Tensor) -> Tensor:
        """二分类：[B, 2, 2, K, d] → [B] P(不利)。"""
        return torch.softmax(self(activations), dim=-1)[:, 1]

    def predict_traits(self, activations: Tensor) -> Tensor:
        """多性状回归/打分：[B, 2, 2, K, d] → [B, n_outputs]，经 sigmoid。"""
        return torch.sigmoid(self(activations))
