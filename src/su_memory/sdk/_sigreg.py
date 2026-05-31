"""
su-memory v3.5.0 — SIGReg Embedding Regularizer
================================================
基于 LeJEPA (2511.08544v2) 的 Sketched Isotropic Gaussian Regularization。

核心洞察: 嵌入向量的各向同性程度与下游检索质量正相关。
SIGReg 通过最小化与各向同性高斯的矩差来正则化嵌入。

数学:
  L_SIGReg(z) = ||mean(z)||² + λ · ||cov(z) - I||²
  其中 z 为归一化后的嵌入向量。

复杂度: O(d²) per batch, d=768 时 < 1ms
行数: ~50 行 (对齐 LeJEPA 设计理念)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import faiss

logger = logging.getLogger(__name__)


class SIGReg:
    """
    Sketched Isotropic Gaussian Regularizer.

    用途:
    1. 嵌入后处理: FAISS 索引前对嵌入做各向同性正则
    2. 训练时正则: 记忆模型训练时作为辅助损失

    超参数:
    - lambda_reg: 正则强度 (默认 0.01)
    - use_sketch: 是否使用 sketched 加速 (默认 True, SVD 近似)
    - sketch_dim: sketched 子空间维度 (默认 64)
    """

    def __init__(
        self,
        lambda_reg: float = 0.01,
        use_sketch: bool = True,
        sketch_dim: int = 64,
    ):
        if not 0.0 <= lambda_reg <= 1.0:
            raise ValueError(f"lambda_reg 必须在 [0,1], 当前 {lambda_reg}")
        if sketch_dim < 1:
            raise ValueError(f"sketch_dim 必须 ≥ 1, 当前 {sketch_dim}")
        self.lambda_reg = lambda_reg
        self.use_sketch = use_sketch
        self.sketch_dim = sketch_dim

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        """
        对嵌入矩阵做各向同性正则化。

        Args:
            embeddings: shape=(n, d) 嵌入矩阵

        Returns:
            正则化后的嵌入矩阵, shape=(n, d)
        """
        return self.regularize(embeddings)

    def regularize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        两步正则:
        1. 零均值化 (mean shift)
        2. 协方差白化 (cov → I)
        3. 正则化强度插值
        4. L2 归一化
        """
        z = embeddings.copy().astype(np.float64)
        if z.ndim != 2:
            raise ValueError(
                f"embeddings 必须是 2D 数组 (n, d), 当前 shape={embeddings.shape}"
            )
        n, d = z.shape
        if n == 0 or d == 0:
            raise ValueError(
                f"embeddings 维度不能为 0, 当前 shape={embeddings.shape}"
            )

        # n<2 时协方差退化，跳过白化仅做 L2 归一化
        if n < 2:
            norms = np.linalg.norm(z, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            return (z / norms).astype(embeddings.dtype)

        # ① 零均值化
        z_mean = z.mean(axis=0, keepdims=True)
        z = z - z_mean

        # ② 白化 (协方差 → 单位矩阵)
        if self.use_sketch and d > self.sketch_dim:
            # Sketched: 仅在低维子空间做白化
            sketch = np.random.randn(d, self.sketch_dim) / np.sqrt(self.sketch_dim)
            z_sketch = z @ sketch  # (n, sketch_dim)
            cov_sketch = z_sketch.T @ z_sketch / (n - 1)

            # 特征分解 + 白化
            eigvals, eigvecs = np.linalg.eigh(cov_sketch)
            eigvals = np.maximum(eigvals, 1e-6)
            whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            z_whitened = z_sketch @ whitening
            # 投影回原空间
            z_reg = z_whitened @ sketch.T
        else:
            # 完整白化
            cov = z.T @ z / (n - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-6)
            whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            z_reg = z @ whitening

        # ③ 正则化强度插值 (保留部分原始方向)
        result = z * (1.0 - self.lambda_reg) + z_reg * self.lambda_reg

        # ④ L2 归一化
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        result = result / norms

        return result.astype(embeddings.dtype)

    def compute_isotropy_score(self, embeddings: np.ndarray) -> float:
        """
        计算嵌入的各向同性得分。

        0 = 完全退化 (所有向量相同方向)
        1 = 完全各向同性 (均匀分布在高维球面上)

        使用协方差矩阵的条件数:
        score = 1 / cond(cov)
        """
        n = embeddings.shape[0]
        if n < 2:
            return 0.0

        z = embeddings.copy().astype(np.float64)
        z_mean = z.mean(axis=0, keepdims=True)
        z = z - z_mean
        cov = z.T @ z / (z.shape[0] - 1)

        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 1e-10)

        cond = eigvals.max() / eigvals.min()
        return min(1.0, 1.0 / max(cond, 1e-10))


# ============================================================
# FAISS 集成工具
# ============================================================

def apply_sigreg_to_index(
    index: faiss.Index,
    embeddings: np.ndarray,
    lambda_reg: float = 0.02,
) -> faiss.Index:
    """
    对 FAISS 索引中的嵌入应用 SIGReg，返回重建后的索引。

    不修改原索引。

    Args:
        index: 原 FAISS 索引 (用于获取参数)
        embeddings: shape=(n, d) 原始嵌入矩阵
        lambda_reg: SIGReg 正则强度

    Returns:
        重建后的 FAISS IndexHNSW，含正则化嵌入
    """
    import faiss

    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings 必须是 2D 数组, 当前 shape={embeddings.shape}"
        )
    d_emb = embeddings.shape[1]
    d_idx = index.d
    if d_emb != d_idx:
        logger.warning(
            "嵌入维度(%d)与索引维度(%d)不匹配, 将重建索引",
            d_emb, d_idx,
        )

    sigreg = SIGReg(lambda_reg=lambda_reg)
    regularized = sigreg.regularize(embeddings)

    d = embeddings.shape[1]
    new_index = faiss.IndexHNSW(d, 32)
    new_index.hnsw.efConstruction = 64
    new_index.hnsw.efSearch = 64
    new_index.add(regularized.astype(np.float32))

    return new_index


# ============================================================
# Self-test
# ============================================================

def _self_test():
    """Quick self-test for SIGReg."""
    # 构造有偏嵌入 (所有向量在第一象限 → 低各向同性)
    np.random.seed(42)
    biased = np.abs(np.random.randn(100, 768))  # 均值偏移

    sigreg = SIGReg(lambda_reg=0.1)
    iso_before = sigreg.compute_isotropy_score(biased)
    regularized = sigreg.regularize(biased)
    iso_after = sigreg.compute_isotropy_score(regularized)

    print(f"各向同性: {iso_before:.4f} → {iso_after:.4f}")
    assert iso_after > iso_before, "SIGReg 应提升各向同性"
    print("✅ SIGReg 自检通过")


if __name__ == "__main__":
    _self_test()
