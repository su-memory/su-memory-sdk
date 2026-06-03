"""
su-memory v3.2.0 — Semantic Reranker Layer

可选依赖 sentence-transformers。不可用时静默降级为直通（不改变 TF-IDF 排序）。
双路检索：TF-IDF 粗排（top-N）→ embedding 精排（top-K）。

用法:
    from su_memory.sdk._semantic_reranker import SemanticReranker

    reranker = SemanticReranker()
    reranked = reranker.rerank(
        query="自然语言处理研究方向",
        candidates=["张明毕业于清华计算机系", "张明研究自然语言处理与深度学习"],
        top_k=2
    )
"""

from __future__ import annotations

import heapq
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# v3.2.0: 推荐模型列表（按体积排序，优先用最小的）
RECOMMENDED_MODELS = [
    "all-MiniLM-L6-v2",            # 80MB, 384-dim, 最快
    "all-mpnet-base-v2",           # 420MB, 768-dim, 最准
    "paraphrase-multilingual-MiniLM-L12-v2",  # 470MB, 384-dim, 多语言
]


class SemanticReranker:
    """
    Embedding 语义重排器。

    特性:
    - 懒加载模型（首次 rerank() 调用时才下载/加载）
    - sentence-transformers 不可用时静默降级（返回原始顺序）
    - 支持自定义模型名
    - 批量候选编码提升吞吐

    Example:
        >>> reranker = SemanticReranker()
        >>> reranker.rerank("天气如何", ["今天天气很好", "股票涨了"], top_k=1)
        [(0, 0.85)]
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: sentence-transformers 模型名。
                        None 时强制降级（仅 TF-IDF）。
        """
        self.model_name = model_name
        self._model = None        # SentenceTransformer 实例
        self._available = None    # True/False/None（未检测）
        self._encode_cache: dict = {}  # 编码缓存（少量查询场景）
        self._cache_hits = 0
        self._cache_misses = 0

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """懒加载模型（仅首次调用时执行）"""
        if self._available is not None:
            return  # 已检测过

        if self.model_name is None:
            self._available = False
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            # Warm-up: encode a tiny sentence to load model into memory
            _ = self._model.encode(["warmup"], show_progress_bar=False)
            self._available = True
            logger.info(
                "SemanticReranker: model '%s' loaded successfully",
                self.model_name,
            )
        except ImportError:
            self._available = False
            logger.warning(
                "SemanticReranker: sentence-transformers not installed, "
                "falling back to TF-IDF only. Install with: "
                "pip install sentence-transformers"
            )
        except Exception as exc:
            self._available = False
            logger.warning(
                "SemanticReranker: failed to load model '%s': %s",
                self.model_name, exc,
            )

    @property
    def is_available(self) -> bool:
        """模型是否可用"""
        self._ensure_model()
        return self._available is True

    # ------------------------------------------------------------------
    # Core reranking
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        对候选列表进行语义重排。

        Args:
            query: 查询文本
            candidates: 候选文本列表（TF-IDF 粗排后的 top-N）
            top_k: 返回数量

        Returns:
            [(candidate_index, cosine_similarity), ...] 按相似度降序。
            模型不可用时返回 [(0, 1.0), (1, 0.9), ...] 模拟递减分数。
        """
        if not candidates:
            return []

        self._ensure_model()

        # 降级路径：返回原始顺序
        if not self._available or self._model is None:
            return self._fallback_rerank(len(candidates), top_k)

        # 缓存检查
        cache_key = (query, tuple(candidates), top_k)
        if cache_key in self._encode_cache:
            self._cache_hits += 1
            return self._encode_cache[cache_key]

        self._cache_misses += 1
        result = self._semantic_rerank(query, candidates, top_k)

        # 缓存（限制大小）
        if len(self._encode_cache) < 256:
            self._encode_cache[cache_key] = result

        return result

    def _semantic_rerank(
        self,
        query: str,
        candidates: list[str],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """实际语义编码+余弦相似度计算"""
        import numpy as np

        # 编码查询和候选
        query_emb = self._model.encode([query], show_progress_bar=False)[0]
        cand_embs = self._model.encode(candidates, show_progress_bar=False)

        # 计算余弦相似度
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            return self._fallback_rerank(len(candidates), top_k)

        scores: list[float] = []
        for _i, c_emb in enumerate(cand_embs):
            c_norm = np.linalg.norm(c_emb)
            if c_norm == 0:
                scores.append(0.0)
            else:
                sim = float(np.dot(query_emb, c_emb) / (query_norm * c_norm))
                scores.append(max(0.0, min(1.0, sim)))

        # heap top-K
        top = heapq.nlargest(min(top_k, len(candidates)), enumerate(scores), key=lambda x: x[1])

        return [(idx, round(score, 4)) for idx, score in top]

    def _fallback_rerank(
        self, n_candidates: int, top_k: int
    ) -> list[tuple[int, float]]:
        """降级模式：保持 TF-IDF 原始顺序，分数递减"""
        k = min(top_k, n_candidates)
        return [(i, round(1.0 - i * 0.1, 2)) for i in range(k)]

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    def rerank_batch(
        self,
        queries: list[str],
        candidates_list: list[list[str]],
        top_k: int = 5,
    ) -> list[list[tuple[int, float]]]:
        """
        批量重排（多个查询共享模型编码上下文）。

        Args:
            queries: 查询文本列表
            candidates_list: 对应的候选列表（长度必须与 queries 一致）
            top_k: 每查询返回数量

        Returns:
            [[(idx, score), ...], ...] 每个查询的重排结果
        """
        if len(queries) != len(candidates_list):
            raise ValueError(
                f"Length mismatch: queries={len(queries)} "
                f"vs candidates={len(candidates_list)}"
            )

        return [self.rerank(q, c, top_k) for q, c in zip(queries, candidates_list, strict=False)]

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """获取重排器状态"""
        self._ensure_model()
        return {
            "available": self.is_available,
            "model_name": self.model_name,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._encode_cache),
        }

    def clear_cache(self) -> None:
        """清除编码缓存"""
        self._encode_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
