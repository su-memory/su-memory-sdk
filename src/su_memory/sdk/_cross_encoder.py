"""
su-memory v3.5.2 — Cross-Encoder Reranker

基于 cross-encoder/ms-marco-MiniLM-L-6-v2 的精排层。
对 RRF 融合后的 top-N 候选做句子对打分，精度高于 Bi-Encoder。

用法:
    from su_memory.sdk._cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(
        query="自然语言处理研究方向",
        candidates=[{"memory_id": "1", "content": "张明研究NLP"}, ...],
        top_k=5
    )
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 推荐模型（按体积和性能排序）
RECOMMENDED_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",   # 80MB, 最快
    "cross-encoder/ms-marco-MiniLM-L-12-v2",  # 400MB, 更准
    "cross-encoder/ms-marco-TinyBERT-L-2-v2", # 30MB, 极速
]


class CrossEncoderReranker:
    """
    Cross-Encoder 精排器。

    特性:
    - 懒加载模型（首次 rerank() 调用时才下载/加载）
    - sentence-transformers 不可用时静默降级（返回原始顺序）
    - 仅对 top-N 候选做精排（N ≤ 20，控制延迟）
    - 支持手动开关

    Example:
        >>> reranker = CrossEncoderReranker()
        >>> reranker.rerank("天气如何", [
        ...     {"memory_id": "1", "content": "今天天气很好", "score": 0.8},
        ...     {"memory_id": "2", "content": "股票涨了", "score": 0.6},
        ... ], top_k=2)
        [{"memory_id": "1", "content": "今天天气很好", "score": 0.95, "ce_score": 0.95}, ...]
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: HuggingFace cross-encoder 模型名。
                        None 时强制降级（不精排）。
        """
        self.model_name = model_name
        self._model = None
        self._available = None  # True/False/None（未检测）

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """懒加载模型"""
        if self._available is not None:
            return

        if self.model_name is None:
            self._available = False
            return

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._available = True
            logger.info(f"[CrossEncoderReranker] 模型已加载: {self.model_name}")
        except ImportError:
            logger.warning(
                "[CrossEncoderReranker] sentence-transformers 未安装，"
                "精排降级为直通。pip install sentence-transformers"
            )
            self._available = False
        except Exception as e:
            logger.warning(f"[CrossEncoderReranker] 模型加载失败: {e}")
            self._available = False

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        max_candidates: int = 20,
    ) -> list[dict]:
        """
        Cross-Encoder 精排。

        Args:
            query: 查询文本
            candidates: 候选列表，每项需含 {"content": str, ...}
            top_k: 最终返回数量
            max_candidates: 精排候选上限（过多会拖慢性能）

        Returns:
            精排后的结果列表，每项附加 ce_score 字段
        """
        self._ensure_model()

        if not self._available or self._model is None or not candidates:
            return candidates[:top_k]

        # 限制候选数量
        subset = candidates[:max_candidates]

        try:
            # 构建 (query, doc) 对
            pairs = [(query, c.get("content", "")) for c in subset]

            # 预测得分
            scores = self._model.predict(pairs)

            # 附加 ce_score
            for i, c in enumerate(subset):
                ce_score = float(scores[i]) if i < len(scores) else 0.0
                c["ce_score"] = ce_score
                # 融合: 原分 30% + CE 分 70%
                orig = c.get("score", 0.5)
                c["score"] = orig * 0.3 + ce_score * 0.7

            # 排序
            subset.sort(key=lambda x: x.get("score", 0), reverse=True)
            return subset[:top_k]

        except Exception as e:
            logger.warning(f"[CrossEncoderReranker] 精排失败: {e}")
            return candidates[:top_k]


def create_cross_encoder_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> CrossEncoderReranker:
    """工厂函数"""
    return CrossEncoderReranker(model_name=model_name)
