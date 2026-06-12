"""
模型维度路由器 — ModelRouter

v3.6.0: 根据问题类型将 LongMemEval 的 LLM 调用路由到不同模型，
最大化各维度的准确率。

路由策略（基于实测数据）:
- single_session:     V4-Pro (+5.8pp vs chat)
- multi_session:      V4-Pro (+0.8pp, 边际收益低但无劣化)
- temporal_reasoning: deepseek-chat (V4-Pro -6.8pp)
- knowledge_update:   deepseek-chat (V4-Pro -19.3pp)
- abstain:            任一模型均可（匹配逻辑不依赖 LLM 答案）
- default:            V4-Pro（回退）

用法:
    from benchmarks._model_router import ModelRouter

    router = ModelRouter(
        default=reranker_v4,
        overrides={
            "temporal-reasoning": reranker_chat,
            "knowledge-update": reranker_chat,
        }
    )
    reranker = router.route("temporal-reasoning")
"""

from __future__ import annotations

from typing import Any


class ModelRouter:
    """按问题类型路由到最优 LLM 模型。"""

    def __init__(
        self,
        default: Any,  # LLMReranker — 默认模型
        overrides: dict[str, Any] | None = None,  # {question_type: LLMReranker}
    ):
        self._default = default
        self._overrides = overrides or {}

    def route(self, question_type: str) -> Any:
        """返回适合该问题类型的 LLMReranker。

        Args:
            question_type: 问题类型（single-session / multi-session /
                          temporal-reasoning / knowledge-update / abstain）

        Returns:
            LLMReranker 实例。
        """
        return self._overrides.get(question_type, self._default)

    def answer_from_context(
        self,
        question: str,
        context_chunks: list[str],
        question_type: str = "unknown",
    ) -> str:
        """路由到正确的模型并从上下文提取答案。"""
        reranker = self.route(question_type)
        if reranker is None:
            return ""
        return reranker.answer_from_context(question, context_chunks, question_type)

    @property
    def model(self) -> str:
        return f"ModelRouter(default={self._default.model})"

    def __getattr__(self, name: str) -> Any:
        """代理未定义属性到默认 reranker。"""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._default, name)

    def __repr__(self) -> str:
        if not hasattr(self, "_default"):
            return "ModelRouter(uninitialized)"
        override_info = ", ".join(
            f"{k}={v.model}" for k, v in self._overrides.items()
        )
        return f"ModelRouter(default={self._default.model}, overrides=[{override_info}])"
