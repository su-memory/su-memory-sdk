"""
su-memory 降级链 (FallbackChain)

实现 7 组件降级路径，确保核心功能在任何条件下可用。

降级矩阵总览:
| 组件     | 主路径           | 降级1          | 降级2                  | 降级3    |
|----------|-----------------|----------------|-----------------------|----------|
| 嵌入     | Ollama(bge-m3)  | MiniMax API    | sentence-transformers | TF-IDF   |
| 向量索引 | FAISS HNSW      | 线性(numpy)     | —                     | —        |
| 图谱     | MemoryGraph     | 纯向量检索       | —                     | —        |
| 时空     | SpacetimeIndex  | TemporalSystem  | —                     | —        |
| 存储     | Qdrant          | SQLite         | 内存 Dict             | —        |
| 能量推断 | LLM(≥85%)       | 关键词规则(≥60%) | 默认值                | —        |
| 会话     | SessionManager  | 内存 Session   | —                     | —        |

使用方式:
    from su_memory._sys.fallback import FallbackChain, FallbackLevel
    chain = FallbackChain()
    result = chain.try_embed("hello")  # 自动降级
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from su_memory.exceptions import ErrorCode, SuMemoryError

logger = logging.getLogger(__name__)


# =============================================================================
# FallbackLevel — 降级级别
# =============================================================================

class FallbackLevel(Enum):
    """降级级别"""
    PRIMARY = 0        # 主路径
    FALLBACK_1 = 1     # 降级1
    FALLBACK_2 = 2     # 降级2
    FALLBACK_3 = 3     # 降级3 (最低保障)
    GUARANTEED = 99    # 兜底


@dataclass
class FallbackStep:
    """降级链中的一步"""
    name: str
    level: FallbackLevel
    description: str
    callable: Callable
    expected_latency_ms: float = 0.0
    accuracy_relative: float = 1.0  # 相对于主路径的准确度


@dataclass
class FallbackResult:
    """降级执行结果"""
    success: bool
    result: Any = None
    level: FallbackLevel = FallbackLevel.GUARANTEED
    step_name: str = ""
    error: Exception | None = None
    attempts: int = 0


# =============================================================================
# FallbackChain — 通用降级链执行器
# =============================================================================

class FallbackChain:
    """通用降级链 — 按顺序尝试一系列操作，直到成功或全部失败"""

    def __init__(self, name: str = "generic"):
        self.name = name
        self._steps: list[FallbackStep] = []
        self._on_fallback: Callable | None = None
        self._stats: dict[str, int] = {"primary": 0, "fallback": 0, "guaranteed": 0, "failed": 0}

    def add_step(
        self,
        name: str,
        func: Callable,
        level: FallbackLevel = FallbackLevel.PRIMARY,
        description: str = "",
        expected_latency_ms: float = 0.0,
        accuracy_relative: float = 1.0,
    ) -> FallbackChain:
        """添加降级步骤"""
        self._steps.append(FallbackStep(
            name=name,
            level=level,
            description=description,
            callable=func,
            expected_latency_ms=expected_latency_ms,
            accuracy_relative=accuracy_relative,
        ))
        return self

    def on_fallback(self, callback: Callable) -> FallbackChain:
        """设置降级回调 — 每次降级时触发"""
        self._on_fallback = callback
        return self

    def try_execute(self, *args, **kwargs) -> FallbackResult:
        """按顺序尝试执行，直到成功或全部失败"""
        last_error = None
        for i, step in enumerate(self._steps):
            try:
                result = step.callable(*args, **kwargs)
                if step.level == FallbackLevel.PRIMARY:
                    self._stats["primary"] += 1
                elif step.level in (FallbackLevel.FALLBACK_1, FallbackLevel.FALLBACK_2, FallbackLevel.FALLBACK_3):
                    self._stats["fallback"] += 1
                else:
                    self._stats["guaranteed"] += 1

                return FallbackResult(
                    success=True,
                    result=result,
                    level=step.level,
                    step_name=step.name,
                    attempts=i + 1,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[FallbackChain:{self.name}] 步骤 '{step.name}' "
                    f"(级别: {step.level.name}) 失败: {e}"
                )
                if self._on_fallback:
                    self._on_fallback(step, e)

        self._stats["failed"] += 1
        return FallbackResult(
            success=False,
            error=last_error,
            attempts=len(self._steps),
        )

    def get_stats(self) -> dict[str, int]:
        """获取降级统计"""
        return dict(self._stats)

    @property
    def steps(self) -> list[FallbackStep]:
        return list(self._steps)


# =============================================================================
# 预定义降级链
# =============================================================================

def create_embedding_fallback_chain(
    ollama_func: Callable | None = None,
    minimax_func: Callable | None = None,
    sentence_transformer_func: Callable | None = None,
    tfidf_func: Callable | None = None,
) -> FallbackChain:
    """创建嵌入降级链: Ollama → MiniMax → sentence-transformers → TF-IDF"""
    chain = FallbackChain("embedding")

    if ollama_func:
        chain.add_step("Ollama(bge-m3)", ollama_func, FallbackLevel.PRIMARY,
                       "本地 Ollama 嵌入服务，bge-m3 模型", expected_latency_ms=50, accuracy_relative=1.0)
    if minimax_func:
        chain.add_step("MiniMax", minimax_func, FallbackLevel.FALLBACK_1,
                       "云端 MiniMax API 嵌入服务", expected_latency_ms=200, accuracy_relative=0.95)
    if sentence_transformer_func:
        chain.add_step("sentence-transformers", sentence_transformer_func, FallbackLevel.FALLBACK_2,
                       "本地 sentence-transformers (all-MiniLM-L6-v2)", expected_latency_ms=100, accuracy_relative=0.85)
    if tfidf_func:
        chain.add_step("TF-IDF", tfidf_func, FallbackLevel.FALLBACK_3,
                       "纯统计 TF-IDF 向量化 (无依赖)", expected_latency_ms=5, accuracy_relative=0.60)

    return chain


def create_storage_fallback_chain(
    qdrant_func: Callable | None = None,
    sqlite_func: Callable | None = None,
    memory_func: Callable | None = None,
) -> FallbackChain:
    """创建存储降级链: Qdrant → SQLite → 内存 Dict"""
    chain = FallbackChain("storage")

    if qdrant_func:
        chain.add_step("Qdrant", qdrant_func, FallbackLevel.PRIMARY,
                       "云端 Qdrant 向量数据库", expected_latency_ms=100, accuracy_relative=1.0)
    if sqlite_func:
        chain.add_step("SQLite", sqlite_func, FallbackLevel.FALLBACK_1,
                       "本地 SQLite 存储 (WAL模式)", expected_latency_ms=10, accuracy_relative=1.0)
    if memory_func:
        chain.add_step("内存Dict", memory_func, FallbackLevel.FALLBACK_2,
                       "内存字典 (不持久化，进程退出丢失)", expected_latency_ms=1, accuracy_relative=1.0)

    return chain


def create_prediction_fallback_chain(
    llm_func: Callable | None = None,
    rule_func: Callable | None = None,
    default_func: Callable | None = None,
) -> FallbackChain:
    """创建能量推断降级链: LLM(≥85%) → 关键词规则(≥60%) → 默认值"""
    chain = FallbackChain("prediction")

    if llm_func:
        chain.add_step("LLM推断", llm_func, FallbackLevel.PRIMARY,
                       "大语言模型能量推断 (≥85%准确率)", expected_latency_ms=500, accuracy_relative=1.0)
    if rule_func:
        chain.add_step("关键词规则", rule_func, FallbackLevel.FALLBACK_1,
                       "基于关键词的规则推断 (≥60%准确率)", expected_latency_ms=10, accuracy_relative=0.71)
    if default_func:
        chain.add_step("默认值", default_func, FallbackLevel.FALLBACK_2,
                       "使用预设默认值 (无推断能力)", expected_latency_ms=1, accuracy_relative=0.40)

    return chain


def create_vector_index_fallback_chain(
    faiss_func: Callable | None = None,
    linear_func: Callable | None = None,
) -> FallbackChain:
    """创建向量索引降级链: FAISS HNSW → 线性检索(numpy)"""
    chain = FallbackChain("vector_index")

    if faiss_func:
        chain.add_step("FAISS_HNSW", faiss_func, FallbackLevel.PRIMARY,
                       "FAISS HNSW 近似检索 (O(log n))", expected_latency_ms=5, accuracy_relative=1.0)
    if linear_func:
        chain.add_step("numpy线性", linear_func, FallbackLevel.FALLBACK_1,
                       "numpy 线性扫描 (O(n))", expected_latency_ms=50, accuracy_relative=1.0)

    return chain


def create_graph_fallback_chain(
    graph_func: Callable | None = None,
    vector_func: Callable | None = None,
) -> FallbackChain:
    """创建图谱降级链: MemoryGraph → 纯向量检索"""
    chain = FallbackChain("graph")

    if graph_func:
        chain.add_step("MemoryGraph", graph_func, FallbackLevel.PRIMARY,
                       "因果图谱多跳推理", expected_latency_ms=20, accuracy_relative=1.0)
    if vector_func:
        chain.add_step("纯向量检索", vector_func, FallbackLevel.FALLBACK_1,
                       "不使用图谱的纯向量相似度检索", expected_latency_ms=10, accuracy_relative=0.70)

    return chain


def create_temporal_fallback_chain(
    spacetime_func: Callable | None = None,
    decay_func: Callable | None = None,
) -> FallbackChain:
    """创建时空降级链: SpacetimeIndex → TemporalSystem(时序衰减)"""
    chain = FallbackChain("temporal")

    if spacetime_func:
        chain.add_step("SpacetimeIndex", spacetime_func, FallbackLevel.PRIMARY,
                       "时空索引 (空间坐标+时间维度)", expected_latency_ms=15, accuracy_relative=1.0)
    if decay_func:
        chain.add_step("TemporalSystem", decay_func, FallbackLevel.FALLBACK_1,
                       "纯时序衰减系统", expected_latency_ms=5, accuracy_relative=0.75)

    return chain


def create_session_fallback_chain(
    manager_func: Callable | None = None,
    memory_func: Callable | None = None,
) -> FallbackChain:
    """创建会话降级链: SessionManager → 内存 Session"""
    chain = FallbackChain("session")

    if manager_func:
        chain.add_step("SessionManager", manager_func, FallbackLevel.PRIMARY,
                       "持久化会话管理器", expected_latency_ms=10, accuracy_relative=1.0)
    if memory_func:
        chain.add_step("内存Session", memory_func, FallbackLevel.FALLBACK_1,
                       "内存会话 (进程重启丢失)", expected_latency_ms=1, accuracy_relative=0.90)

    return chain


# =============================================================================
# FallbackManager — 全局降级管理器
# =============================================================================

class FallbackManager:
    """全局降级管理器 — 管理所有组件的降级链"""

    def __init__(self):
        self._chains: dict[str, FallbackChain] = {}

    def register(self, component: str, chain: FallbackChain) -> FallbackManager:
        """注册组件降级链"""
        self._chains[component] = chain
        return self

    def get_chain(self, component: str) -> FallbackChain | None:
        """获取组件降级链"""
        return self._chains.get(component)

    def execute(self, component: str, *args, **kwargs) -> FallbackResult:
        """执行组件降级链"""
        chain = self._chains.get(component)
        if not chain:
            return FallbackResult(
                success=False,
                error=SuMemoryError(
                    ErrorCode.CONFIG_INVALID_PARAM,
                    param="component",
                    value=component,
                    reason=f"未注册的降级组件。已注册: {list(self._chains.keys())}",
                ),
            )
        return chain.try_execute(*args, **kwargs)

    def get_all_stats(self) -> dict[str, dict[str, int]]:
        """获取所有组件降级统计"""
        return {name: chain.get_stats() for name, chain in self._chains.items()}


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "FallbackLevel",
    "FallbackStep",
    "FallbackResult",
    "FallbackChain",
    "FallbackManager",
    "create_embedding_fallback_chain",
    "create_storage_fallback_chain",
    "create_prediction_fallback_chain",
    "create_vector_index_fallback_chain",
    "create_graph_fallback_chain",
    "create_temporal_fallback_chain",
    "create_session_fallback_chain",
]
