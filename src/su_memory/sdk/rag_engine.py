"""
rag_engine — RAG 统一入口（facade）

经第一性原理评估，vector_graph_rag（语义向量 + 因果图谱）与 spatial_rag
（物理空间 + 轨迹）职责分明、无功能重叠，**不强行物理合并**（合并会降低内聚）。
本模块提供统一入口，便于外部用一致的 API 选择/组合两类 RAG。

用法:
    from su_memory.sdk.rag_engine import create_rag, RAGType

    # 语义向量 RAG（默认）
    rag = create_rag(RAGType.VECTOR_GRAPH)

    # 物理/轨迹空间 RAG
    rag = create_rag(RAGType.SPATIAL)

两个底层模块保持独立演进，本入口只做选择与（未来）组合编排。
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RAGType(str, Enum):
    """RAG 变体枚举。"""

    VECTOR_GRAPH = "vector_graph"  # 语义向量 + 因果图谱多跳
    SPATIAL = "spatial"            # 物理空间 KDTree + 轨迹追踪


def create_rag(
    rag_type: RAGType | str = RAGType.VECTOR_GRAPH,
    **kwargs: Any,
) -> Any:
    """统一创建 RAG 实例。

    Parameters
    ----------
    rag_type : RAGType | str
        RAG 变体。VECTOR_GRAPH 走 vector_graph_rag，SPATIAL 走 spatial_rag。
    **kwargs
        透传给底层 RAG 构造器。

    Returns
    -------
    VectorGraphRAG | SpatialRAG
        对应的 RAG 实例。
    """
    if isinstance(rag_type, str):
        rag_type = RAGType(rag_type)

    if rag_type == RAGType.VECTOR_GRAPH:
        try:
            from su_memory.sdk.vector_graph_rag import create_vector_graph_rag
        except ImportError as e:
            logger.error("vector_graph_rag 不可用: %s", e)
            raise
        return create_vector_graph_rag(**kwargs)

    if rag_type == RAGType.SPATIAL:
        try:
            from su_memory.sdk.spatial_rag import create_spatial_rag
        except ImportError as e:
            logger.error("spatial_rag 不可用: %s", e)
            raise
        return create_spatial_rag(**kwargs)

    raise ValueError(f"未知 RAG 类型: {rag_type}")


__all__ = ["RAGType", "create_rag"]
