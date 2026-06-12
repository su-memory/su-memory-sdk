"""
Session 级语义聚类工具函数。

v3.6.1: 从 longmem_eval.py 与 locomo_eval.py 中提取共享的
session 聚类逻辑，消除 P1-3 重复代码。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def cluster_by_session(
    results: list[Any],
    *,
    get_meta: Callable[[Any, str, Any], Any] | None = None,
    max_per_session: int = 3,
    max_total: int | None = None,
) -> list[Any]:
    """对检索结果按 session_id 语义聚类。

    1. 按 session_id 分组，每组取 top-{max_per_session}（假设结果已按
       score 降序排列）
    2. 按 chunk_index 稳定排序
    3. 截断到 max_total（None 时不截断）

    Args:
        results: 检索结果列表（dict 或具有 metadata 属性的对象）。
        get_meta: 可选的 metadata 提取函数 ``(item, key, default) -> value``。
                  若为 None，使用内置的 dict/object 兼容提取。
        max_per_session: 每组 session 保留的最大 chunk 数。
        max_total: 最终截断上限（None 表示不截断）。

    Returns:
        聚类后的结果列表。
    """
    if get_meta is None:
        get_meta = _default_get_meta

    # 按 session_id 分组
    session_groups: dict[str, list[Any]] = {}
    for r in results:
        sid = str(get_meta(r, "session_id", ""))
        if not sid:
            sid = "__default__"
        session_groups.setdefault(sid, []).append(r)

    # 每组取 top-N
    context_chunks: list[Any] = []
    for _sid, chunks in session_groups.items():
        context_chunks.extend(chunks[:max_per_session])

    # 按 chunk_index 稳定排序
    context_chunks.sort(key=lambda r: get_meta(r, "chunk_index", 0))

    # 截断
    if max_total is not None and len(context_chunks) > max_total:
        context_chunks = context_chunks[:max_total]

    return context_chunks


def _default_get_meta(item: Any, key: str, default: Any = "") -> Any:
    """内置 metadata 提取（兼容 dict 与对象类型）。"""
    meta = (
        item.get("metadata")
        if isinstance(item, dict)
        else getattr(item, "metadata", None) or {}
    )
    return meta.get(key, default)
