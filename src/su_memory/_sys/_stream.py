"""
su-memory 流式查询引擎

提供异步流式查询的底层支持：
- SSE (Server-Sent Events) 适配器
- 流式多跳推理
- 流聚合与收集工具

Example:
    >>> from su_memory._sys._stream import to_sse
    >>> async for event in to_sse(client.astream_query("项目")):
    ...     print(event)  # "data: {...}\\n\\n"
"""


from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any


# =============================================================================
# SSE 适配器 — 将 StreamChunk 转为 Server-Sent Events
# =============================================================================

async def to_sse(
    chunks: AsyncIterator[Any],
    event_type: str = "message",
    retry_ms: int = 3000,
) -> AsyncIterator[str]:
    """将异步 StreamChunk 迭代器转换为 SSE 格式

    每个 chunk 转为:
        event: {event_type}
        data: {json.dumps(chunk)}
        retry: {retry_ms}

    Args:
        chunks: StreamChunk 异步迭代器
        event_type: SSE event 名称
        retry_ms: 重连间隔毫秒

    Yields:
        SSE 格式的字符串
    """
    async for chunk in chunks:
        data_str = json.dumps(
            _chunk_to_dict(chunk),
            ensure_ascii=False,
            default=str,
        )
        lines = [f"event: {event_type}", f"data: {data_str}"]
        if retry_ms:
            lines.append(f"retry: {retry_ms}")
        yield "\n".join(lines) + "\n\n"


def _chunk_to_dict(chunk: Any) -> dict:
    """将 StreamChunk 或任意对象转为可序列化字典"""
    if hasattr(chunk, "__slots__"):
        d = {}
        for slot in chunk.__slots__:
            val = getattr(chunk, slot, None)
            d[slot] = _serialize_value(val)
        return d
    if isinstance(chunk, dict):
        return chunk
    if isinstance(chunk, str):
        return {"data": chunk}
    return {"data": str(chunk)}


def _serialize_value(val: Any) -> Any:
    """递归序列化值"""
    if hasattr(val, "to_dict"):
        return val.to_dict()
    if hasattr(val, "__dict__"):
        return val.__dict__
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    if isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    return str(val)


# =============================================================================
# 流式多跳推理
# =============================================================================

async def astream_multihop(
    seed_query_fn,
    text: str,
    max_hops: int = 3,
) -> AsyncIterator[dict]:
    """流式多跳推理 — 逐跳 yield 中间结果

    每 hop 产生一个 partial chunk，最终产生 complete chunk。

    Args:
        seed_query_fn: async def fn(text) -> List 种子查询函数
        text: 查询文本
        max_hops: 最大跳数

    Yields:
        dict: {"type": "partial"/"complete", "hop": N, "results": [...], "progress": 0.X}
    """
    yield {
        "type": "partial",
        "hop": 0,
        "stage": "seeding",
        "progress": 0.05,
    }

    seed_results = await seed_query_fn(text)

    yield {
        "type": "partial",
        "hop": 0,
        "stage": "seeded",
        "results": seed_results[:5],
        "count": len(seed_results),
        "progress": 0.2,
    }

    total_results = list(seed_results)

    for hop in range(1, max_hops + 1):
        yield {
            "type": "partial",
            "hop": hop,
            "stage": "expanding",
            "progress": 0.2 + (hop / max_hops) * 0.6,
        }

        # 基于上一 hop 的结果扩展
        expanded = []
        for r in total_results[:3]:
            content = r.get("content", "") if isinstance(r, dict) else getattr(r, "content", "")
            if content:
                expanded_text = f"{text} → {content[:50]}"
                try:
                    hop_results = await seed_query_fn(expanded_text)
                    for hr in hop_results:
                        hr_dict = hr if isinstance(hr, dict) else {
                            "id": getattr(hr, "memory_id", ""),
                            "content": getattr(hr, "content", ""),
                            "score": getattr(hr, "score", 0.0) * (0.85 ** hop),
                            "hop": hop,
                        }
                        expanded.append(hr_dict)
                except Exception as e:
                    logger.debug("降级处理: %s", e)

        total_results.extend(expanded)
        await asyncio.sleep(0)  # yield to event loop

    yield {
        "type": "complete",
        "results": total_results,
        "hops": max_hops,
        "total": len(total_results),
        "progress": 1.0,
    }


# =============================================================================
# 流聚合工具
# =============================================================================

async def collect_chunks(
    chunks: AsyncIterator[Any],
    filter_type: str | None = None,
) -> list[Any]:
    """收集所有 chunk（可选过滤类型）

    Args:
        chunks: StreamChunk 迭代器
        filter_type: 仅收集此类型的 chunk ("partial"/"complete"/"error")

    Returns:
        收集到的 chunk 列表
    """
    results = []
    async for chunk in chunks:
        if filter_type:
            chunk_type = getattr(chunk, "type", "") if hasattr(chunk, "type") else ""
            if chunk_type != filter_type:
                continue
        results.append(chunk)
    return results


async def first_complete(chunks: AsyncIterator[Any]) -> Any | None:
    """获取第一个 complete 类型的 chunk

    Args:
        chunks: StreamChunk 迭代器

    Returns:
        第一个 type="complete" 的 chunk，如果没有则返回 None
    """
    async for chunk in chunks:
        chunk_type = getattr(chunk, "type", "") if hasattr(chunk, "type") else ""
        if chunk_type == "complete":
            return chunk
        elif chunk_type == "error":
            raise RuntimeError(
                f"流式查询错误: {getattr(chunk, 'data', str(chunk))}"
            )
    return None


# =============================================================================
# FastAPI SSE Response Helper
# =============================================================================

def create_sse_response(chunks: AsyncIterator[Any]) -> Any:
    """创建 FastAPI StreamingResponse

    用法:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse

        @app.get("/query/stream")
        async def stream_query(q: str):
            chunks = client.astream_query(q)
            return create_sse_response(chunks)

    Args:
        chunks: StreamChunk 异步迭代器

    Returns:
        StreamingResponse 实例
    """
    try:
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            to_sse(chunks),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
            },
        )
    except ImportError:
        raise ImportError(
            "需要 fastapi 才能创建 SSE Response: pip install su-memory[api]"
        )


__all__ = [
    "to_sse",
    "astream_multihop",
    "collect_chunks",
    "first_complete",
    "create_sse_response",
]
