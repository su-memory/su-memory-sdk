"""
su-memory REST API Server (v3.5.5)
轻量级 FastAPI wrapper，一行启动覆盖非 Python 生态

启动方式：
    uvicorn su_memory.api.server:app --reload --port 8000

或：
    python -m su_memory.api.server
"""

import asyncio
import time
from collections import deque
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi import Query as FastQuery
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from su_memory import SuMemory
from su_memory._sys._stream import to_sse

# ── Pydantic Models ──────────────────────────────────────────────────────────

class MemoryAddRequest(BaseModel):
    """添加记忆请求"""
    content: str = Field(..., description="记忆内容", examples=["用户喜欢 Python 编程"])
    metadata: dict[str, Any] | None = Field(None, description="可选元数据", examples=[{"source": "chat"}])


class MemoryAddBatchRequest(BaseModel):
    """批量添加记忆请求"""
    items: list[dict[str, Any]] = Field(
        ...,
        description="记忆列表",
        examples=[[
            {"content": "用户喜欢 Python", "metadata": {"source": "chat"}},
            {"content": "用户住在北京", "metadata": {"source": "profile"}},
        ]],
    )


class MemoryQueryRequest(BaseModel):
    """查询请求"""
    text: str = Field(..., description="查询文本", examples=["用户喜欢什么编程语言？"])
    top_k: int = Field(5, description="返回数量", ge=1, le=100, examples=[5])


class MemoryMultiHopRequest(BaseModel):
    """多跳推理请求"""
    query: str = Field(..., description="查询文本", examples=["用户去年和今年的技术栈有什么变化？"])
    max_hops: int = Field(3, description="最大跳数", ge=1, le=10, examples=[3])
    top_k: int = Field(5, description="每跳返回数量", ge=1, le=50, examples=[5])
    fusion_mode: str = Field("hybrid", description="融合模式", examples=["hybrid"])


class MemoryUpdateRequest(BaseModel):
    """更新记忆请求 (v3.5.5 新增)"""
    content: str | None = Field(None, description="新的记忆内容", examples=["更新后的内容"])
    metadata: dict[str, Any] | None = Field(None, description="新的元数据", examples=[{"source": "updated"}])


class DocumentIngestRequest(BaseModel):
    """文档摄入请求 (v3.5.5 新增)"""
    text: str = Field(..., description="文档原文内容", examples=["这是一篇关于机器学习的文档..."])
    chunk_size: int = Field(512, description="分块大小（字符数）", ge=64, le=4096, examples=[512])
    chunk_overlap: int = Field(64, description="分块重叠（字符数）", ge=0, le=512, examples=[64])
    metadata: dict[str, Any] | None = Field(None, description="文档级元数据", examples=[{"filename": "ml_intro.txt"}])


class ProfileResponse(BaseModel):
    """用户画像响应 (v3.5.5 新增)"""
    total_memories: int = Field(0, description="总记忆数")
    category_distribution: dict[str, int] = Field(default_factory=dict, description="分类分布")
    top_keywords: list[str] = Field(default_factory=list, description="高频关键词")
    recent_topics: list[str] = Field(default_factory=list, description="最近话题")
    interaction_count: int = Field(0, description="总交互次数")


# ── FastAPI App ───────────────────────────────────────────────────────────────

# v3.5.5 P0-4: API Key 鉴权
import os as _os

_API_KEY = _os.environ.get("SU_MEMORY_API_KEY", "")
_security = HTTPBearer(auto_error=False)


def verify_api_key(credentials: HTTPAuthorizationCredentials | None = Depends(_security)) -> bool:
    """验证 API Key (v3.5.5 P0-4修复)"""
    if not _API_KEY:
        return True  # 未设置 API Key 时允许所有请求 (向后兼容)
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if credentials.credentials != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

app = FastAPI(
    title="su-memory API",
    description="语义记忆引擎 REST API — 一行代码让 AI 拥有记忆能力。支持记忆增删查改、语义检索、多跳推理、SSE 流式查询、WebSocket 实时指标推送。",
    version="4.4.1",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Health", "description": "健康检查"},
        {"name": "Memories", "description": "记忆 CRUD 操作"},
        {"name": "Query", "description": "语义检索与多跳推理"},
        {"name": "Lifecycle", "description": "记忆生命周期管理（衰减、摘要、冲突检测）"},
        {"name": "Profile & Docs", "description": "用户画像与文档摄入 (v3.5.5 新增)"},
        {"name": "Metrics & Logs", "description": "性能指标与检索日志 (v3.5.5 新增)"},
    ],
)

# CORS - 允许所有来源，方便调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# v3.5.5 P0-4: 全局 API Key 鉴权中间件
@app.middleware("http")
async def auth_middleware(request, call_next):
    """鉴权中间件：除 health/docs/redoc/openapi 外均需 API Key"""
    public_paths = {"/health", "/docs", "/redoc", "/openapi.json"}
    if request.url.path in public_paths or request.url.path.startswith("/docs/") or request.url.path.startswith("/redoc/"):
        return await call_next(request)
    if _API_KEY:
        auth_header = request.headers.get("Authorization", "")
        client_key = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
        if client_key != _API_KEY:
            return JSONResponse(status_code=401, content={"error": "Unauthorized", "detail": "Missing or invalid API Key"})
    return await call_next(request)

# ── 全局客户端实例 ──────────────────────────────────────────────────────────

_client: SuMemory | None = None


def get_client() -> SuMemory:
    """获取或创建客户端实例"""
    global _client
    if _client is None:
        _client = SuMemory()
    return _client


# ── 服务端指标收集 (v3.5.5 新增) ───────────────────────────────────────────

_query_log: deque[dict[str, Any]] = deque(maxlen=1000)
_latency_buffer: deque[float] = deque(maxlen=500)
_qps_window: deque[float] = deque(maxlen=60)  # 最近 60 秒的请求时间戳
_query_counter: int = 0


def _record_query(query_text: str, latency_ms: float, hit_count: int) -> None:
    """记录查询日志与延迟样本"""
    global _query_counter
    _query_counter += 1
    entry = {
        "id": _query_counter,
        "timestamp": time.time(),
        "query": query_text[:200],
        "latency_ms": round(latency_ms, 3),
        "hit_count": hit_count,
    }
    _query_log.appendleft(entry)
    _latency_buffer.append(latency_ms)
    _qps_window.append(time.time())


def _compute_metrics() -> dict[str, Any]:
    """计算性能指标"""
    now = time.time()
    # QPS: 最近 60 秒内的请求数
    cutoff = now - 60
    recent_qps = sum(1 for ts in _qps_window if ts > cutoff)
    qps = recent_qps / 60.0 if recent_qps > 0 else 0.0

    # 延迟分位数
    latencies = sorted(_latency_buffer) if _latency_buffer else [0]
    n = len(latencies)

    def _pct(p: float) -> float:
        """计算第 p 百分位延迟"""
        if n == 0:
            return 0.0
        idx = int(n * p / 100)
        return round(latencies[min(idx, n - 1)], 3)

    return {
        "qps": round(qps, 2),
        "total_queries": _query_counter,
        "latency_p50_ms": _pct(50),
        "latency_p95_ms": _pct(95),
        "latency_p99_ms": _pct(99),
        "latency_avg_ms": round(sum(latencies) / n, 3) if n > 0 else 0.0,
        "latency_samples": n,
    }


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"], summary="健康检查",
         description="返回服务健康状态，可用于 Kubernetes liveness probe。")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "su-memory API", "version": "4.4.1"}


# ── Memory Operations ──────────────────────────────────────────────────────────

@app.post("/memories", response_model=dict, tags=["Memories"], summary="添加单条记忆",
          description="向记忆库添加一条文本记忆，返回唯一 memory_id。")
async def add_memory(req: MemoryAddRequest):
    """添加单条记忆"""
    client = get_client()
    memory_id = client.add(req.content, req.metadata)
    return {"memory_id": memory_id, "status": "added"}


@app.post("/memories/batch", response_model=dict, tags=["Memories"], summary="批量添加记忆",
          description="一次添加多条记忆，内部批量编码优化。")
async def add_memory_batch(req: MemoryAddBatchRequest):
    """批量添加记忆"""
    client = get_client()
    memory_ids = client.add_batch(req.items)
    return {"count": len(memory_ids), "memory_ids": memory_ids}


@app.get("/memories", response_model=list[dict], tags=["Memories"], summary="列出记忆",
         description="分页列出所有记忆，返回 id、内容摘要和分类。")
async def list_memories(limit: int = FastQuery(100, ge=1, le=1000, description="返回数量上限")):
    """列出所有记忆"""
    client = get_client()
    memories = client.get_all_memories()
    return [{"id": m.get("id"), "content": m.get("content")[:100],
             "category": m.get("category")} for m in memories[:limit]]


@app.get("/memories/{memory_id}", response_model=dict, tags=["Memories"], summary="获取单条记忆",
         description="按 ID 获取记忆的完整详情。")
async def get_memory(memory_id: str):
    """获取单条记忆"""
    client = get_client()
    memory = client.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@app.put("/memories/{memory_id}", response_model=dict, tags=["Memories"], summary="编辑记忆 (v3.5.5 新增)",
         description="更新指定记忆的内容和/或元数据。如果 content 为空则仅更新 metadata。")
async def update_memory(memory_id: str, req: MemoryUpdateRequest):
    """编辑单条记忆 (v3.5.5 新增)"""
    client = get_client()
    memory = client.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    # 更新逻辑：forget + add，保留旧 ID 的历史关联
    new_content = req.content if req.content is not None else memory.get("content", "")
    new_metadata = req.metadata if req.metadata is not None else memory.get("metadata", {})

    client.forget(memory_id)
    new_id = client.add(new_content, new_metadata)
    return {"memory_id": new_id, "previous_id": memory_id, "status": "updated"}


@app.delete("/memories/{memory_id}", response_model=dict, tags=["Memories"], summary="删除单条记忆",
            description="按 ID 删除一条记忆。")
async def delete_memory(memory_id: str):
    """删除单条记忆"""
    client = get_client()
    success = client.forget(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"memory_id": memory_id, "status": "deleted"}


# ── Query Operations ────────────────────────────────────────────────────────────

@app.post("/query", response_model=dict, tags=["Query"], summary="语义检索记忆",
          description="输入自然语言查询，返回最相关的 top_k 条记忆。同时记录查询日志与性能指标。")
async def query_memories(req: MemoryQueryRequest):
    """语义检索记忆"""
    client = get_client()
    t0 = time.perf_counter()
    results = client.query(req.text, req.top_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    # 记录指标
    _record_query(req.text, latency_ms, len(results))

    return {
        "query": req.text,
        "count": len(results),
        "latency_ms": round(latency_ms, 3),
        "results": [
            {
                "memory_id": r.memory_id,
                "content": r.content,
                "score": float(r.score),
                "metadata": r.metadata,
            }
            for r in results
        ]
    }


@app.post("/query/multihop", response_model=dict, tags=["Query"], summary="多跳推理查询",
          description="跨记忆的多跳推理：从初始查询出发，逐跳扩展，融合多轮结果。")
async def query_multihop(req: MemoryMultiHopRequest):
    """多跳推理查询"""
    client = get_client()

    if not hasattr(client, 'query_multihop'):
        raise HTTPException(
            status_code=501,
            detail="query_multihop not available, use SuMemoryLitePro"
        )

    results = client.query_multihop(
        req.query,
        max_hops=req.max_hops,
        top_k=req.top_k,
        fusion_mode=req.fusion_mode,
    )

    return {
        "query": req.query,
        "hops": req.max_hops,
        "count": len(results),
        "chain": results
    }


# ── Memory Lifecycle ──────────────────────────────────────────────────────────

@app.post("/memories/decay", response_model=dict, tags=["Lifecycle"], summary="时间衰减",
          description="对超过指定天数的旧记忆执行衰减策略。")
async def decay_memories(days: int = FastQuery(30, ge=1, description="衰减阈值天数")):
    """时间衰减：归档旧记忆"""
    client = get_client()
    result = client.decay(days)
    return result


@app.post("/memories/summarize", response_model=dict, tags=["Lifecycle"], summary="记忆摘要",
          description="将多段记忆压缩为摘要文本，支持按主题过滤。")
async def summarize_memories(
    topic: str | None = FastQuery(None, description="可选主题过滤"),
    max_memories: int = FastQuery(10, ge=1, le=100, description="最大记忆数"),
):
    """压缩记忆为摘要"""
    client = get_client()
    summary = client.summarize(topic, max_memories)
    return {"summary": summary}


@app.get("/memories/conflicts", response_model=dict, tags=["Lifecycle"], summary="矛盾检测",
         description="检测记忆库中的语义矛盾（如用户说了相反的信息）。")
async def detect_conflicts():
    """检测矛盾记忆"""
    client = get_client()
    conflicts = client.conflict_resolution()
    return {"count": len(conflicts), "conflicts": conflicts}


@app.delete("/memories", response_model=dict, tags=["Lifecycle"], summary="清空记忆",
            description="删除全部记忆数据。此操作不可逆。")
async def clear_all_memories():
    """清空所有记忆"""
    client = get_client()
    count = client.clear()
    return {"deleted_count": count}


# ── Profile & Documents (v3.5.5 新增) ────────────────────────────────────────

@app.get("/profile", response_model=dict, tags=["Profile & Docs"], summary="获取用户画像 (v3.5.5 新增)",
         description="从记忆库中自动提取用户画像：分类分布、高频关键词、最近话题。")
async def get_profile():
    """获取用户画像 (v3.5.5 新增)"""
    client = get_client()
    stats = client.get_stats()
    memories = client.get_all_memories()

    # 提取关键词：简单词频统计（取 top 20）
    word_freq: dict[str, int] = {}
    for m in memories:
        content = m.get("content", "")
        # 简单分词：按空白和常见标点拆分
        for word in content.replace(",", " ").replace("，", " ").replace(".", " ").replace("。", " ").split():
            word = word.strip().lower()
            if len(word) >= 2:
                word_freq[word] = word_freq.get(word, 0) + 1

    top_keywords = sorted(word_freq, key=word_freq.get, reverse=True)[:20]

    # 最近话题：最近 10 条记忆
    recent = memories[-10:] if memories else []
    recent_topics = [m.get("content", "")[:80] for m in reversed(recent)]

    return {
        "total_memories": stats.get("total_memories", 0),
        "category_distribution": stats.get("category_distribution", {}),
        "energy_distribution": stats.get("energy_distribution", {}),
        "top_keywords": top_keywords,
        "recent_topics": recent_topics,
        "interaction_count": _query_counter,
    }


@app.post("/documents/ingest", response_model=dict, tags=["Profile & Docs"], summary="文档摄入 (v3.5.5 新增)",
          description="摄入文本文档：自动分块 → 批量编码 → 写入记忆库。")
async def ingest_document(req: DocumentIngestRequest):
    """文档摄入 (v3.5.5 新增)"""
    client = get_client()
    text = req.text
    chunk_size = req.chunk_size
    chunk_overlap = req.chunk_overlap

    # 简单分块策略（按字符数 + overlap）
    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        chunk = text[pos:pos + chunk_size]
        chunks.append(chunk)
        pos += chunk_size - chunk_overlap
        if pos >= len(text):
            break

    # 构建批量添加项
    items = [
        {
            "content": chunk,
            "metadata": {
                **(req.metadata or {}),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "ingest_source": "document",
            },
        }
        for i, chunk in enumerate(chunks)
    ]

    memory_ids = client.add_batch(items)
    return {
        "status": "ingested",
        "total_chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "memory_ids": memory_ids,
    }


# ── Metrics & Logs (v3.5.5 新增) ─────────────────────────────────────────────

@app.get("/metrics", response_model=dict, tags=["Metrics & Logs"], summary="性能指标 (v3.5.5 新增)",
         description="返回实时性能指标：QPS、P50/P95/P99 延迟、请求总数、延迟样本数。")
async def get_metrics():
    """获取性能指标 (v3.5.5 新增)"""
    metrics = _compute_metrics()
    client = get_client()
    stats = client.get_stats()
    metrics["memory_count"] = stats.get("total_memories", 0)
    metrics["faiss_enabled"] = stats.get("faiss_enabled", False)
    metrics["async_embed"] = stats.get("async_embed", False)
    metrics["pending_embeddings"] = stats.get("pending_embeddings", 0)
    return metrics


@app.get("/logs/queries", response_model=dict, tags=["Metrics & Logs"], summary="检索日志 (v3.5.5 新增)",
         description="返回最近查询日志列表，支持分页。")
async def get_query_logs(
    page: int = FastQuery(1, ge=1, description="页码"),
    page_size: int = FastQuery(20, ge=1, le=100, description="每页条数"),
):
    """检索日志列表 (v3.5.5 新增)"""
    start = (page - 1) * page_size
    end = start + page_size
    logs = list(_query_log)
    total = len(logs)
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": logs[start:end],
    }


@app.get("/logs/queries/{query_id}", response_model=dict, tags=["Metrics & Logs"], summary="查询日志详情 (v3.5.5 新增)",
         description="按 ID 获取单条查询日志。")
async def get_query_log_detail(query_id: int):
    """单条查询日志详情 (v3.5.5 新增)"""
    for entry in _query_log:
        if entry["id"] == query_id:
            return entry
    raise HTTPException(status_code=404, detail="Query log entry not found")


# ── WebSocket Metrics (v3.5.5 新增) ──────────────────────────────────────────

@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket):
    """WebSocket 实时性能指标推送 (v3.5.5 新增)

    每秒推送一次当前的性能指标快照：
    ```json
    {"qps": 1.5, "latency_p50_ms": 12.3, "latency_p95_ms": 45.6, "memory_count": 150, ...}
    ```
    """
    await websocket.accept()
    try:
        while True:
            metrics = _compute_metrics()
            client = get_client()
            stats = client.get_stats()
            metrics["memory_count"] = stats.get("total_memories", 0)
            metrics["faiss_enabled"] = stats.get("faiss_enabled", False)
            await websocket.send_json(metrics)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close()


# ── Stream Query (SSE) ───────────────────────────────────────────────────────

@app.get("/query/stream", tags=["Query"], summary="SSE 流式查询",
         description="返回 Server-Sent Events 格式的流式查询结果。客户端可逐条接收结果，无需等待全部查询完成。\n\n"
                     "用法: `curl -N \"http://localhost:8000/query/stream?q=项目&top_k=5\"`")
async def stream_query(
    q: str = FastQuery(..., description="查询文本"),
    top_k: int = FastQuery(5, description="返回数量"),
):
    """SSE 流式查询"""
    client = get_client()

    async def _stream():
        chunks = client.astream_query(q, top_k)
        async for event in to_sse(chunks):
            yield event

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Async Memory Operations ─────────────────────────────────────────────────

@app.post("/memories/async", response_model=dict, tags=["Memories"], summary="异步添加记忆",
          description="异步添加单条记忆到记忆库。")
async def add_memory_async(req: MemoryAddRequest):
    """异步添加单条记忆"""
    client = get_client()
    memory_id = await asyncio.to_thread(client.add, req.content, req.metadata)
    return {"memory_id": memory_id, "status": "added"}


@app.post("/query/async", response_model=dict, tags=["Query"], summary="异步语义检索",
          description="异步执行语义检索，适合批量查询场景。")
async def query_memories_async(req: MemoryQueryRequest):
    """异步语义检索"""
    client = get_client()
    t0 = time.perf_counter()
    results = await asyncio.to_thread(client.query, req.text, req.top_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    _record_query(req.text, latency_ms, len(results))

    return {
        "query": req.text,
        "count": len(results),
        "latency_ms": round(latency_ms, 3),
        "results": [
            {
                "memory_id": r.memory_id,
                "content": r.content,
                "score": float(r.score),
                "metadata": r.metadata,
            }
            for r in results
        ]
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.get("/stats", response_model=dict, tags=["Memories"], summary="记忆统计",
         description="获取记忆库的完整统计信息：总量、分类分布、FAISS 状态、异步嵌入状态。")
async def get_stats():
    """获取记忆统计"""
    client = get_client()
    return client.get_stats()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """启动服务器"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    print("🚀 启动 su-memory API Server v3.5.5...")
    print("   访问 http://localhost:8000/docs 查看 API 文档")
    run_server()
