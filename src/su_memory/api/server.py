"""
su-memory REST API Server
轻量级 FastAPI wrapper，一行启动覆盖非 Python 生态

启动方式：
    uvicorn su_memory.api.server:app --reload --port 8000

或：
    python -m su_memory.api.server
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn

from su_memory import SuMemory

# ── Pydantic Models ──────────────────────────────────────────────────────────

class MemoryAddRequest(BaseModel):
    """添加记忆请求"""
    content: str = Field(..., description="记忆内容")
    metadata: Optional[Dict[str, Any]] = Field(None, description="可选元数据")


class MemoryAddBatchRequest(BaseModel):
    """批量添加记忆请求"""
    items: List[Dict[str, Any]] = Field(..., description="记忆列表")


class MemoryQueryRequest(BaseModel):
    """查询请求"""
    text: str = Field(..., description="查询文本")
    top_k: int = Field(5, description="返回数量")


class MemoryMultiHopRequest(BaseModel):
    """多跳推理请求"""
    query: str = Field(..., description="查询文本")
    max_hops: int = Field(3, description="最大跳数")
    top_k: int = Field(5, description="每跳返回数量")
    fusion_mode: str = Field("hybrid", description="融合模式")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="su-memory API",
    description="语义记忆引擎 REST API - 一行代码让 AI 拥有记忆能力",
    version="1.7.2",
)

# CORS - 允许所有来源，方便调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局客户端实例
_client: Optional[SuMemory] = None


def get_client() -> SuMemory:
    """获取或创建客户端实例"""
    global _client
    if _client is None:
        _client = SuMemory()
    return _client


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "su-memory API"}


# ── Memory Operations ──────────────────────────────────────────────────────────

@app.post("/memories", response_model=dict)
async def add_memory(req: MemoryAddRequest):
    """添加单条记忆"""
    client = get_client()
    memory_id = client.add(req.content, req.metadata)
    return {"memory_id": memory_id, "status": "added"}


@app.post("/memories/batch", response_model=dict)
async def add_memory_batch(req: MemoryAddBatchRequest):
    """批量添加记忆"""
    client = get_client()
    memory_ids = client.add_batch(req.items)
    return {"count": len(memory_ids), "memory_ids": memory_ids}


@app.get("/memories", response_model=List[dict])
async def list_memories(limit: int = 100):
    """列出所有记忆"""
    client = get_client()
    memories = client.get_all_memories()
    return [{"id": m.get("id"), "content": m.get("content")[:100], 
             "category": m.get("category")} for m in memories[:limit]]


@app.get("/memories/{memory_id}", response_model=dict)
async def get_memory(memory_id: str):
    """获取单条记忆"""
    client = get_client()
    memory = client.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@app.delete("/memories/{memory_id}", response_model=dict)
async def delete_memory(memory_id: str):
    """删除单条记忆"""
    client = get_client()
    success = client.forget(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"memory_id": memory_id, "status": "deleted"}


# ── Query Operations ────────────────────────────────────────────────────────────

@app.post("/query", response_model=dict)
async def query_memories(req: MemoryQueryRequest):
    """语义检索记忆"""
    client = get_client()
    results = client.query(req.text, req.top_k)
    return {
        "query": req.text,
        "count": len(results),
        "results": [
            {
                "memory_id": r.memory.get("id"),
                "content": r.memory.get("content"),
                "score": r.score,
                "category": r.memory.get("category"),
            }
            for r in results
        ]
    }


@app.post("/query/multihop", response_model=dict)
async def query_multihop(req: MemoryMultiHopRequest):
    """多跳推理查询"""
    client = get_client()
    
    # 检查是否支持多跳
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

@app.post("/memories/decay", response_model=dict)
async def decay_memories(days: int = 30):
    """时间衰减：归档旧记忆"""
    client = get_client()
    result = client.decay(days)
    return result


@app.post("/memories/summarize", response_model=dict)
async def summarize_memories(topic: Optional[str] = None, max_memories: int = 10):
    """压缩记忆为摘要"""
    client = get_client()
    summary = client.summarize(topic, max_memories)
    return {"summary": summary}


@app.get("/memories/conflicts", response_model=dict)
async def detect_conflicts():
    """检测矛盾记忆"""
    client = get_client()
    conflicts = client.conflict_resolution()
    return {"count": len(conflicts), "conflicts": conflicts}


@app.delete("/memories", response_model=dict)
async def clear_all_memories():
    """清空所有记忆"""
    client = get_client()
    count = client.clear()
    return {"deleted_count": count}


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.get("/stats", response_model=dict)
async def get_stats():
    """获取记忆统计"""
    client = get_client()
    return client.get_stats()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """启动服务器"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    print("🚀 启动 su-memory API Server...")
    print("   访问 http://localhost:8000/docs 查看 API 文档")
    run_server()
