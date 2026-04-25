"""
Gateway路由 - API端点定义
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from .auth import verify_api_key, get_current_tenant
from memory_engine.manager import MemoryManager
import os

logger = logging.getLogger(__name__)
router = APIRouter()

# 记忆管理器实例
memory_manager = MemoryManager()


# ==================== 租户接口 ====================

class CreateTenantRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    plan: str = Field(default="standard")  # standard / enterprise


class CreateTenantResponse(BaseModel):
    tenant_id: str
    name: str
    api_key: str
    created_at: str


@router.post("/tenant/create", response_model=CreateTenantResponse)
async def create_tenant(req: CreateTenantRequest, api_key: str = Depends(verify_api_key)):
    """创建租户（需要管理员鉴权）"""
    # 验证管理员权限：需要有效的API Key
    admin_secret = os.getenv("ADMIN_SECRET")
    if admin_secret and api_key != admin_secret:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for tenant creation"
        )
    result = await memory_manager.create_tenant(req.name, req.plan)
    return result


# ==================== 记忆接口 ====================

class AddMemoryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1, max_length=100000)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MemoryItem(BaseModel):
    id: str
    content: str
    score: float
    timestamp: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AddMemoryResponse(BaseModel):
    memory_id: str
    status: str


@router.post("/memory/add", response_model=AddMemoryResponse)
async def add_memory(
    req: AddMemoryRequest,
    tenant_id: str = Depends(verify_api_key)
):
    """写入记忆"""
    memory_id = await memory_manager.add_memory(
        tenant_id=tenant_id,
        user_id=req.user_id,
        content=req.content,
        metadata=req.metadata
    )
    return AddMemoryResponse(memory_id=memory_id, status="stored")


class QueryMemoryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    limit: int = Field(default=8, ge=1, le=100)


class QueryMemoryResponse(BaseModel):
    memories: List[MemoryItem]
    query_time_ms: float


@router.post("/memory/query", response_model=QueryMemoryResponse)
async def query_memory(
    req: QueryMemoryRequest,
    tenant_id: str = Depends(verify_api_key)
):
    """检索记忆"""
    import time
    start = time.time()
    
    memories = await memory_manager.query_memory(
        tenant_id=tenant_id,
        user_id=req.user_id,
        query=req.query,
        limit=req.limit
    )
    
    query_time = (time.time() - start) * 1000
    
    return QueryMemoryResponse(
        memories=[MemoryItem(**m) for m in memories],
        query_time_ms=round(query_time, 2)
    )


class DeleteMemoryRequest(BaseModel):
    user_id: str
    memory_id: str


@router.post("/memory/delete")
async def delete_memory(
    req: DeleteMemoryRequest,
    tenant_id: str = Depends(verify_api_key)
):
    """删除记忆"""
    await memory_manager.delete_memory(
        tenant_id=tenant_id,
        user_id=req.user_id,
        memory_id=req.memory_id
    )
    return {"status": "deleted"}


# ==================== 对话接口 ====================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str = "default"
    messages: List[ChatMessage]
    user_id: str
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=2048, ge=1)


class ChatCompletionsResponse(BaseModel):
    id: str
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


@router.post("/chat/completions", response_model=ChatCompletionsResponse)
async def chat_completions(
    req: ChatCompletionsRequest,
    tenant_id: str = Depends(verify_api_key)
):
    """带记忆的对话"""
    # 检索相关记忆
    last_message = req.messages[-1].content if req.messages else ""
    
    memories = await memory_manager.query_memory(
        tenant_id=tenant_id,
        user_id=req.user_id,
        query=last_message,
        limit=8
    )
    
    # 构建带记忆的Prompt
    memory_context = "\n".join([m["content"] for m in memories])
    system_prompt = f"你是一个专业的AI助手。以下是与当前用户相关的背景信息：\n{memory_context}\n\n请基于以上信息回答用户问题。"
    
    # 构建消息
    from fastapi import Request
    from llm_adapter.openai_compat import LLMAdapter
    
    llm = LLMAdapter()
    response = await llm.chat(
        messages=[{"role": "system", "content": system_prompt}] + 
                 [{"role": m.role, "content": m.content} for m in req.messages],
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens
    )
    
    return response


# ==================== 统计接口 ====================

class MemoryStats(BaseModel):
    user_id: str
    total_memories: int
    active_memories: int
    archived_memories: int
    storage_bytes: int


@router.get("/memory/stats/{user_id}", response_model=MemoryStats)
async def get_memory_stats(
    user_id: str,
    tenant_id: str = Depends(verify_api_key)
):
    """获取记忆统计"""
    stats = await memory_manager.get_stats(tenant_id, user_id)
    return MemoryStats(**stats)
