"""
存储抽象层 — StorageBackend ABC (v2.7.0)

定义异步存储后端的统一接口，支持：
- PgVectorBackend (hot tier, pgvector + asyncpg)
- SQLiteBackend  (warm tier, SQLite via asyncio.to_thread)
- MemoryBackend   (dev/test, 纯内存)

所有方法均为 async，CPU密集型通过 asyncio.to_thread() 包装。

Example:
    >>> from su_memory.storage.base import StorageBackend, AsyncMemoryItem
    >>> backend = SomeAsyncBackend()
    >>> item = AsyncMemoryItem(id="m1", content="hello", embedding=[0.1, 0.2])
    >>> await backend.aadd_memory(item)
    >>> results = await backend.aquery([0.1, 0.2], top_k=5)
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# AsyncMemoryItem — 异步存储的记忆条目
# =============================================================================

@dataclass
class AsyncMemoryItem:
    """异步记忆条目数据类

    与现有 MemoryItem 保持兼容，额外增加 tier/access_count 字段
    用于分层存储的自动升降级。

    Attributes:
        id: 唯一标识符
        content: 记忆内容
        embedding: 向量嵌入
        metadata: 元数据字典
        energy_type: 能量类型 (用于能量感知)
        category: 分类
        timestamp: 时间戳
        tier: 存储层级 (hot/warm/cold)
        access_count: 访问次数
        last_access: 最后访问时间
    """
    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    energy_type: str = "neutral"
    category: str = "general"
    timestamp: float = field(default_factory=time.time)
    tier: str = "hot"
    access_count: int = 0
    last_access: float | None = None

    def to_dict(self) -> dict:
        """转为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "energy_type": self.energy_type,
            "category": self.category,
            "timestamp": self.timestamp,
            "tier": self.tier,
            "access_count": self.access_count,
            "last_access": self.last_access,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AsyncMemoryItem:
        """从字典创建"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            energy_type=data.get("energy_type", "neutral"),
            category=data.get("category", "general"),
            timestamp=data.get("timestamp", time.time()),
            tier=data.get("tier", "hot"),
            access_count=data.get("access_count", 0),
            last_access=data.get("last_access"),
        )


# =============================================================================
# StorageBackend — 异步存储抽象基类
# =============================================================================

class StorageBackend(ABC):
    """异步存储后端抽象基类

    所有存储后端必须实现此接口。
    方法命名以 'a' 前缀表示 async，与 AsyncSuMemory 保持一致。
    """

    # ── 元信息 ───────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称 (e.g. 'pgvector', 'sqlite', 'memory')"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """后端版本"""
        pass

    @property
    def supports_vector(self) -> bool:
        """是否支持向量检索"""
        return True

    @property
    def supports_tiered(self) -> bool:
        """是否原生支持分层存储"""
        return False

    # ── 核心 CRUD ────────────────────────────────────────────────────

    @abstractmethod
    async def aadd_memory(self, item: AsyncMemoryItem) -> str:
        """异步添加单条记忆

        Args:
            item: 记忆条目

        Returns:
            记忆 ID
        """
        pass

    @abstractmethod
    async def aadd_batch(self, items: list[AsyncMemoryItem]) -> list[str]:
        """异步批量添加记忆

        Args:
            items: 记忆列表

        Returns:
            ID 列表
        """
        pass

    @abstractmethod
    async def aquery(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[AsyncMemoryItem]:
        """异步向量检索

        Args:
            embedding: 查询向量
            top_k: 返回数量
            filters: 可选的元数据过滤条件

        Returns:
            按相似度排序的记忆列表
        """
        pass

    @abstractmethod
    async def aget(self, memory_id: str) -> AsyncMemoryItem | None:
        """获取单条记忆"""
        pass

    @abstractmethod
    async def adelete(self, memory_id: str) -> bool:
        """删除单条记忆"""
        pass

    @abstractmethod
    async def adelete_batch(self, memory_ids: list[str]) -> int:
        """批量删除"""
        pass

    # ── 管理 ─────────────────────────────────────────────────────────

    @abstractmethod
    async def aget_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        pass

    @abstractmethod
    async def aclear(self) -> int:
        """清空所有数据，返回删除数"""
        pass

    @abstractmethod
    async def aclose(self) -> None:
        """关闭连接/释放资源"""
        pass

    @abstractmethod
    async def ahealth_check(self) -> bool:
        """健康检查"""
        pass

    # ── 生命周期 (可选) ──────────────────────────────────────────────

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.aclose()
        return False

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
