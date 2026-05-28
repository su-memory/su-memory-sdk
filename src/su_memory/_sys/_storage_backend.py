"""
StorageBackend — 统一存储后端抽象层

定义所有存储后端 (SQLite / PostgreSQL / Redis) 的公共接口契约。
v3.0.0: 插件化 + 分布式存储架构的核心抽象。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# BackendType — 后端类型枚举
# =============================================================================

class BackendType(Enum):
    """存储后端类型"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    AUTO = "auto"  # 自动检测最优可用后端


# =============================================================================
# StorageMemory — 存储记忆数据模型
# =============================================================================

@dataclass
class StorageMemory:
    """存储层记忆数据模型"""
    memory_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    energy_type: Optional[str] = None
    created_at: Optional[float] = None
    score: float = 0.0


# =============================================================================
# StorageConfig — 存储配置
# =============================================================================

@dataclass
class StorageConfig:
    """存储后端通用配置"""
    # PostgreSQL
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "su_memory"
    pg_user: str = "postgres"
    pg_password: str = ""
    pg_pool_min: int = 5
    pg_pool_max: int = 20
    pg_table: str = "memories"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_ttl: Optional[int] = None  # 记忆过期时间 (秒)

    # SQLite
    sqlite_path: Optional[str] = None

    # 通用
    embedding_dim: int = 1536
    backend_type: BackendType = BackendType.SQLITE


# =============================================================================
# BackendHealth — 后端健康状态
# =============================================================================

@dataclass
class BackendHealth:
    """后端健康检查结果"""
    available: bool
    backend_type: BackendType
    latency_ms: float = 0.0
    memory_count: int = 0
    detail: str = ""
    error: Optional[str] = None


# =============================================================================
# StorageBackend — 抽象基类
# =============================================================================

class StorageBackend(ABC):
    """
    统一存储后端抽象接口。

    所有存储后端 (SQLite / PostgreSQL / Redis) 必须实现此接口。
    支持异步操作，确保分布式环境下的一致性。

    Required methods:
        - add(): 添加单条记忆
        - add_batch(): 批量添加记忆
        - query(): 向量相似度检索
        - delete(): 删除记忆
        - count(): 记忆总数
        - health_check(): 后端健康检查
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self._initialized = False

    @property
    def backend_type(self) -> BackendType:
        """返回后端类型"""
        raise NotImplementedError

    @abstractmethod
    async def initialize(self) -> bool:
        """
        初始化后端连接。

        Returns:
            是否初始化成功
        """
        ...

    @abstractmethod
    async def add(
        self,
        memory_id: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        energy_type: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> bool:
        """
        添加单条记忆。

        Args:
            memory_id: 记忆唯一标识
            content: 记忆内容
            embedding: 向量嵌入 (可选)
            metadata: 元数据
            energy_type: 能量类型 (semantic/causal/spacetime/generative/trust)
            created_at: 创建时间戳

        Returns:
            是否添加成功
        """
        ...

    async def add_batch(
        self,
        memories: List[StorageMemory],
    ) -> List[str]:
        """
        批量添加记忆。
        默认实现：逐条调用 add()。子类可重写为更高效的批量实现。

        Args:
            memories: 记忆列表

        Returns:
            成功添加的记忆ID列表
        """
        ids = []
        for mem in memories:
            ok = await self.add(
                memory_id=mem.memory_id,
                content=mem.content,
                embedding=mem.embedding,
                metadata=mem.metadata,
                energy_type=mem.energy_type,
                created_at=mem.created_at,
            )
            if ok:
                ids.append(mem.memory_id)
        return ids

    @abstractmethod
    async def query(
        self,
        vector: Optional[List[float]],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[StorageMemory]:
        """
        向量相似度检索。

        Args:
            vector: 查询向量 (为 None 时执行关键词检索)
            top_k: 返回结果数量
            filter_expr: 过滤表达式 (可选, 如 "energy_type == 'causal'")

        Returns:
            相似记忆列表，按 score 降序
        """
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """
        删除指定记忆。

        Args:
            memory_id: 记忆唯一标识

        Returns:
            是否删除成功
        """
        ...

    @abstractmethod
    async def count(self) -> int:
        """
        获取记忆总数。

        Returns:
            当前存储的记忆数量
        """
        ...

    @abstractmethod
    async def health_check(self) -> BackendHealth:
        """
        后端健康检查。

        Returns:
            健康状态信息
        """
        ...

    async def close(self) -> None:
        """
        关闭后端连接。
        默认空实现，子类按需重写。
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.backend_type.value if hasattr(self, 'backend_type') else 'unknown'})"


# =============================================================================
# BackendFactory — 后端工厂函数
# =============================================================================

async def create_backend(
    backend_type: BackendType,
    config: Optional[StorageConfig] = None,
) -> Optional[StorageBackend]:
    """
    创建存储后端实例。

    Args:
        backend_type: 后端类型
        config: 存储配置

    Returns:
        StorageBackend 实例，创建失败返回 None
    """
    cfg = config or StorageConfig()
    cfg.backend_type = backend_type

    if backend_type == BackendType.POSTGRESQL:
        try:
            from su_memory._sys._pg_storage import PgStorageBackend
            backend = PgStorageBackend(cfg)
            ok = await backend.initialize()
            return backend if ok else None
        except ImportError:
            return None

    elif backend_type == BackendType.REDIS:
        try:
            from su_memory._sys._redis_storage import RedisStorageBackend
            backend = RedisStorageBackend(cfg)
            ok = await backend.initialize()
            return backend if ok else None
        except ImportError:
            return None

    elif backend_type == BackendType.SQLITE:
        from su_memory._sys._sqlite_storage import SqliteStorageBackend
        backend = SqliteStorageBackend(cfg)
        ok = await backend.initialize()
        return backend if ok else None

    elif backend_type == BackendType.AUTO:
        return await _auto_detect_backend(cfg)

    return None


async def _auto_detect_backend(config: StorageConfig) -> StorageBackend:
    """
    自动检测最优可用后端。

    检测顺序: PostgreSQL → Redis → SQLite

    Args:
        config: 存储配置

    Returns:
        第一个可用的后端实例
    """
    import logging
    logger = logging.getLogger(__name__)

    # 1. 尝试 PostgreSQL
    try:
        from su_memory._sys._pg_storage import PgStorageBackend
        pg = PgStorageBackend(config)
        health = await pg.health_check()
        if health.available:
            logger.info("Auto-detected PostgreSQL backend")
            return pg
        else:
            await pg.close()
    except Exception as e:
        logger.debug("PostgreSQL auto-detect skipped: %s", e)

    # 2. 尝试 Redis
    try:
        from su_memory._sys._redis_storage import RedisStorageBackend
        redis = RedisStorageBackend(config)
        health = await redis.health_check()
        if health.available:
            logger.info("Auto-detected Redis backend")
            return redis
        else:
            await redis.close()
    except Exception as e:
        logger.debug("Redis auto-detect skipped: %s", e)

    # 3. 回退到 SQLite
    logger.info("Falling back to SQLite backend")
    from su_memory._sys._sqlite_storage import SqliteStorageBackend
    sqlite = SqliteStorageBackend(config)
    await sqlite.initialize()
    return sqlite
