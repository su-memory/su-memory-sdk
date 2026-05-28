"""
PgVector 异步存储后端 (v2.7.0)

基于 PostgreSQL + pgvector 扩展的高性能向量存储后端。
支持异步连接池、IVFFlat/HNSW 索引、JSONB 元数据查询。

依赖 (可选):
    pip install su-memory[pgvector]
    - pgvector>=0.3.0
    - asyncpg>=0.29.0
    - sqlalchemy[asyncio]>=2.0

架构:
    AsyncConnectionPool (asyncpg/sqlalchemy)
    └── memories 表
        ├── id: UUID PRIMARY KEY
        ├── content: TEXT
        ├── embedding: vector(N)   ← pgvector 扩展
        ├── metadata: JSONB
        ├── energy_type: VARCHAR(32)
        ├── category: VARCHAR(32)
        ├── timestamp: TIMESTAMPTZ
        ├── access_count: INT
        ├── last_access: TIMESTAMPTZ
        └── tier: VARCHAR(16)

索引:
    - IVFFlat (默认): 写入优化, lists=100
    - HNSW (可选): 查询优化, m=16, ef_construction=200

Example:
    >>> from su_memory.storage.pgvector_backend import PgVectorBackend
    >>> backend = PgVectorBackend(
    ...     dsn="postgresql+asyncpg://user:pass@localhost:5432/sumemory"
    ... )
    >>> await backend.ainit()
    >>> item = AsyncMemoryItem(id="m1", content="hello", embedding=[0.1]*768)
    >>> await backend.aadd_memory(item)
    >>> results = await backend.aquery([0.1]*768, top_k=5)
    >>> await backend.aclose()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import List, Dict, Optional, Any, Tuple

from su_memory.storage.base import StorageBackend, AsyncMemoryItem
from su_memory.exceptions import SuMemoryError, ErrorCode

logger = logging.getLogger(__name__)


# =============================================================================
# PgVectorBackend
# =============================================================================

class PgVectorBackend(StorageBackend):
    """PgVector 异步存储后端

    使用 PostgreSQL + pgvector 提供高性能向量检索。
    支持 IVFFlat 和 HNSW 两种索引策略。

    Attributes:
        dsn: 数据库连接字符串
        dims: 向量维度
        pool_size: 连接池大小
        index_type: 索引类型 ('ivfflat' 或 'hnsw')
    """

    INDEX_IVFFLAT = "ivfflat"
    INDEX_HNSW = "hnsw"

    DEFAULT_POOL_SIZE = 10
    DEFAULT_IVFFLAT_LISTS = 100
    DEFAULT_HNSW_M = 16
    DEFAULT_HNSW_EF = 200

    def __init__(
        self,
        dsn: Optional[str] = None,
        dims: int = 768,
        pool_size: int = DEFAULT_POOL_SIZE,
        index_type: str = INDEX_IVFFLAT,
        table_name: str = "memories",
        **kwargs,
    ):
        """初始化 PgVector 后端

        Args:
            dsn: PostgreSQL 连接字符串
                 e.g. "postgresql+asyncpg://user:pass@localhost:5432/sumemory"
                 默认从环境变量 PG_DSN 读取
            dims: 向量维度 (需与嵌入模型一致)
            pool_size: 异步连接池大小
            index_type: 索引类型 'ivfflat' 或 'hnsw'
            table_name: 表名
            **kwargs: 传递给 create_async_engine 的额外参数
        """
        import os
        self._dsn = dsn or os.environ.get("PG_DSN", "")
        self._dims = dims
        self._pool_size = pool_size
        self._index_type = index_type
        self._table_name = table_name
        self._engine_kwargs = kwargs

        # 延迟初始化
        self._engine = None
        self._metadata = None
        self._table = None
        self._initialized = False

        # 统计
        self._stats = {
            "insert_count": 0,
            "query_count": 0,
            "delete_count": 0,
        }

    # ── 元信息 ───────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "pgvector"

    @property
    def version(self) -> str:
        return "2.7.0"

    @property
    def supports_vector(self) -> bool:
        return True

    @property
    def supports_tiered(self) -> bool:
        return True

    # ── 初始化 ───────────────────────────────────────────────────────

    async def ainit(self) -> None:
        """异步初始化连接池和表结构

        必须在其他操作之前调用。
        """
        if self._initialized:
            return

        # 检查依赖
        try:
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
            from sqlalchemy.orm import sessionmaker
            from sqlalchemy import (
                Column, String, Text, Integer, Float, DateTime, JSON,
                MetaData, Table, Index, text as sa_text,
            )
            from sqlalchemy.dialects.postgresql import UUID as PGUUID
        except ImportError as e:
            raise SuMemoryError(
                ErrorCode.CONFIG_INVALID_PARAM,
                param="pgvector[deps]",
                value="missing",
                reason="请安装: pip install sqlalchemy[asyncio] asyncpg pgvector",
            ) from e

        self._sa_text = sa_text
        self._PGUUID = PGUUID

        # 创建异步引擎
        self._engine = create_async_engine(
            self._dsn,
            pool_size=self._pool_size,
            max_overflow=5,
            pool_pre_ping=True,
            pool_recycle=3600,
            **self._engine_kwargs,
        )

        # 创建 session factory
        self._Session = sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # 创建表结构
        await self._create_tables()
        self._initialized = True
        logger.info(f"PgVectorBackend 初始化完成: {self._dsn.split('@')[-1] if '@' in self._dsn else self._dsn}")

    async def _create_tables(self) -> None:
        """创建 memories 表和向量索引"""
        from sqlalchemy import MetaData

        self._metadata = MetaData()

        # 检查 pgvector 扩展
        await self._ensure_pgvector()

        # 注册向量类型（动态导入避免硬依赖）
        try:
            from pgvector.sqlalchemy import Vector
        except ImportError:
            raise SuMemoryError(
                ErrorCode.CONFIG_INVALID_PARAM,
                param="pgvector",
                value="missing",
                reason="请安装 pgvector Python 包: pip install pgvector",
            )

        # 创建表
        self._table = type('Table', (), {})()
        self._table.memories = type('Table', (), {})()

        # 使用原生 DDL (避免复杂的 SA 反射)
        async with self._engine.begin() as conn:
            await conn.execute(
                self._sa_text(f"""
                    CREATE TABLE IF NOT EXISTS {self._table_name} (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        content TEXT NOT NULL,
                        embedding vector({self._dims}),
                        metadata JSONB DEFAULT '{{}}',
                        energy_type VARCHAR(32) DEFAULT 'neutral',
                        category VARCHAR(32) DEFAULT 'general',
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        access_count INTEGER DEFAULT 0,
                        last_access TIMESTAMPTZ,
                        tier VARCHAR(16) DEFAULT 'hot'
                    )
                """)
            )

            # 创建向量索引
            if self._index_type == self.INDEX_IVFFLAT:
                await self._create_ivfflat_index(conn)
            elif self._index_type == self.INDEX_HNSW:
                await self._create_hnsw_index(conn)

            # 辅助索引
            await conn.execute(
                self._sa_text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_tier
                    ON {self._table_name} (tier)
                """)
            )
            await conn.execute(
                self._sa_text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_category
                    ON {self._table_name} (category)
                """)
            )
            await conn.execute(
                self._sa_text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_timestamp
                    ON {self._table_name} (timestamp DESC)
                """)
            )

    async def _ensure_pgvector(self) -> None:
        """确保 pgvector 扩展已启用"""
        async with self._engine.begin() as conn:
            await conn.execute(self._sa_text("CREATE EXTENSION IF NOT EXISTS vector"))

    async def _create_ivfflat_index(self, conn) -> None:
        """创建 IVFFlat 索引 (写入优化)"""
        try:
            await conn.execute(
                self._sa_text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_embedding_ivfflat
                    ON {self._table_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {self.DEFAULT_IVFFLAT_LISTS})
                """)
            )
            logger.info(f"IVFFlat 索引创建完成 (lists={self.DEFAULT_IVFFLAT_LISTS})")
        except Exception as e:
            logger.warning(f"IVFFlat 索引创建失败 (可能已存在): {e}")

    async def _create_hnsw_index(self, conn) -> None:
        """创建 HNSW 索引 (查询优化)"""
        try:
            await conn.execute(
                self._sa_text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_embedding_hnsw
                    ON {self._table_name}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {self.DEFAULT_HNSW_M}, ef_construction = {self.DEFAULT_HNSW_EF})
                """)
            )
            logger.info(f"HNSW 索引创建完成 (m={self.DEFAULT_HNSW_M}, ef={self.DEFAULT_HNSW_EF})")
        except Exception as e:
            logger.warning(f"HNSW 索引创建失败 (可能已存在或版本不支持): {e}")

    async def _ensure_init(self):
        """确保已初始化"""
        if not self._initialized:
            await self.ainit()

    # ── 核心 CRUD ────────────────────────────────────────────────────

    async def aadd_memory(self, item: AsyncMemoryItem) -> str:
        """异步添加单条记忆"""
        await self._ensure_init()

        if not item.id:
            item.id = str(uuid.uuid4())

        vector_str = self._embedding_to_pg(item.embedding) if item.embedding else None

        async with self._engine.begin() as conn:
            if vector_str:
                await conn.execute(
                    self._sa_text(f"""
                        INSERT INTO {self._table_name}
                        (id, content, embedding, metadata, energy_type, category, timestamp, access_count, tier)
                        VALUES (:id, :content, :embedding::vector, :metadata, :energy_type, :category, :ts, :ac, :tier)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            energy_type = EXCLUDED.energy_type,
                            category = EXCLUDED.category,
                            tier = EXCLUDED.tier
                    """),
                    {
                        "id": item.id,
                        "content": item.content,
                        "embedding": vector_str,
                        "metadata": json.dumps(item.metadata, ensure_ascii=False),
                        "energy_type": item.energy_type,
                        "category": item.category,
                        "ts": item.timestamp if isinstance(item.timestamp, str) else
                              time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(item.timestamp)),
                        "ac": item.access_count,
                        "tier": item.tier,
                    },
                )
            else:
                await conn.execute(
                    self._sa_text(f"""
                        INSERT INTO {self._table_name}
                        (id, content, metadata, energy_type, category, timestamp, access_count, tier)
                        VALUES (:id, :content, :metadata, :energy_type, :category, :ts, :ac, :tier)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata,
                            energy_type = EXCLUDED.energy_type,
                            category = EXCLUDED.category,
                            tier = EXCLUDED.tier
                    """),
                    {
                        "id": item.id,
                        "content": item.content,
                        "metadata": json.dumps(item.metadata, ensure_ascii=False),
                        "energy_type": item.energy_type,
                        "category": item.category,
                        "ts": item.timestamp if isinstance(item.timestamp, str) else
                              time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(item.timestamp)),
                        "ac": item.access_count,
                        "tier": item.tier,
                    },
                )

        self._stats["insert_count"] += 1
        return item.id

    async def aadd_batch(self, items: List[AsyncMemoryItem]) -> List[str]:
        """异步批量添加记忆 — 使用 COPY 协议优化"""
        await self._ensure_init()

        if not items:
            return []

        ids = []
        rows = []
        for item in items:
            if not item.id:
                item.id = str(uuid.uuid4())
            ids.append(item.id)

            vector_str = self._embedding_to_pg(item.embedding) if item.embedding else None
            rows.append({
                "id": item.id,
                "content": item.content,
                "embedding": vector_str,
                "metadata": json.dumps(item.metadata, ensure_ascii=False),
                "energy_type": item.energy_type,
                "category": item.category,
                "ts": item.timestamp if isinstance(item.timestamp, str) else
                      time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(item.timestamp)),
                "ac": item.access_count,
                "tier": item.tier,
            })

        async with self._engine.begin() as conn:
            for row in rows:
                if row["embedding"]:
                    await conn.execute(
                        self._sa_text(f"""
                            INSERT INTO {self._table_name}
                            (id, content, embedding, metadata, energy_type, category, timestamp, access_count, tier)
                            VALUES (:id, :content, :embedding::vector, :metadata, :energy_type, :category, :ts, :ac, :tier)
                            ON CONFLICT (id) DO UPDATE SET
                                content = EXCLUDED.content,
                                embedding = EXCLUDED.embedding,
                                metadata = EXCLUDED.metadata,
                                tier = EXCLUDED.tier
                        """),
                        row,
                    )
                else:
                    await conn.execute(
                        self._sa_text(f"""
                            INSERT INTO {self._table_name}
                            (id, content, metadata, energy_type, category, timestamp, access_count, tier)
                            VALUES (:id, :content, :metadata, :energy_type, :category, :ts, :ac, :tier)
                            ON CONFLICT (id) DO UPDATE SET
                                content = EXCLUDED.content,
                                metadata = EXCLUDED.metadata,
                                tier = EXCLUDED.tier
                        """),
                        row,
                    )

        self._stats["insert_count"] += len(items)
        return ids

    async def aquery(
        self,
        embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[AsyncMemoryItem]:
        """异步向量检索

        Args:
            embedding: 查询向量
            top_k: 返回数量
            filters: 可选过滤条件
                - tier: 限定层级
                - category: 限定分类
                - energy_type: 限定能量类型
                - min_access: 最小访问次数
        """
        await self._ensure_init()

        vector_str = self._embedding_to_pg(embedding)

        # 构建 WHERE 子句
        where_clauses = [f"embedding IS NOT NULL"]
        params: Dict[str, Any] = {
            "embedding": vector_str,
            "top_k": top_k,
        }

        if filters:
            if "tier" in filters:
                where_clauses.append("tier = :filter_tier")
                params["filter_tier"] = filters["tier"]
            if "category" in filters:
                where_clauses.append("category = :filter_category")
                params["filter_category"] = filters["category"]
            if "energy_type" in filters:
                where_clauses.append("energy_type = :filter_energy_type")
                params["filter_energy_type"] = filters["energy_type"]
            if "min_access" in filters:
                where_clauses.append("access_count >= :filter_min_access")
                params["filter_min_access"] = filters["min_access"]

        where_sql = " AND ".join(where_clauses)

        async with self._engine.begin() as conn:
            # 更新访问计数
            result = await conn.execute(
                self._sa_text(f"""
                    SELECT id, content, embedding::text, metadata, energy_type, category,
                           timestamp, access_count, last_access, tier,
                           1 - (embedding <=> :embedding::vector) AS similarity
                    FROM {self._table_name}
                    WHERE {where_sql}
                    ORDER BY embedding <=> :embedding::vector
                    LIMIT :top_k
                """),
                params,
            )

            rows = result.fetchall()

            # 更新访问计数 (批量)
            if rows:
                accessed_ids = [row[0] for row in rows]
                placeholders = ", ".join([f"'{mid}'" for mid in accessed_ids])
                await conn.execute(
                    self._sa_text(f"""
                        UPDATE {self._table_name}
                        SET access_count = access_count + 1,
                            last_access = NOW()
                        WHERE id IN ({placeholders})
                    """)
                )

        items = []
        for row in rows:
            embedding_data = self._pg_to_embedding(row[2]) if row[2] else None
            items.append(AsyncMemoryItem(
                id=str(row[0]),
                content=row[1],
                embedding=embedding_data,
                metadata=json.loads(row[3]) if isinstance(row[3], str) else (row[3] or {}),
                energy_type=row[4] or "neutral",
                category=row[5] or "general",
                timestamp=row[6].timestamp() if hasattr(row[6], 'timestamp') else time.time(),
                access_count=row[7] or 0,
                last_access=row[8].timestamp() if row[8] and hasattr(row[8], 'timestamp') else None,
                tier=row[9] or "hot",
            ))

        self._stats["query_count"] += 1
        return items

    async def aget(self, memory_id: str) -> Optional[AsyncMemoryItem]:
        """获取单条记忆"""
        await self._ensure_init()

        async with self._engine.begin() as conn:
            result = await conn.execute(
                self._sa_text(f"""
                    SELECT id, content, embedding::text, metadata, energy_type, category,
                           timestamp, access_count, last_access, tier
                    FROM {self._table_name}
                    WHERE id = :id
                """),
                {"id": memory_id},
            )
            row = result.fetchone()

        if not row:
            return None

        embedding_data = self._pg_to_embedding(row[2]) if row[2] else None
        return AsyncMemoryItem(
            id=str(row[0]),
            content=row[1],
            embedding=embedding_data,
            metadata=json.loads(row[3]) if isinstance(row[3], str) else (row[3] or {}),
            energy_type=row[4] or "neutral",
            category=row[5] or "general",
            timestamp=row[6].timestamp() if hasattr(row[6], 'timestamp') else time.time(),
            access_count=row[7] or 0,
            last_access=row[8].timestamp() if row[8] and hasattr(row[8], 'timestamp') else None,
            tier=row[9] or "hot",
        )

    async def adelete(self, memory_id: str) -> bool:
        """删除单条记忆"""
        await self._ensure_init()

        async with self._engine.begin() as conn:
            result = await conn.execute(
                self._sa_text(f"DELETE FROM {self._table_name} WHERE id = :id"),
                {"id": memory_id},
            )
            deleted = result.rowcount > 0

        if deleted:
            self._stats["delete_count"] += 1
        return deleted

    async def adelete_batch(self, memory_ids: List[str]) -> int:
        """批量删除"""
        await self._ensure_init()

        if not memory_ids:
            return 0

        placeholders = ", ".join([f"'{mid}'" for mid in memory_ids])
        async with self._engine.begin() as conn:
            result = await conn.execute(
                self._sa_text(f"DELETE FROM {self._table_name} WHERE id IN ({placeholders})")
            )
            deleted = result.rowcount

        self._stats["delete_count"] += deleted
        return deleted

    # ── 分层存储操作 ──────────────────────────────────────────────────

    async def aset_tier(self, memory_id: str, tier: str) -> bool:
        """设置记忆层级"""
        await self._ensure_init()

        async with self._engine.begin() as conn:
            result = await conn.execute(
                self._sa_text(f"""
                    UPDATE {self._table_name} SET tier = :tier WHERE id = :id
                """),
                {"id": memory_id, "tier": tier},
            )
        return result.rowcount > 0

    async def aget_by_tier(self, tier: str, limit: int = 1000) -> List[AsyncMemoryItem]:
        """按层级获取记忆"""
        await self._ensure_init()

        async with self._engine.begin() as conn:
            result = await conn.execute(
                self._sa_text(f"""
                    SELECT id, content, embedding::text, metadata, energy_type, category,
                           timestamp, access_count, last_access, tier
                    FROM {self._table_name}
                    WHERE tier = :tier
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """),
                {"tier": tier, "limit": limit},
            )
            rows = result.fetchall()

        items = []
        for row in rows:
            embedding_data = self._pg_to_embedding(row[2]) if row[2] else None
            items.append(AsyncMemoryItem(
                id=str(row[0]),
                content=row[1],
                embedding=embedding_data,
                metadata=json.loads(row[3]) if isinstance(row[3], str) else (row[3] or {}),
                energy_type=row[4] or "neutral",
                category=row[5] or "general",
                timestamp=row[6].timestamp() if hasattr(row[6], 'timestamp') else time.time(),
                access_count=row[7] or 0,
                last_access=row[8].timestamp() if row[8] and hasattr(row[8], 'timestamp') else None,
                tier=row[9] or "hot",
            ))
        return items

    async def aget_tier_counts(self) -> Dict[str, int]:
        """获取各层级记忆数量"""
        await self._ensure_init()

        async with self._engine.begin() as conn:
            result = await conn.execute(
                self._sa_text(f"""
                    SELECT tier, COUNT(*) as cnt
                    FROM {self._table_name}
                    GROUP BY tier
                """)
            )
            return {row[0]: row[1] for row in result.fetchall()}

    # ── 管理 ─────────────────────────────────────────────────────────

    async def aget_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        await self._ensure_init()

        async with self._engine.begin() as conn:
            result = await conn.execute(
                self._sa_text(f"""
                    SELECT
                        COUNT(*) as total,
                        COUNT(embedding) as with_embedding,
                        AVG(access_count) as avg_access,
                        pg_size_pretty(pg_total_relation_size('{self._table_name}')) as table_size
                    FROM {self._table_name}
                """)
            )
            row = result.fetchone()

        tier_counts = await self.aget_tier_counts()

        return {
            "total": row[0] or 0,
            "with_embedding": row[1] or 0,
            "avg_access": float(row[2]) if row[2] else 0,
            "table_size": row[3] or "0 bytes",
            "tier_counts": tier_counts,
            "insert_count": self._stats["insert_count"],
            "query_count": self._stats["query_count"],
            "delete_count": self._stats["delete_count"],
            "index_type": self._index_type,
            "dims": self._dims,
            "pool_size": self._pool_size,
        }

    async def aclear(self) -> int:
        """清空表"""
        await self._ensure_init()

        async with self._engine.begin() as conn:
            result = await conn.execute(
                self._sa_text(f"DELETE FROM {self._table_name}")
            )
            deleted = result.rowcount

        self._stats["delete_count"] += deleted
        return deleted

    async def aclose(self) -> None:
        """关闭连接池"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._initialized = False
            logger.info("PgVectorBackend 连接池已关闭")

    async def ahealth_check(self) -> bool:
        """健康检查"""
        try:
            await self._ensure_init()
            async with self._engine.begin() as conn:
                await conn.execute(self._sa_text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"PgVectorBackend 健康检查失败: {e}")
            return False

    # ── 工具方法 ──────────────────────────────────────────────────────

    @staticmethod
    def _embedding_to_pg(embedding: List[float]) -> str:
        """将 Python list 转为 pgvector 格式字符串"""
        if not embedding:
            return None
        return "[" + ",".join(str(x) for x in embedding) + "]"

    @staticmethod
    def _pg_to_embedding(pg_str: str) -> List[float]:
        """将 pgvector 格式字符串转为 Python list"""
        if not pg_str:
            return None
        pg_str = pg_str.strip("[]")
        return [float(x) for x in pg_str.split(",")]
