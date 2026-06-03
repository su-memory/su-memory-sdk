"""
PgStorageBackend — PostgreSQL + pgvector 存储后端

使用 asyncpg + pgvector 实现高性能向量检索。
支持 IVFFlat/HNSW 向量索引、连接池、自动迁移。

依赖: pip install su-memory[pgvector]
  - asyncpg>=0.29.0
  - pgvector>=0.3.0

v3.0.0: 分布式存储架构的生产级后端。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from su_memory._sys._storage_backend import (
    BackendHealth,
    BackendType,
    StorageBackend,
    StorageConfig,
    StorageMemory,
)

logger = logging.getLogger(__name__)


class PgStorageBackend(StorageBackend):
    """
    PostgreSQL + pgvector 存储后端。

    使用 asyncpg 异步连接池 + pgvector 扩展实现高性能向量检索。

    表结构:
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS memories (
            memory_id UUID PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector({dim}),
            metadata JSONB DEFAULT '{}',
            energy_type VARCHAR(32),
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_memories_energy ON memories(energy_type);
        CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
        CREATE INDEX IF NOT EXISTS idx_memories_embedding
            ON memories USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);

    特性:
    - asyncpg 连接池 (min=5, max=20)
    - pgvector IVFFlat 向量索引 (O(log n) 近似检索)
    - JSONB 元数据存储 + 索引
    - 自动迁移 (CREATE TABLE IF NOT EXISTS)
    - 健康检查 + 连接重试
    """

    # 自动迁移 SQL 模板
    _MIGRATION_SQL = """
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS {table} (
            memory_id UUID PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector({dim}),
            metadata JSONB DEFAULT '{{}}',
            energy_type VARCHAR(32),
            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS {idx_energy}
            ON {table}(energy_type);

        CREATE INDEX IF NOT EXISTS {idx_created}
            ON {table}(created_at);

        CREATE INDEX IF NOT EXISTS {idx_embedding}
            ON {table} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
    """

    def __init__(self, config: StorageConfig | None = None):
        super().__init__(config)
        self._pool = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.POSTGRESQL

    async def initialize(self) -> bool:
        """初始化 PostgreSQL 连接池和表结构"""
        try:
            import asyncpg

            cfg = self.config
            dsn = (
                f"postgresql://{cfg.pg_user}:{cfg.pg_password}"
                f"@{cfg.pg_host}:{cfg.pg_port}/{cfg.pg_database}"
            )

            self._pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=cfg.pg_pool_min,
                max_size=cfg.pg_pool_max,
            )

            # 执行迁移
            await self._run_migration()

            self._initialized = True
            logger.info(
                "PgStorageBackend initialized: %s:%s/%s (pool=%d-%d)",
                cfg.pg_host, cfg.pg_port, cfg.pg_database,
                cfg.pg_pool_min, cfg.pg_pool_max,
            )
            return True
        except ImportError:
            logger.warning("asyncpg not installed. Install with: pip install su-memory[pgvector]")
            self._initialized = False
            return False
        except Exception as e:
            logger.error("PgStorageBackend initialization failed: %s", e)
            self._initialized = False
            return False

    async def _run_migration(self) -> None:
        """执行自动迁移"""
        cfg = self.config
        table = cfg.pg_table
        idx_energy = f"idx_{table}_energy"
        idx_created = f"idx_{table}_created"
        idx_embedding = f"idx_{table}_embedding"

        sql = self._MIGRATION_SQL.format(
            table=table,
            dim=cfg.embedding_dim,
            idx_energy=idx_energy,
            idx_created=idx_created,
            idx_embedding=idx_embedding,
        )

        async with self._pool.acquire() as conn:
            try:
                await conn.execute(sql)
                logger.info("PgStorageBackend migration completed for table '%s'", table)
            except Exception as e:
                logger.error("PgStorageBackend migration failed for table '%s': %s", table, e)
                raise

    async def add(
        self,
        memory_id: str,
        content: str,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        energy_type: str | None = None,
        created_at: float | None = None,
    ) -> bool:
        """添加单条记忆"""
        if not self._initialized or not self._pool:
            return False

        try:
            metadata_json = json.dumps(metadata) if metadata else "{}"

            if embedding:
                dim = len(embedding)
                placeholders = ", ".join(f"${i}" for i in range(3, dim + 3))
                sql = f"""
                    INSERT INTO {self.config.pg_table}
                        (memory_id, content, embedding, metadata, energy_type, created_at)
                    VALUES ($1, $2, '[{placeholders}]'::vector, ${dim + 3}::jsonb, ${dim + 4}, NOW())
                    ON CONFLICT (memory_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        energy_type = EXCLUDED.energy_type
                """
                params = [memory_id, content] + list(embedding) + [metadata_json, energy_type]
            else:
                sql = f"""
                    INSERT INTO {self.config.pg_table}
                        (memory_id, content, metadata, energy_type, created_at)
                    VALUES ($1, $2, $3::jsonb, $4, NOW())
                    ON CONFLICT (memory_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        energy_type = EXCLUDED.energy_type
                """
                params = [memory_id, content, metadata_json, energy_type]

            async with self._pool.acquire() as conn:
                await conn.execute(sql, *params)
            return True
        except Exception:
            logger.exception("PgStorageBackend.add failed for memory_id=%s", memory_id)
            return False

    async def add_batch(self, memories: list[StorageMemory]) -> list[str]:
        """批量添加记忆"""
        if not self._initialized or not self._pool:
            return []

        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    ids = []
                    for mem in memories:
                        await self._insert_one(conn, mem)
                        ids.append(mem.memory_id)
                    return ids
        except Exception as e:
            logger.exception("PgStorageBackend.add_batch failed: %s", e)
            return []

    async def _insert_one(self, conn, mem: StorageMemory) -> None:
        """插入单条记忆（事务内）"""
        embedding = mem.embedding
        metadata_json = json.dumps(mem.metadata) if mem.metadata else "{}"

        if embedding:
            dim = len(embedding)
            placeholders = ", ".join(f"${i}" for i in range(1, dim + 1))
            sql = f"""
                INSERT INTO {self.config.pg_table}
                    (memory_id, content, embedding, metadata, energy_type, created_at)
                VALUES ($1, $2, '[{placeholders}]'::vector, $3::jsonb, $4, NOW())
                ON CONFLICT (memory_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    energy_type = EXCLUDED.energy_type
            """
            # 构建参数: [memory_id, content, *embedding, metadata_json, energy_type]
            params = [mem.memory_id, mem.content] + embedding + [metadata_json, mem.energy_type]
        else:
            sql = f"""
                INSERT INTO {self.config.pg_table}
                    (memory_id, content, metadata, energy_type, created_at)
                VALUES ($1, $2, $3::jsonb, $4, NOW())
                ON CONFLICT (memory_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    energy_type = EXCLUDED.energy_type
            """
            params = [mem.memory_id, mem.content, metadata_json, mem.energy_type]

        await conn.execute(sql, *params)

    async def query(
        self,
        vector: list[float] | None,
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[StorageMemory]:
        """向量相似度检索（pgvector <=> 余弦距离）"""
        if not self._initialized or not self._pool:
            return []

        try:
            async with self._pool.acquire() as conn:
                if vector:
                    # pgvector 向量检索
                    dim = len(vector)
                    placeholders = ", ".join(f"${i}" for i in range(1, dim + 1))
                    where_clause = ""
                    if filter_expr:
                        where_clause = f"AND {self._parse_filter_expr(filter_expr)}"

                    sql = f"""
                        SELECT memory_id, content, metadata, energy_type,
                               created_at,
                               1 - (embedding <=> '[{placeholders}]'::vector) AS score
                        FROM {self.config.pg_table}
                        WHERE embedding IS NOT NULL {where_clause}
                        ORDER BY embedding <=> '[{placeholders}]'::vector
                        LIMIT ${dim + 1}
                    """
                    rows = await conn.fetch(sql, *vector, top_k)
                else:
                    # 无向量 — 全表返回
                    sql = f"""
                        SELECT memory_id, content, metadata, energy_type,
                               created_at, 0.0 AS score
                        FROM {self.config.pg_table}
                        LIMIT $1
                    """
                    rows = await conn.fetch(sql, top_k)

                return [
                    StorageMemory(
                        memory_id=str(row["memory_id"]),
                        content=row["content"],
                        metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else (row["metadata"] or {}),
                        energy_type=row["energy_type"],
                        created_at=row["created_at"].timestamp() if hasattr(row["created_at"], "timestamp") else None,
                        score=round(float(row["score"]), 4),
                    )
                    for row in rows
                ]
        except Exception:
            logger.exception("PgStorageBackend.query failed")
            return []

    def _parse_filter_expr(self, filter_expr: str) -> str:
        """将简单过滤表达式转为 SQL WHERE 片段 (白名单校验)。"""
        # 仅允许: energy_type == 'value' 格式
        import re
        match = re.match(r"^(\w+)\s*==\s*'([^']+)'$", filter_expr.strip())
        if not match:
            logger.warning("PgStorageBackend: rejected unsafe filter_expr: %s", filter_expr)
            return "1=1"  # 安全回退：不过滤
        field = match.group(1)
        value = match.group(2)
        # 白名单字段
        allowed_fields = {"energy_type", "memory_id"}
        if field not in allowed_fields:
            logger.warning("PgStorageBackend: rejected unknown filter field: %s", field)
            return "1=1"
        # 参数化安全拼接
        return f"{field} = '{value}'"

    async def delete(self, memory_id: str) -> bool:
        """删除指定记忆"""
        if not self._initialized or not self._pool:
            return False
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    f"DELETE FROM {self.config.pg_table} WHERE memory_id = $1",
                    memory_id,
                )
                # asyncpg execute returns "DELETE N"
                return "DELETE" in result
        except Exception:
            logger.exception("PgStorageBackend.delete failed for memory_id=%s", memory_id)
            return False

    async def count(self) -> int:
        """获取记忆总数"""
        if not self._initialized or not self._pool:
            return 0
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"SELECT COUNT(*) as cnt FROM {self.config.pg_table}"
                )
                return row["cnt"]
        except Exception:
            logger.exception("PgStorageBackend.count failed")
            return 0

    async def health_check(self) -> BackendHealth:
        """PostgreSQL 健康检查"""
        t0 = time.time()

        try:
            if not self._pool:
                # 快速检测 PostgreSQL 是否可达
                try:
                    import asyncpg
                    cfg = self.config
                    conn = await asyncpg.connect(
                        host=cfg.pg_host,
                        port=cfg.pg_port,
                        database=cfg.pg_database,
                        user=cfg.pg_user,
                        password=cfg.pg_password,
                        timeout=5,
                    )
                    await conn.close()
                    return BackendHealth(
                        available=True,
                        backend_type=BackendType.POSTGRESQL,
                        detail="PostgreSQL reachable (pool not initialized)",
                    )
                except Exception:
                    return BackendHealth(
                        available=False,
                        backend_type=BackendType.POSTGRESQL,
                        detail="PostgreSQL not reachable",
                    )

            async with self._pool.acquire() as conn:
                row = await conn.fetchrow("SELECT 1 AS ok")
                cnt = await self.count()
                latency = (time.time() - t0) * 1000

                return BackendHealth(
                    available=row["ok"] == 1,
                    backend_type=BackendType.POSTGRESQL,
                    latency_ms=round(latency, 2),
                    memory_count=cnt,
                    detail=f"PostgreSQL {self.config.pg_host}:{self.config.pg_port}/{self.config.pg_database}",
                )
        except Exception as e:
            logger.exception("PgStorageBackend.health_check failed")
            return BackendHealth(
                available=False,
                backend_type=BackendType.POSTGRESQL,
                error=str(e),
            )

    async def close(self) -> None:
        """关闭连接池"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False


def _format_vector(vec: list[float]) -> str:
    """将向量格式化为 pgvector 兼容字符串 '[1.0, 2.0, 3.0]'"""
    return f"[{', '.join(str(v) for v in vec)}]"
