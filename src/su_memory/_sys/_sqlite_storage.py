"""
SqliteStorageBackend — SQLite 存储后端

基于标准库 sqlite3 的异步存储后端，零额外依赖。
作为默认回退后端，确保 su-memory 在无外部数据库时仍可用。

v3.0.0: StorageBackend 抽象层的默认实现。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
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


class SqliteStorageBackend(StorageBackend):
    """
    SQLite 存储后端。

    基于标准库 sqlite3，零额外依赖。
    支持向量存储 (JSON 序列化) 和元数据索引。

    表结构:
        CREATE TABLE IF NOT EXISTS memories (
            memory_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            embedding TEXT,          -- JSON 序列化的向量
            metadata TEXT,           -- JSON 序列化的元数据
            energy_type TEXT,
            created_at REAL
        )
        CREATE INDEX IF NOT EXISTS idx_energy ON memories(energy_type)
        CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at)
    """

    def __init__(self, config: StorageConfig | None = None):
        super().__init__(config)
        self._conn: sqlite3.Connection | None = None
        self._loop = asyncio.get_event_loop()
        self._db_path: str = ""

    @property
    def backend_type(self) -> BackendType:
        return BackendType.SQLITE

    async def initialize(self) -> bool:
        """初始化 SQLite 连接和表结构"""
        try:
            db_path = self.config.sqlite_path
            if not db_path:
                db_path = os.path.join(
                    os.path.expanduser("~"), ".su_memory", "storage.db"
                )
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self._db_path = db_path

            self._conn = await self._run_sync(
                lambda: sqlite3.connect(db_path, check_same_thread=False)
            )
            self._conn.row_factory = sqlite3.Row

            # 创建表结构
            await self._run_sync(self._create_tables)
            self._initialized = True
            return True
        except Exception:
            self._initialized = False
            return False

    def _create_tables(self) -> None:
        """创建表结构"""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding TEXT,
                metadata TEXT,
                energy_type TEXT,
                created_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_energy ON memories(energy_type);
            CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at);
        """)
        self._conn.commit()

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
        if not self._initialized:
            return False

        try:
            embedding_json = json.dumps(embedding) if embedding else None
            metadata_json = json.dumps(metadata) if metadata else None
            ts = created_at or time.time()

            await self._run_sync(
                lambda: self._conn.execute(
                    """INSERT OR REPLACE INTO memories
                       (memory_id, content, embedding, metadata, energy_type, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (memory_id, content, embedding_json, metadata_json, energy_type, ts)
                ).rowcount
            )
            await self._run_sync(lambda: self._conn.commit())
            return True
        except Exception:
            logger.exception("SqliteStorageBackend.add failed for memory_id=%s", memory_id)
            return False

    async def add_batch(self, memories: list[StorageMemory]) -> list[str]:
        """批量添加记忆（高效批量事务）"""
        if not self._initialized:
            return []

        try:
            def _batch_insert():
                rows = []
                for mem in memories:
                    embedding_json = json.dumps(mem.embedding) if mem.embedding else None
                    metadata_json = json.dumps(mem.metadata) if mem.metadata else None
                    ts = mem.created_at or time.time()
                    rows.append((
                        mem.memory_id, mem.content, embedding_json,
                        metadata_json, mem.energy_type, ts
                    ))

                self._conn.executemany(
                    """INSERT OR REPLACE INTO memories
                       (memory_id, content, embedding, metadata, energy_type, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    rows
                )
                self._conn.commit()
                return [m.memory_id for m in memories]

            return await self._run_sync(_batch_insert)
        except Exception:
            logger.exception("SqliteStorageBackend.add_batch failed")
            return []

    async def query(
        self,
        vector: list[float] | None,
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[StorageMemory]:
        """向量相似度检索（线性扫描 + 余弦相似度）"""
        if not self._initialized:
            return []

        try:
            def _query():
                cursor = self._conn.execute(
                    "SELECT memory_id, content, embedding, metadata, energy_type, created_at FROM memories"
                )
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    emb = json.loads(row["embedding"]) if row["embedding"] else None
                    meta = json.loads(row["metadata"]) if row["metadata"] else {}

                    score = 0.0
                    if vector and emb and len(vector) == len(emb):
                        score = self._cosine_similarity(vector, emb)

                    results.append(StorageMemory(
                        memory_id=row["memory_id"],
                        content=row["content"],
                        embedding=emb,
                        metadata=meta,
                        energy_type=row["energy_type"],
                        created_at=row["created_at"],
                        score=score,
                    ))

                # 过滤
                if filter_expr:
                    results = self._apply_filter(results, filter_expr)

                # 按 score 降序排列
                results.sort(key=lambda x: x.score, reverse=True)
                return results[:top_k]

            return await self._run_sync(_query)
        except Exception:
            logger.exception("SqliteStorageBackend.query failed")
            return []

    def _apply_filter(
        self, memories: list[StorageMemory], filter_expr: str
    ) -> list[StorageMemory]:
        """应用简单过滤表达式 (如 energy_type == 'causal')"""
        try:
            # 简单解析: field == 'value'
            parts = filter_expr.split("==")
            if len(parts) != 2:
                return memories
            field = parts[0].strip()
            value = parts[1].strip().strip("'\"")

            return [
                m for m in memories
                if str(getattr(m, field, None)) == value
            ]
        except Exception:
            logger.exception("SqliteStorageBackend._apply_filter failed")
            return memories

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = (sum(x * x for x in a)) ** 0.5
        norm_b = (sum(x * x for x in b)) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def delete(self, memory_id: str) -> bool:
        """删除指定记忆"""
        if not self._initialized:
            return False
        try:
            await self._run_sync(
                lambda: self._conn.execute(
                    "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
                ).rowcount
            )
            await self._run_sync(lambda: self._conn.commit())
            return True
        except Exception:
            logger.exception("SqliteStorageBackend.delete failed for memory_id=%s", memory_id)
            return False

    async def count(self) -> int:
        """获取记忆总数"""
        if not self._initialized:
            return 0
        try:
            def _count():
                row = self._conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
                return row["cnt"]
            return await self._run_sync(_count)
        except Exception:
            logger.exception("SqliteStorageBackend.count failed")
            return 0

    async def health_check(self) -> BackendHealth:
        """SQLite 健康检查"""
        import time as _time
        start = _time.time()

        try:
            if not self._conn:
                return BackendHealth(
                    available=False,
                    backend_type=BackendType.SQLITE,
                    detail="Not initialized",
                )

            cnt = await self.count()
            latency = (_time.time() - start) * 1000

            return BackendHealth(
                available=True,
                backend_type=BackendType.SQLITE,
                latency_ms=round(latency, 2),
                memory_count=cnt,
                detail=f"SQLite at {self._db_path}",
            )
        except Exception as e:
            logger.exception("SqliteStorageBackend.health_check failed")
            return BackendHealth(
                available=False,
                backend_type=BackendType.SQLITE,
                error=str(e),
            )

    async def close(self) -> None:
        """关闭 SQLite 连接"""
        if self._conn:
            await self._run_sync(lambda: self._conn.close())
            self._conn = None
            self._initialized = False

    async def _run_sync(self, func):
        """在线程池中运行同步函数"""
        # 如果已经在主线程的事件循环中，直接运行
        return await self._loop.run_in_executor(None, func)
