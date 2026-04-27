"""
SQLite存储后端 - 本地数据持久化

提供高性能本地存储，支持向量相似度查询。

Example:
    >>> from su_memory.storage import SQLiteBackend, MemoryItem
    >>> backend = SQLiteBackend("memories.db")
    >>> memory = MemoryItem(
    ...     id="mem_001",
    ...     content="今天学习了Python",
    ...     metadata={"type": "learning"},
    ...     embedding=[0.1, 0.2, 0.3],
    ...     timestamp=time.time()
    ... )
    >>> backend.add_memory(memory)
    >>> results = backend.query("Python", top_k=5)
"""

import sqlite3
import json
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import uuid
import time
import os


@dataclass
class MemoryItem:
    """记忆条目数据类

    Attributes:
        id: 唯一标识符
        content: 记忆内容
        metadata: 元数据字典
        embedding: 向量嵌入（可选）
        timestamp: 时间戳
        causal_links: 因果链关联ID列表
    """
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    timestamp: float = field(default_factory=time.time)
    causal_links: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "timestamp": self.timestamp,
            "causal_links": self.causal_links or []
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryItem":
        """从字典创建实例"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            timestamp=data.get("timestamp", time.time()),
            causal_links=data.get("causal_links", [])
        )


class SQLiteBackend:
    """
    SQLite存储后端 - 性能优化版

    提供本地数据持久化，支持向量相似度查询和全文搜索。
    线程安全，支持多线程并发访问。

    v1.7.0 性能优化:
    - 线程本地连接池复用
    - WAL模式优化
    - 查询缓存 (LRU)
    - 批量插入优化
    - 索引优化

    Attributes:
        db_path: 数据库文件路径
        enable_compression: 是否启用压缩（默认True）

    Example:
        >>> backend = SQLiteBackend("memories.db")
        >>> memory = MemoryItem(
        ...     id="mem_001",
        ...     content="今天学习了Python",
        ...     metadata={"type": "learning"}
        ... )
        >>> backend.add_memory(memory)
        >>> results = backend.query("test", top_k=5)
        >>> backend.vacuum()  # 定期整理数据库
    """

    def __init__(self, db_path: str = "su_memory.db", enable_compression: bool = True, timeout: float = 30.0):
        """初始化SQLite后端

        Args:
            db_path: 数据库文件路径
            enable_compression: 是否启用压缩存储
            timeout: 数据库操作超时时间（秒）
        """
        self._db_path = db_path
        self._enable_compression = enable_compression
        self._timeout = timeout  # P0-3修复：添加超时设置
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._embedding_dim: Optional[int] = None

        # 线程本地连接
        self._local = threading.local()

        # 查询缓存 (LRU)
        self._query_cache: Dict[str, List[Dict]] = {}
        self._cache_size = 256
        self._cache_order: List[str] = []  # LRU顺序

        # 批量缓冲
        self._batch_buffer: List[MemoryItem] = []
        self._batch_size = 100
        self._batch_lock = threading.Lock()

        # 性能统计
        self._stats = {
            "query_count": 0,
            "insert_count": 0,
            "batch_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self._perf_timers = {}

        # 初始化数据库
        self._init_db()

    def _execute_with_retry(self, func, max_retries: int = 3):
        """带重试机制的执行（P0-3修复）"""
        for attempt in range(max_retries):
            try:
                return func()
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                raise

    @contextmanager
    def _get_conn(self):
        """获取线程本地连接（P0-3修复：添加超时和WAL优化）"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(
                self._db_path,
                timeout=self._timeout,  # 使用可配置超时
                check_same_thread=False
            )
            conn.execute("PRAGMA busy_timeout = 30000")  # 30秒忙等待
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB缓存
            conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn = conn
        yield self._local.conn

    def _init_db(self):
        """初始化数据库表结构"""
        # 确保目录存在
        db_dir = os.path.dirname(self._db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        # 使用独立连接初始化
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")
        conn.execute("PRAGMA temp_store=MEMORY")
        self._conn = conn

        # 主记忆表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB,
                timestamp REAL,
                causal_links TEXT,
                compressed INTEGER DEFAULT 0,
                importance REAL DEFAULT 1.0
            )
        """)

        # 索引优化
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON memories(timestamp DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_importance
            ON memories(importance DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metadata_tags
            ON memories(metadata)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp_importance
            ON memories(timestamp DESC, importance DESC)
        """)

        # 全文搜索虚拟表（使用独立 FTS 表，不使用 content sync）
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id UNINDEXED,
                content
            )
        """)

    def _rowid_to_id(self, rowid: int) -> Optional[str]:
        """通过rowid获取id"""
        cursor = self._conn.execute(
            "SELECT id FROM memories WHERE rowid = ?", (rowid,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def add_memory(self, memory: MemoryItem) -> str:
        """添加或更新记忆

        Args:
            memory: MemoryItem实例

        Returns:
            记忆ID
        """
        start = time.perf_counter()

        with self._get_conn() as conn:
            if not memory.id:
                memory.id = f"mem_{uuid.uuid4().hex[:12]}"

            embedding_blob = None
            if memory.embedding:
                embedding_blob = np.array(memory.embedding).tobytes()
                if self._embedding_dim is None:
                    self._embedding_dim = len(memory.embedding)

            importance = memory.metadata.get("importance", 1.0) if memory.metadata else 1.0

            conn.execute("""
                INSERT OR REPLACE INTO memories
                (id, content, metadata, embedding, timestamp, causal_links, compressed, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.content,
                json.dumps(memory.metadata, ensure_ascii=False),
                embedding_blob,
                memory.timestamp,
                json.dumps(memory.causal_links or [], ensure_ascii=False),
                0,
                importance,
            ))

            # 更新FTS索引
            conn.execute("""
                INSERT OR REPLACE INTO memories_fts (id, content)
                VALUES (?, ?)
            """, (memory.id, memory.content))

            # 清空查询缓存
            self._query_cache.clear()
            self._cache_order.clear()

        # 性能统计
        self._stats["insert_count"] += 1
        self._perf_timers["_last_insert_time"] = time.perf_counter() - start

        return memory.id

    def add_memory_batch(self, memories: List[MemoryItem]) -> List[str]:
        """批量添加记忆 - 优化版

        Args:
            memories: MemoryItem列表

        Returns:
            添加的记忆ID列表
        """
        start = time.perf_counter()

        with self._get_conn() as conn:
            ids = []
            for memory in memories:
                if not memory.id:
                    memory.id = f"mem_{uuid.uuid4().hex[:12]}"
                ids.append(memory.id)

            # 使用executemany提高性能
            data = []
            fts_data = []
            for memory in memories:
                embedding_blob = None
                if memory.embedding:
                    embedding_blob = np.array(memory.embedding).tobytes()
                    if self._embedding_dim is None:
                        self._embedding_dim = len(memory.embedding)

                importance = memory.metadata.get("importance", 1.0) if memory.metadata else 1.0

                data.append((
                    memory.id,
                    memory.content,
                    json.dumps(memory.metadata, ensure_ascii=False),
                    embedding_blob,
                    memory.timestamp,
                    json.dumps(memory.causal_links or [], ensure_ascii=False),
                    0,
                    importance,
                ))
                fts_data.append((memory.id, memory.content))

            # 批量插入优化 - 使用事务
            try:
                conn.execute("BEGIN")
                conn.executemany("""
                    INSERT OR REPLACE INTO memories
                    (id, content, metadata, embedding, timestamp, causal_links, compressed, importance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, data)

                conn.executemany("""
                    INSERT OR REPLACE INTO memories_fts (id, content)
                    VALUES (?, ?)
                """, fts_data)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        # 清空查询缓存（数据已变化）
        self._query_cache.clear()
        self._cache_order.clear()

        # 性能统计
        self._stats["batch_count"] += 1
        self._stats["insert_count"] += len(memories)
        self._perf_timers["_last_batch_time"] = time.perf_counter() - start

        return ids

    def _update_cache(self, key: str, value: List[Dict]):
        """更新LRU缓存"""
        if key in self._query_cache:
            # 更新已有项
            self._cache_order.remove(key)
        elif len(self._query_cache) >= self._cache_size:
            # LRU淘汰
            oldest = self._cache_order.pop(0)
            del self._query_cache[oldest]

        self._query_cache[key] = value
        self._cache_order.append(key)

    def query(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """查询记忆（使用全文搜索 + 查询缓存）

        Args:
            query_text: 查询文本
            top_k: 返回结果数量

        Returns:
            匹配的记亿列表，按相关性排序
        """
        start = time.perf_counter()

        # 检查缓存
        cache_key = f"{query_text}:{top_k}"
        if cache_key in self._query_cache:
            self._stats["cache_hits"] += 1
            self._perf_timers["_last_query_time"] = time.perf_counter() - start
            return self._query_cache[cache_key]

        self._stats["cache_misses"] += 1

        with self._get_conn() as conn:
            # 使用子查询方式避免 bm25 上下文问题
            cursor = conn.execute("""
                SELECT m.id, m.content, m.metadata, m.timestamp, m.importance
                FROM memories m
                WHERE m.id IN (
                    SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?
                )
                ORDER BY m.importance DESC, m.timestamp DESC
                LIMIT ?
            """, (query_text, top_k * 2))  # 多取一些

            results = []
            seen_ids = set()
            for row in cursor.fetchall():
                if row[0] not in seen_ids:
                    seen_ids.add(row[0])
                    results.append({
                        "id": row[0],
                        "content": row[1],
                        "metadata": json.loads(row[2]) if row[2] else {},
                        "timestamp": row[3],
                        "importance": row[4],
                        "score": row[4],  # 使用 importance 作为相关性分数
                    })
                    if len(results) >= top_k:
                        break

            # 如果没有FTS结果，降级为LIKE查询
            if not results:
                cursor = conn.execute("""
                    SELECT id, content, metadata, timestamp, importance
                    FROM memories
                    WHERE content LIKE ?
                    ORDER BY importance DESC, timestamp DESC
                    LIMIT ?
                """, (f"%{query_text}%", top_k))

                for row in cursor.fetchall():
                    results.append({
                        "id": row[0],
                        "content": row[1],
                        "metadata": json.loads(row[2]) if row[2] else {},
                        "timestamp": row[3],
                        "importance": row[4],
                        "score": row[4],
                    })

        # 更新缓存
        self._update_cache(cache_key, results)

        # 性能统计
        self._stats["query_count"] += 1
        self._perf_timers["_last_query_time"] = time.perf_counter() - start

        return results

    def search(self, filters: Dict, top_k: int = 100) -> List[MemoryItem]:
        """条件搜索

        Args:
            filters: 搜索过滤器
                - keywords: 关键词列表（AND关系）
                - start_time: 开始时间戳
                - end_time: 结束时间戳
                - tags: 元数据标签列表
            top_k: 返回结果数量

        Returns:
            匹配的MemoryItem列表
        """
        with self._get_conn() as conn:
            conditions = []
            params = []

            # 关键词过滤
            keywords = filters.get("keywords", [])
            if keywords:
                keyword_conditions = " AND ".join(["content LIKE ?" for _ in keywords])
                conditions.append(f"({keyword_conditions})")
                params.extend([f"%{kw}%" for kw in keywords])

            # 时间范围过滤
            start_time = filters.get("start_time")
            if start_time is not None:
                conditions.append("timestamp >= ?")
                params.append(start_time)

            end_time = filters.get("end_time")
            if end_time is not None:
                conditions.append("timestamp <= ?")
                params.append(end_time)

            # 构建查询
            where_clause = " AND ".join(conditions) if conditions else "1=1"

            cursor = conn.execute(f"""
                SELECT id, content, metadata, embedding, timestamp, causal_links
                FROM memories
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """, (*params, top_k))

            results = []
            for row in cursor.fetchall():
                embedding = None
                if row[3]:
                    embedding = np.frombuffer(row[3], dtype=np.float32).tolist()

                results.append(MemoryItem(
                    id=row[0],
                    content=row[1],
                    metadata=json.loads(row[2]) if row[2] else {},
                    embedding=embedding,
                    timestamp=row[4],
                    causal_links=json.loads(row[5]) if row[5] else []
                ))

            return results

    def search_by_vector(self, query_vector: List[float], top_k: int = 10) -> List[Dict]:
        """向量相似度搜索

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量

        Returns:
            按相似度排序的记忆列表
        """
        if self._embedding_dim is None:
            return []

        with self._get_conn() as conn:
            # 获取所有记忆及其向量
            cursor = conn.execute("""
                SELECT id, content, metadata, timestamp, embedding
                FROM memories
                WHERE embedding IS NOT NULL
            """)

            results = []
            query_arr = np.array(query_vector, dtype=np.float32)

            # P0-2修复：确保向量维度一致
            if self._embedding_dim is not None and len(query_arr) != self._embedding_dim:
                if len(query_arr) < self._embedding_dim:
                    query_arr = np.pad(query_arr, (0, self._embedding_dim - len(query_arr)))
                else:
                    query_arr = query_arr[:self._embedding_dim]

            query_norm = np.linalg.norm(query_arr)

            for row in cursor.fetchall():
                if row[4]:
                    embedding = np.frombuffer(row[4], dtype=np.float32)

                    # 确保存储的向量维度一致
                    if len(embedding) != self._embedding_dim and self._embedding_dim is not None:
                        if len(embedding) < self._embedding_dim:
                            embedding = np.pad(embedding, (0, self._embedding_dim - len(embedding)))
                        else:
                            embedding = embedding[:self._embedding_dim]

                    # 计算余弦相似度
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        similarity = np.dot(query_arr, embedding) / (query_norm * norm)
                        results.append({
                            "id": row[0],
                            "content": row[1],
                            "metadata": json.loads(row[2]) if row[2] else {},
                            "timestamp": row[3],
                            "score": float(similarity)
                        })

            # 按相似度排序
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """获取单个记忆

        Args:
            memory_id: 记忆ID

        Returns:
            MemoryItem或None
        """
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT id, content, metadata, embedding, timestamp, causal_links
                FROM memories
                WHERE id = ?
            """, (memory_id,))

            row = cursor.fetchone()
            if not row:
                return None

            embedding = None
            if row[3]:
                embedding = np.frombuffer(row[3], dtype=np.float32).tolist()

            return MemoryItem(
                id=row[0],
                content=row[1],
                metadata=json.loads(row[2]) if row[2] else {},
                embedding=embedding,
                timestamp=row[4],
                causal_links=json.loads(row[5]) if row[5] else []
            )

    def delete(self, memory_id: str) -> bool:
        """删除记忆

        Args:
            memory_id: 记忆ID

        Returns:
            是否成功删除
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            # 同时删除FTS索引
            conn.execute(
                "DELETE FROM memories_fts WHERE id = ?", (memory_id,)
            )

            # 清空查询缓存
            self._query_cache.clear()
            self._cache_order.clear()

            return cursor.rowcount > 0

    def delete_batch(self, memory_ids: List[str]) -> int:
        """批量删除记忆

        Args:
            memory_ids: 记忆ID列表

        Returns:
            删除的数量
        """
        with self._get_conn() as conn:
            placeholders = ",".join(["?" for _ in memory_ids])
            cursor = conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",
                memory_ids
            )
            conn.execute(
                f"DELETE FROM memories_fts WHERE id IN ({placeholders})",
                memory_ids
            )

            # 清空查询缓存
            self._query_cache.clear()
            self._cache_order.clear()

            return cursor.rowcount

    def get_all(self, limit: int = 1000, offset: int = 0) -> List[MemoryItem]:
        """获取所有记忆

        Args:
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            MemoryItem列表
        """
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT id, content, metadata, embedding, timestamp, causal_links
                FROM memories
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

            results = []
            for row in cursor.fetchall():
                embedding = None
                if row[3]:
                    embedding = np.frombuffer(row[3], dtype=np.float32).tolist()

                results.append(MemoryItem(
                    id=row[0],
                    content=row[1],
                    metadata=json.loads(row[2]) if row[2] else {},
                    embedding=embedding,
                    timestamp=row[4],
                    causal_links=json.loads(row[5]) if row[5] else []
                ))

            return results

    def get_stats(self) -> Dict:
        """获取统计信息

        Returns:
            统计信息字典
        """
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_count,
                    SUM(LENGTH(content)) as total_content_size,
                    SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as embedded_count,
                    MIN(timestamp) as oldest_timestamp,
                    MAX(timestamp) as newest_timestamp
                FROM memories
            """)
            row = cursor.fetchone()

        self._stats["cache_hits"] + self._stats["cache_misses"]

        return {
            "count": row[0] or 0,
            "total_content_size": row[1] or 0,
            "embedded_count": row[2] or 0,
            "oldest_timestamp": row[3],
            "newest_timestamp": row[4],
            "embedding_dim": self._embedding_dim,
            "db_path": self._db_path,
            **self.get_performance_stats(),
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息

        Returns:
            性能统计字典
        """
        total_cache = self._stats["cache_hits"] + self._stats["cache_misses"]
        cache_hit_rate = (
            self._stats["cache_hits"] / max(1, total_cache)
        )

        return {
            "query_count": self._stats["query_count"],
            "insert_count": self._stats["insert_count"],
            "batch_count": self._stats["batch_count"],
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "cache_size": len(self._query_cache),
            "cache_hit_rate": cache_hit_rate,
            "avg_query_time": self._perf_timers.get("_last_query_time", 0),
            "avg_batch_time": self._perf_timers.get("_last_batch_time", 0),
        }

    def clear_cache(self):
        """清空查询缓存"""
        self._query_cache.clear()
        self._cache_order.clear()

    def get_db_size(self) -> int:
        """获取数据库文件大小（字节）"""
        if os.path.exists(self._db_path):
            return os.path.getsize(self._db_path)
        return 0

    def vacuum(self):
        """数据库整理（P0-3修复：VACUUM不能在事务内执行）"""
        conn = sqlite3.connect(self._db_path)
        conn.execute("VACUUM")
        conn.close()

    def optimize(self):
        """优化数据库"""
        with self._get_conn() as conn:
            conn.execute("PRAGMA optimize")

    def close(self):
        """关闭数据库连接"""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
