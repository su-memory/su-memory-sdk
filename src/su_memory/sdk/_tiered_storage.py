"""
su-memory v3.2.0 — Tiered Storage (混合存储策略)

三层自动降级存储:
  L0 (热层): 内存   — 默认，≤ max_memories
  L1 (温层): SQLite — 触发条件: L0 超过 80% 容量，最旧 20% 自动下沉
  L2 (冷层): PG/Redis 预留 — 显式配置

查询时三层联合检索，结果合并去重。

用法:
    from su_memory.sdk._tiered_storage import TieredStorage

    tiered = TieredStorage(storage_dir="/data/su_memory", max_hot=1000)
    tiered.add_hot(memory_dict)       # 添加到热层
    tiered.auto_tier()                # 自动降级（通常在 add 后调用）
    results = tiered.query("关键词")  # 三层联合查询
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any

# ---------------------------------------------------------------------------
# TieredStorage
# ---------------------------------------------------------------------------

class TieredStorage:
    """
    三层混合存储管理器。

    特性:
    - L0 热层：内存 list，最快访问
    - L1 温层：SQLite，触发自动降级时写入
    - 自动降级：L0 超过 80% 上限时，最旧 20% → L1
    - 联合查询：L0 精确检索 + L1 关键词匹配
    - 线程安全
    """

    def __init__(
        self,
        storage_dir: str | None = None,
        max_hot: int = 1000,
        auto_tier_threshold: float = 0.8,
        tier_ratio: float = 0.2,
    ):
        """
        Args:
            storage_dir: 存储目录（None 时使用临时目录）
            max_hot: L0 热层最大容量
            auto_tier_threshold: 触发降级的容量比例（默认 80%）
            tier_ratio: 每次降级的比例（默认 20%）
        """
        self.max_hot = max_hot
        self.auto_tier_threshold = auto_tier_threshold
        self.tier_ratio = tier_ratio

        # L0 热层: memory list
        self._hot: list[dict[str, Any]] = []
        self._hot_ids: set = set()

        # L1 温层: SQLite
        if storage_dir is None:
            import tempfile
            storage_dir = os.path.join(tempfile.gettempdir(), "su_memory_tiered")
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        self._db_path = os.path.join(storage_dir, "tiered_l1.db")
        self._init_db()

        # Thread safety
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Initialize L1 SQLite database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    timestamp REAL DEFAULT 0,
                    tiered_at REAL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_timestamp
                ON memories(timestamp)
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Hot layer (L0)
    # ------------------------------------------------------------------

    def add_hot(self, memory: dict[str, Any]) -> None:
        """Add memory to L0 hot layer."""
        with self._lock:
            if memory["id"] not in self._hot_ids:
                self._hot.append(memory)
                self._hot_ids.add(memory["id"])

    def get_hot(self, memory_id: str) -> dict[str, Any] | None:
        """Get memory from hot layer by id."""
        with self._lock:
            for m in self._hot:
                if m["id"] == memory_id:
                    return m
        return None

    @property
    def hot_count(self) -> int:
        """Number of items in hot layer."""
        return len(self._hot)

    @property
    def hot_fullness(self) -> float:
        """Hot layer fullness ratio (0.0 - 1.0)."""
        if self.max_hot == 0:
            return 1.0
        return len(self._hot) / self.max_hot

    @property
    def needs_tier(self) -> bool:
        """Whether auto-tiering should be triggered."""
        return self.hot_fullness >= self.auto_tier_threshold

    # ------------------------------------------------------------------
    # Warm layer (L1)
    # ------------------------------------------------------------------

    def add_warm(self, memory: dict[str, Any]) -> bool:
        """Add memory to L1 warm layer (SQLite)."""
        try:
            conn = self._get_conn()
            keywords_json = json.dumps(
                memory.get("keywords", []), ensure_ascii=False
            )
            metadata_json = json.dumps(
                memory.get("metadata", {}), ensure_ascii=False
            )
            conn.execute(
                """INSERT OR REPLACE INTO memories
                   (id, content, keywords, metadata_json, timestamp, tiered_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    memory["id"],
                    memory["content"],
                    keywords_json,
                    metadata_json,
                    memory.get("timestamp", time.time()),
                    time.time(),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def get_warm(self, memory_id: str) -> dict[str, Any] | None:
        """Get memory from warm layer by id."""
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            conn.close()
            if row:
                return {
                    "id": row["id"],
                    "content": row["content"],
                    "keywords": json.loads(row["keywords"]),
                    "metadata": json.loads(row["metadata_json"]),
                    "timestamp": row["timestamp"],
                }
        except Exception:
            pass
        return None

    def query_warm(
        self, keywords: list[str], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Query warm layer by keyword matching."""
        if not keywords:
            return []

        try:
            conn = self._get_conn()
            # Build LIKE query for each keyword
            conditions = " OR ".join(
                "content LIKE ?" for _ in keywords
            )
            params = [f"%{kw}%" for kw in keywords]
            sql = f"SELECT * FROM memories WHERE {conditions} LIMIT ?"
            params.append(top_k * 3)  # Retrieve more for dedup

            rows = conn.execute(sql, params).fetchall()
            conn.close()

            results = []
            seen_ids = set()
            seen_contents = set()  # 按 content 去重, 防止同内容不同 id 的冗余副本
            for row in rows:
                if row["id"] in seen_ids:
                    continue
                content = row["content"]
                if content in seen_contents:
                    continue
                seen_ids.add(row["id"])
                seen_contents.add(content)
                results.append({
                    "id": row["id"],
                    "content": content,
                    "keywords": json.loads(row["keywords"]),
                    "metadata": json.loads(row["metadata_json"]),
                    "timestamp": row["timestamp"],
                    "tier": "warm",
                })
            return results[:top_k]
        except Exception:
            return []

    @property
    def warm_count(self) -> int:
        """Number of items in warm layer."""
        try:
            conn = self._get_conn()
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Auto-tiering
    # ------------------------------------------------------------------

    def auto_tier(self) -> int:
        """
        Auto-tier: move oldest 20% from L0 to L1 when L0 exceeds 80%.

        Returns:
            Number of items moved.
        """
        with self._lock:
            if not self.needs_tier:
                return 0

            move_count = max(1, int(len(self._hot) * self.tier_ratio))
            moved = 0

            for _ in range(move_count):
                if not self._hot:
                    break
                oldest = self._hot.pop(0)
                self._hot_ids.discard(oldest["id"])
                if self.add_warm(oldest):
                    moved += 1

            return moved

    def force_tier(self, count: int = None) -> int:
        """
        Force-tier a specific number of items from L0 to L1.

        Args:
            count: Number to move (None = all)

        Returns:
            Number of items moved.
        """
        with self._lock:
            if count is None:
                count = len(self._hot)
            move_count = min(count, len(self._hot))
            moved = 0

            for _ in range(move_count):
                oldest = self._hot.pop(0)
                self._hot_ids.discard(oldest["id"])
                if self.add_warm(oldest):
                    moved += 1

            return moved

    # ------------------------------------------------------------------
    # Unified query
    # ------------------------------------------------------------------

    def query(
        self,
        query_keywords: list[str],
        top_k: int = 5,
        search_warm: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Unified three-layer query.

        L0 (hot) query is done by the caller's TF-IDF index.
        This method searches L1 (warm) and returns candidates for merging.

        Args:
            query_keywords: Query keywords from tokenizer
            top_k: Max results from warm layer
            search_warm: Whether to search warm layer

        Returns:
            Warm layer results (caller merges with hot results).
        """
        if not search_warm:
            return []

        return self.query_warm(query_keywords, top_k)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get tiered storage statistics."""
        return {
            "hot_count": self.hot_count,
            "warm_count": self.warm_count,
            "max_hot": self.max_hot,
            "hot_fullness": round(self.hot_fullness, 3),
            "needs_tier": self.needs_tier,
            "db_path": self._db_path,
        }

    def clear_all(self) -> None:
        """Clear all layers."""
        with self._lock:
            self._hot.clear()
            self._hot_ids.clear()
            try:
                conn = self._get_conn()
                conn.execute("DELETE FROM memories")
                conn.commit()
                conn.close()
            except Exception:
                pass

    def close(self) -> None:
        """No-op (SQLite auto-closes)."""
        pass
