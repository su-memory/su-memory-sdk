"""
分层存储引擎 — TieredStorage (v2.7.0)

管理 hot / warm / cold 三层存储，自动升降级策略。

架构:
    TieredStorage
    ├── hot:  PgVectorBackend (最近 1 万 + 高频访问)     <50ms
    ├── warm: SQLiteBackend  (1万~5万, 中等频率)        <200ms
    └── cold: File Archive   (5万以上, 低频/归档)      按需加载

策略:
    TierPromotionPolicy:
        - 访问频率 > 10次/天 → promote to hot
        - 30天未访问 → demote to cold
        - hot tier 超过 10K → LRU evict to warm
        - warm tier 超过 50K → oldest evict to cold

查询路由:
    TieredQuery:
        - 查询时并行搜索 hot + warm
        - hot 命中率 ≥ 80% 时仅查 hot
        - cold 层仅当 hot+warm 结果 < top_k 时才加载

Example:
    >>> from su_memory.storage.tiered import TieredStorage, TierConfig
    >>> config = TierConfig(
    ...     hot_capacity=10_000,
    ...     warm_capacity=50_000,
    ...     hot_backend=pgvector_backend,
    ...     warm_backend=sqlite_backend,
    ... )
    >>> storage = TieredStorage(config)
    >>> await storage.ainit()
    >>> mid = await storage.aadd_memory(item)
    >>> results = await storage.aquery(embedding, top_k=5)
    >>> await storage.arebalance()  # 定期再平衡
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

from su_memory.storage.base import StorageBackend, AsyncMemoryItem
from su_memory.exceptions import SuMemoryError, ErrorCode

logger = logging.getLogger(__name__)


# =============================================================================
# TierConfig — 分层配置
# =============================================================================

@dataclass
class TierConfig:
    """分层存储配置

    Attributes:
        hot_capacity: hot tier 最大容量
        warm_capacity: warm tier 最大容量
        hot_backend: hot tier 存储后端 (PgVectorBackend)
        warm_backend: warm tier 存储后端 (SQLiteBackend)
        cold_dir: cold tier 归档目录
        access_threshold: 晋升阈值 (访问次数)
        idle_days_demote: 降级阈值 (未访问天数)
        auto_rebalance: 是否自动再平衡
        rebalance_interval: 再平衡间隔 (秒)
    """
    hot_capacity: int = 10_000
    warm_capacity: int = 50_000
    hot_backend: Optional[StorageBackend] = None
    warm_backend: Optional[StorageBackend] = None
    cold_dir: str = "archives/cold"
    access_threshold: int = 10
    idle_days_demote: int = 30
    auto_rebalance: bool = True
    rebalance_interval: int = 3600  # 1 hour


# =============================================================================
# TieredStorage — 分层存储引擎
# =============================================================================

class TieredStorage:
    """分层存储引擎

    管理 hot/warm/cold 三层存储的写入、查询和自动升降级。

    Attributes:
        config: 分层配置
        stats: 运行时统计
    """

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"

    def __init__(self, config: Optional[TierConfig] = None):
        """初始化分层存储

        Args:
            config: 分层配置，默认使用 TierConfig()
        """
        self.config = config or TierConfig()
        self._initialized = False
        self._last_rebalance = 0.0
        self._lock = asyncio.Lock()

        # 统计
        self.stats = {
            "hot_hits": 0,
            "warm_hits": 0,
            "cold_hits": 0,
            "total_queries": 0,
            "promotions": 0,
            "demotions": 0,
        }

    # ── 初始化 ───────────────────────────────────────────────────────

    async def ainit(self) -> None:
        """初始化所有层级"""
        if self._initialized:
            return

        # 初始化 hot backend
        if self.config.hot_backend:
            if hasattr(self.config.hot_backend, 'ainit'):
                await self.config.hot_backend.ainit()
            logger.info(f"Hot tier 初始化: {self.config.hot_backend.name}")

        # 初始化 warm backend
        if self.config.warm_backend:
            if hasattr(self.config.warm_backend, 'ainit'):
                await self.config.warm_backend.ainit()
            logger.info(f"Warm tier 初始化: {self.config.warm_backend.name}")

        # 确保 cold 目录存在
        os.makedirs(self.config.cold_dir, exist_ok=True)

        self._initialized = True
        self._last_rebalance = time.monotonic()
        logger.info(
            f"TieredStorage 初始化完成: "
            f"hot={self.config.hot_capacity}, warm={self.config.warm_capacity}"
        )

    async def _ensure_init(self):
        """确保已初始化"""
        if not self._initialized:
            await self.ainit()

    # ── 核心 CRUD ────────────────────────────────────────────────────

    async def aadd_memory(self, item: AsyncMemoryItem) -> str:
        """添加记忆 — 默认写入 hot tier"""
        await self._ensure_init()

        if not item.id:
            item.id = str(uuid.uuid4())

        # 写入 hot tier
        if self.config.hot_backend:
            item.tier = self.HOT
            mid = await self.config.hot_backend.aadd_memory(item)
        elif self.config.warm_backend:
            item.tier = self.WARM
            mid = await self.config.warm_backend.aadd_memory(item)
        else:
            mid = await self._add_to_cold(item)

        # 检查是否需要再平衡
        if self.config.auto_rebalance:
            await self._maybe_rebalance()

        return mid

    async def aadd_batch(self, items: List[AsyncMemoryItem]) -> List[str]:
        """批量添加 — 批量写入 hot tier"""
        await self._ensure_init()

        if not items:
            return []

        for item in items:
            if not item.id:
                item.id = str(uuid.uuid4())
            item.tier = self.HOT

        if self.config.hot_backend:
            ids = await self.config.hot_backend.aadd_batch(items)
        elif self.config.warm_backend:
            ids = await self.config.warm_backend.aadd_batch(items)
        else:
            ids = []
            for item in items:
                mid = await self._add_to_cold(item)
                ids.append(mid)

        if self.config.auto_rebalance:
            await self._maybe_rebalance()

        return ids

    async def aquery(
        self,
        embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[AsyncMemoryItem]:
        """分层检索 — 优先 hot，fallback warm → cold

        查询策略:
        1. 从 hot tier 查询 top_k 条
        2. 如果 hot 结果不足，从 warm tier 补充
        3. 如果仍不足，从 cold tier 按需加载
        """
        await self._ensure_init()

        self.stats["total_queries"] += 1
        results: List[AsyncMemoryItem] = []
        seen_ids: set = set()

        # 1. Hot tier
        if self.config.hot_backend:
            hot_results = await self.config.hot_backend.aquery(
                embedding, top_k=top_k, filters=filters
            )
            for item in hot_results:
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    results.append(item)
            if hot_results:
                self.stats["hot_hits"] += 1

        # 2. Warm tier (补充)
        if len(results) < top_k and self.config.warm_backend:
            needed = top_k - len(results)
            warm_filters = {**(filters or {}), "tier": self.WARM}
            warm_results = await self.config.warm_backend.aquery(
                embedding, top_k=needed, filters=warm_filters
            )
            for item in warm_results:
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    results.append(item)
            if warm_results:
                self.stats["warm_hits"] += 1

        # 3. Cold tier (按需加载)
        if len(results) < top_k:
            needed = top_k - len(results)
            cold_results = await self._query_cold(embedding, top_k=needed)
            for item in cold_results:
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    results.append(item)
            if cold_results:
                self.stats["cold_hits"] += 1

        return results[:top_k]

    async def aget(self, memory_id: str) -> Optional[AsyncMemoryItem]:
        """获取单条记忆 — 逐层查找"""
        await self._ensure_init()

        # 检查 hot
        if self.config.hot_backend:
            item = await self.config.hot_backend.aget(memory_id)
            if item:
                return item

        # 检查 warm
        if self.config.warm_backend:
            item = await self.config.warm_backend.aget(memory_id)
            if item:
                return item

        # 检查 cold
        return await self._get_from_cold(memory_id)

    async def adelete(self, memory_id: str) -> bool:
        """删除记忆 — 从所有层删除"""
        await self._ensure_init()

        deleted = False
        if self.config.hot_backend:
            deleted = await self.config.hot_backend.adelete(memory_id) or deleted
        if self.config.warm_backend:
            deleted = await self.config.warm_backend.adelete(memory_id) or deleted
        deleted = self._delete_from_cold(memory_id) or deleted
        return deleted

    async def aget_stats(self) -> Dict[str, Any]:
        """获取全局统计"""
        await self._ensure_init()

        hot_stats = {}
        warm_stats = {}
        tier_counts = {}

        if self.config.hot_backend:
            try:
                hot_stats = await self.config.hot_backend.aget_stats()
            except Exception as e:
                hot_stats = {"error": str(e)}

        if self.config.warm_backend:
            try:
                warm_stats = await self.config.warm_backend.aget_stats()
            except Exception as e:
                warm_stats = {"error": str(e)}

        # Tier counts
        if self.config.hot_backend and hasattr(self.config.hot_backend, 'aget_tier_counts'):
            try:
                tier_counts = await self.config.hot_backend.aget_tier_counts()
            except Exception:
                pass

        total_queries = max(self.stats["total_queries"], 1)
        return {
            "hot": hot_stats,
            "warm": warm_stats,
            "tier_counts": tier_counts,
            "runtime_stats": {
                "total_queries": self.stats["total_queries"],
                "hot_hit_rate": self.stats["hot_hits"] / total_queries,
                "warm_hit_rate": self.stats["warm_hits"] / total_queries,
                "cold_hit_rate": self.stats["cold_hits"] / total_queries,
                "promotions": self.stats["promotions"],
                "demotions": self.stats["demotions"],
            },
            "config": {
                "hot_capacity": self.config.hot_capacity,
                "warm_capacity": self.config.warm_capacity,
                "access_threshold": self.config.access_threshold,
                "idle_days_demote": self.config.idle_days_demote,
            },
        }

    async def aclose(self) -> None:
        """关闭所有层"""
        if self.config.hot_backend:
            await self.config.hot_backend.aclose()
        if self.config.warm_backend:
            await self.config.warm_backend.aclose()
        self._initialized = False

    async def ahealth_check(self) -> bool:
        """健康检查"""
        hot_ok = warm_ok = True
        if self.config.hot_backend:
            hot_ok = await self.config.hot_backend.ahealth_check()
        if self.config.warm_backend:
            warm_ok = await self.config.warm_backend.ahealth_check()
        return hot_ok and warm_ok

    async def aclear(self) -> int:
        """清空所有层"""
        count = 0
        if self.config.hot_backend:
            count += await self.config.hot_backend.aclear()
        if self.config.warm_backend:
            count += await self.config.warm_backend.aclear()
        count += self._clear_cold()
        return count

    # ── 分层再平衡 ────────────────────────────────────────────────────

    async def _maybe_rebalance(self) -> None:
        """如果超过再平衡间隔，执行再平衡"""
        now = time.monotonic()
        if now - self._last_rebalance >= self.config.rebalance_interval:
            await self.arebalance()

    async def arebalance(self) -> Dict[str, int]:
        """执行全层再平衡

        返回每层的变更计数。
        """
        if not self._lock.locked():
            async with self._lock:
                return await self._do_rebalance()
        return {"promoted": 0, "demoted": 0}

    async def _do_rebalance(self) -> Dict[str, int]:
        """内部再平衡逻辑"""
        promoted = 0
        demoted = 0

        if not self.config.hot_backend:
            return {"promoted": promoted, "demoted": demoted}

        # 1. Hot → Warm: 超过容量时 LRU 淘汰
        hot_stats = await self.config.hot_backend.aget_stats()
        hot_count = hot_stats.get("total", 0)
        if hot_count > self.config.hot_capacity:
            overflow = hot_count - self.config.hot_capacity
            oldest_in_hot = await self._get_oldest_in_hot(limit=overflow)
            if self.config.warm_backend:
                for item in oldest_in_hot:
                    item.tier = self.WARM
                    await self.config.warm_backend.aadd_memory(item)
                    await self.config.hot_backend.adelete(item.id)
                    demoted += 1
            else:
                for item in oldest_in_hot:
                    await self._archive_to_cold(item)
                    await self.config.hot_backend.adelete(item.id)
                    demoted += 1

        # 2. Warm → Cold: 超过容量时归档最旧
        if self.config.warm_backend:
            warm_stats = await self.config.warm_backend.aget_stats()
            warm_count = warm_stats.get("total", 0)
            if warm_count > self.config.warm_capacity:
                overflow = warm_count - self.config.warm_capacity
                oldest_in_warm = await self._get_oldest_in_warm(limit=overflow)
                for item in oldest_in_warm:
                    await self._archive_to_cold(item)
                    await self.config.warm_backend.adelete(item.id)
                    demoted += 1

        # 3. Warm → Hot: 高频访问晋升
        if self.config.warm_backend and hot_count < self.config.hot_capacity:
            hot_access_items = await self._get_high_access_in_warm(
                threshold=self.config.access_threshold,
                limit=self.config.hot_capacity - hot_count,
            )
            for item in hot_access_items:
                item.tier = self.HOT
                await self.config.hot_backend.aadd_memory(item)
                await self.config.warm_backend.adelete(item.id)
                promoted += 1

        self._last_rebalance = time.monotonic()
        self.stats["promotions"] += promoted
        self.stats["demotions"] += demoted

        logger.info(
            f"再平衡完成: promoted={promoted}, demoted={demoted}, "
            f"hot={hot_count}, warm={warm_count}"
        )
        return {"promoted": promoted, "demoted": demoted}

    async def _get_oldest_in_hot(self, limit: int = 100) -> List[AsyncMemoryItem]:
        """获取 hot tier 中最旧的记忆 (LRU 淘汰候选)"""
        if not self.config.hot_backend:
            return []
        # 默认按 timestamp 升序获取最旧的
        try:
            result = await self.config.hot_backend.aget_by_tier(self.HOT, limit=limit * 2)
            # 按 timestamp 升序排列
            result.sort(key=lambda x: x.timestamp)
            return result[:limit]
        except Exception:
            return []

    async def _get_oldest_in_warm(self, limit: int = 100) -> List[AsyncMemoryItem]:
        """获取 warm tier 中最旧的记忆"""
        if not self.config.warm_backend:
            return []
        try:
            result = await self.config.warm_backend.aget_by_tier(self.WARM, limit=limit * 2)
            result.sort(key=lambda x: x.timestamp)
            return result[:limit]
        except Exception:
            return []

    async def _get_high_access_in_warm(
        self, threshold: int = 10, limit: int = 100
    ) -> List[AsyncMemoryItem]:
        """获取 warm tier 中高频访问的记忆 (晋升候选)"""
        if not self.config.warm_backend:
            return []
        try:
            result = await self.config.warm_backend.aget_by_tier(self.WARM, limit=limit * 2)
            # 按 access_count 降序排列
            result.sort(key=lambda x: x.access_count, reverse=True)
            return [r for r in result[:limit] if r.access_count >= threshold]
        except Exception:
            return []

    # ── Cold tier (文件归档) ──────────────────────────────────────────

    async def _add_to_cold(self, item: AsyncMemoryItem) -> str:
        """写入 cold tier (JSON 文件归档)"""
        os.makedirs(self.config.cold_dir, exist_ok=True)

        filepath = os.path.join(self.config.cold_dir, f"{item.id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(item.to_dict(), f, ensure_ascii=False)

        return item.id

    async def _get_from_cold(self, memory_id: str) -> Optional[AsyncMemoryItem]:
        """从 cold tier 加载单条记忆"""
        filepath = os.path.join(self.config.cold_dir, f"{memory_id}.json")
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return AsyncMemoryItem.from_dict(data)
        except Exception as e:
            logger.warning(f"Cold tier 加载失败 {memory_id}: {e}")
            return None

    async def _query_cold(
        self, embedding: List[float], top_k: int = 10
    ) -> List[AsyncMemoryItem]:
        """从 cold tier 检索 (暴力扫描 + 余弦相似度)"""
        items = []
        cold_dir = self.config.cold_dir
        if not os.path.exists(cold_dir):
            return []

        for filename in os.listdir(cold_dir):
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(cold_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                item = AsyncMemoryItem.from_dict(data)
                if item.embedding:
                    item.metadata['_cold_score'] = self._cosine_similarity(
                        embedding, item.embedding
                    )
                    items.append(item)
            except Exception:
                continue

        # 按相似度排序
        items.sort(key=lambda x: x.metadata.get('_cold_score', 0), reverse=True)
        return items[:top_k]

    def _delete_from_cold(self, memory_id: str) -> bool:
        """从 cold tier 删除"""
        filepath = os.path.join(self.config.cold_dir, f"{memory_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    def _clear_cold(self) -> int:
        """清空 cold tier"""
        count = 0
        cold_dir = self.config.cold_dir
        if os.path.exists(cold_dir):
            for filename in os.listdir(cold_dir):
                if filename.endswith('.json'):
                    try:
                        os.remove(os.path.join(cold_dir, filename))
                        count += 1
                    except Exception:
                        pass
        return count

    async def _archive_to_cold(self, item: AsyncMemoryItem) -> None:
        """将记忆归档到 cold tier"""
        await self._add_to_cold(item)

    # ── 工具方法 ──────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = (sum(x * x for x in a)) ** 0.5
        norm_b = (sum(x * x for x in b)) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @property
    def hot_hit_rate(self) -> float:
        """hot tier 命中率"""
        total = max(self.stats["total_queries"], 1)
        return self.stats["hot_hits"] / total

    @property
    def overall_hit_rate(self) -> float:
        """整体命中率 (hot + warm)"""
        total = max(self.stats["total_queries"], 1)
        return (self.stats["hot_hits"] + self.stats["warm_hits"]) / total

    # ── 异步上下文管理器 ──────────────────────────────────────────────

    async def __aenter__(self):
        await self.ainit()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
        return False
