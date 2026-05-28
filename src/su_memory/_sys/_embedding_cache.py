"""
su-memory 嵌入缓存 (Embedding Cache)

LFU (Least Frequently Used) + TTL (Time-To-Live) 策略的嵌入向量缓存。
将文本到向量的转换结果缓存起来，避免重复调用嵌入服务。

核心指标:
- 命中率 > 90% (对高频重复查询)
- 写操作: O(1) 平均
- 读操作: O(1) 平均
- 内存开销可控 (max_entries 限制)

策略:
- LFU: 按访问频率淘汰，低频数据优先驱逐
- TTL: 每个条目有过期时间，过期自动失效
- Lazy eviction: 在访问时检查过期，不主动扫描

使用方式:
    from su_memory._sys._embedding_cache import EmbeddingCache
    cache = EmbeddingCache(max_entries=10000, ttl_seconds=3600)

    # 带缓存嵌入
    vec = cache.get_or_compute("hello world", lambda: embed("hello world"))
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    access_count: int = 0
    created_at: float = field(default_factory=time.monotonic)
    last_access: float = field(default_factory=time.monotonic)

    def touch(self):
        """记录一次访问"""
        self.access_count += 1
        self.last_access = time.monotonic()


class EmbeddingCache:
    """LFU + TTL 嵌入缓存

    内部结构:
    - _store: Dict[key, CacheEntry] — O(1) 读写
    - _lfu_bins: Dict[freq, OrderedDict[key, None]] — 按频率分组
    - _min_freq: 当前最低频率 (eviction 用)
    """

    def __init__(
        self,
        max_entries: int = 10000,
        ttl_seconds: float = 3600.0,
        name: str = "embedding",
    ):
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._name = name
        self._lock = threading.RLock()

        self._store: Dict[str, CacheEntry] = {}
        self._lfu_bins: Dict[int, OrderedDict] = {}
        self._min_freq: int = 0

        # 统计
        self._hits: int = 0
        self._misses: int = 0

    @staticmethod
    def _hash_text(text: str) -> str:
        """文本哈希作为缓存键"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[Any]:
        """获取缓存值，无缓存返回 None"""
        key = self._hash_text(text)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            # 检查 TTL
            if self._ttl_seconds > 0:
                age = time.monotonic() - entry.created_at
                if age > self._ttl_seconds:
                    self._remove(key, entry)
                    self._misses += 1
                    return None

            # 命中：更新频率统计
            self._hits += 1
            self._update_frequency(key, entry)
            return entry.value

    def set(self, text: str, value: Any):
        """写入缓存"""
        key = self._hash_text(text)
        with self._lock:
            # 已存在 → 更新
            if key in self._store:
                entry = self._store[key]
                entry.value = value
                entry.created_at = time.monotonic()  # 重置 TTL
                entry.touch()
                self._update_frequency(key, entry)
                return

            # 容量控制
            if len(self._store) >= self._max_entries:
                self._evict_lfu()

            # 新条目
            entry = CacheEntry(key=key, value=value)
            self._store[key] = entry
            self._add_to_freq_bin(key, 1)
            if self._min_freq == 0 or 1 < self._min_freq:
                self._min_freq = 1

    def get_or_compute(self, text: str, compute_fn: Callable[[], Any]) -> Any:
        """获取缓存，未命中时调用 compute_fn 计算

        Args:
            text: 文本
            compute_fn: 无参回调，返回嵌入向量

        Returns:
            嵌入向量
        """
        result = self.get(text)
        if result is not None:
            return result
        # 未命中：计算并缓存
        result = compute_fn()
        self.set(text, result)
        return result

    def _update_frequency(self, key: str, entry: CacheEntry):
        """更新 LFU 频率"""
        old_freq = entry.access_count
        entry.touch()
        new_freq = entry.access_count

        # 从旧频率箱移除
        if old_freq in self._lfu_bins:
            self._lfu_bins[old_freq].pop(key, None)
            if not self._lfu_bins[old_freq] and old_freq == self._min_freq:
                self._min_freq += 1

        # 加到新频率箱
        self._add_to_freq_bin(key, new_freq)

    def _add_to_freq_bin(self, key: str, freq: int):
        """添加条目到频率箱"""
        if freq not in self._lfu_bins:
            self._lfu_bins[freq] = OrderedDict()
        self._lfu_bins[freq][key] = None

    def _evict_lfu(self):
        """LFU 驱逐：移除最低频率箱中最老的条目"""
        if not self._lfu_bins or self._min_freq not in self._lfu_bins:
            # 回退：移除整个缓存中最老的
            if self._store:
                oldest_key = next(iter(self._store))
                self._remove(oldest_key, self._store[oldest_key])
            return

        freq_bin = self._lfu_bins[self._min_freq]
        if freq_bin:
            evict_key, _ = freq_bin.popitem(last=False)  # FIFO within same freq
            if evict_key in self._store:
                self._remove(evict_key, self._store[evict_key])

        # 如果当前 min_freq 箱空了，找下一个
        while self._min_freq in self._lfu_bins and not self._lfu_bins[self._min_freq]:
            del self._lfu_bins[self._min_freq]
            self._min_freq += 1

    def _remove(self, key: str, entry: CacheEntry):
        """移除条目"""
        freq = entry.access_count
        if freq in self._lfu_bins:
            self._lfu_bins[freq].pop(key, None)
            if not self._lfu_bins[freq] and freq == self._min_freq:
                self._min_freq += 1
        self._store.pop(key, None)

    # === 统计 ===

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self._name,
                "size": len(self._store),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{self.hit_rate:.1%}",
                "ttl_seconds": self._ttl_seconds,
                "min_freq": self._min_freq,
                "freq_bins": len(self._lfu_bins),
            }

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._store.clear()
            self._lfu_bins.clear()
            self._min_freq = 0
            self._hits = 0
            self._misses = 0

    def __len__(self) -> int:
        return len(self._store)
