"""
su-memory SDK 轻量级版本
适用于资源受限环境（嵌入式设备/移动端）
内存占用：<50MB
"""
import heapq
import json
import math
import os
import re
import threading
import uuid
from collections import OrderedDict
from typing import Any

from su_memory.sdk._causal import CausalEngine  # v3.3.0
from su_memory.sdk._memory_protocol import MemoryProtocol
from su_memory.sdk._semantic_reranker import SemanticReranker  # v3.2.0
from su_memory.sdk._tiered_storage import TieredStorage  # v3.2.0

# 中文停用词表（使用frozenset减少内存占用，P2-3优化）
STOP_WORDS: frozenset = frozenset({
    '的', '了', '和', '是', '在', '有', '我', '你', '他', '她', '它',
    '这', '那', '都', '也', '就', '要', '会', '能', '对', '与', '及',
    '把', '被', '给', '但', '却', '而', '或', '而且', '并且', '所以',
    '因为', '如果', '虽然', '然后', '还是', '可以', '一个', '没有',
    '什么', '怎么', '这个', '那个', '一些', '已经', '非常', '可能',
    '应该', '知道', '觉得', '现在', '时候', '这里', '那里',
    '他们', '她们', '我们', '自己', '不是', '只是', '不能', '通过', '进行', '使用', '支持', '提供', '需要', '根据', '按照',
    '由于', '关于', '对于', '以及', '或者', '不过', '然而',
    '因此', '那么', '之后', '之前', '当时',
    '一直', '一种', '这种', '两种', '每个', '各种', '其他', '另外',
    '其中', '之间', '以后', '以前', '只有', '才能', '一定',
    '比较', '更加', '特别', '尤其', '主要', '一般', '基本', '例如',
    '比如', '包括', '就是', '不同', '相同', '同时'
})

# v3.1.0: 预编译分词器正则（避免每次 tokenize 重新编译）
_RE_CLEAN = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9]')
_RE_STRIP_ALNUM = re.compile(r'[a-zA-Z0-9]')
_RE_STRIP_CHINESE = re.compile(r'[\u4e00-\u9fa5]')
_RE_CN_DIGIT_COMBO = re.compile(r'[\u4e00-\u9fa5]\d|\d[\u4e00-\u9fa5]')
_RE_DIGIT_BLOCKS = re.compile(r'\d+')
_RE_HAS_DIGIT = re.compile(r'\d')

# v3.3.0: 分段索引阈值
_PARTITION_DF_THRESHOLD = 5000   # df 超过此值时启动分片
_PARTITION_BUCKET_SIZE = 500      # 每分片包含的文档数


class SuMemoryLite(MemoryProtocol):
    """
    轻量级SDK客户端

    适用于资源受限环境:
    - 嵌入式设备 (树莓派/Arduino)
    - 移动端 (iOS/Android)
    - 边缘计算节点

    特点:
    - 内存占用 <50MB
    - 模型大小 <20MB
    - 纯Python实现，无需额外依赖
    - 支持TF-IDF相似度计算
    - 支持持久化存储
    - 使用__slots__减少内存占用（P2-3优化）

    Example:
        >>> from su_memory.sdk import SuMemoryLite
        >>> client = SuMemoryLite()
        >>> mid = client.add("天气很好")
        >>> results = client.query("天气")
    """

    # 使用__slots__减少内存占用（P2-3优化）
    __slots__ = (
        'max_memories', 'enable_tfidf', 'enable_persistence',
        '_memories', '_index', '_doc_freq', '_total_docs',
        '_cache_size', '_query_cache', '_cache_hits', '_cache_misses',
        'storage_path', '_lock', '_storage_backend', '_storage_backend_type',
        '_semantic_reranker',  # v3.2.0: 延迟加载语义重排器
        '_tiered_storage',     # v3.2.0: 三级混合存储
        '_causal_engine',      # v3.3.0: 因果推理引擎
        '_index_partitions',   # v3.3.0: 分段索引 {kw: [(bucket, set), ...]}
        '_multihop_retriever',   # v3.4.0: 多跳检索器（lazy）
    )

    def __init__(
        self,
        max_memories: int = 10000,
        storage_path: str | None = None,
        enable_tfidf: bool = True,
        enable_persistence: bool = True,
        cache_size: int = 128,
        storage_backend: str = "default",
    ):
        """
        初始化轻量级客户端

        Args:
            max_memories: 最大记忆数量
            storage_path: 持久化存储路径（可选，默认 ~/.su_memory）
            enable_tfidf: 是否启用TF-IDF评分（默认启用）
            enable_persistence: 是否启用持久化（默认启用）
            cache_size: 查询缓存大小（默认128）
            storage_backend: 存储后端选择
                - "default": 使用内置 JSON 持久化（默认，零依赖）
                - "sqlite": SQLite 后端（零依赖）
                - "postgresql": PostgreSQL + pgvector（需 asyncpg）
                - "redis": Redis 后端（需 redis[hiredis]）
                - "auto": 自动检测 PostgreSQL → Redis → SQLite 最佳可用后端
        """
        self.max_memories = max_memories
        self.enable_tfidf = enable_tfidf
        self.enable_persistence = enable_persistence
        self._memories: list[dict[str, Any]] = []
        self._index: dict[str, set] = {}  # 使用set去重
        self._doc_freq: dict[str, int] = {}  # 文档频率（用于TF-IDF）
        self._total_docs: int = 0
        self._index_partitions: dict[str, list[tuple[int, set]]] = {}  # v3.3.0

        # LRU查询缓存
        self._cache_size = cache_size
        self._query_cache: OrderedDict[tuple[str, int], list[dict[str, Any]]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        # 自动设置默认存储路径
        if enable_persistence and not storage_path:
            storage_path = self._get_default_storage_path()

        self.storage_path = storage_path

        # 线程安全锁 (CONC-001)
        self._lock = threading.RLock()

        # v3.0.0: 可选分布式存储后端
        self._storage_backend = None
        self._storage_backend_type = storage_backend
        if storage_backend != "default":
            self._init_storage_backend(storage_backend)

        # v3.2.0: 语义重排器（延迟加载，仅首次 semantic_rerank=True 时初始化）
        self._semantic_reranker = None

        # v3.2.0: 三级混合存储（淘汰记忆自动写入温层）
        # 当 storage_path 可用时启用，避免 temp dir 下的 SQLite 碎片
        tier_dir = storage_path if (enable_persistence and storage_path) else None
        self._tiered_storage = TieredStorage(
            storage_dir=tier_dir,
            max_hot=max_memories,
        )

        # v3.3.0: 因果推理引擎（延迟初始化）
        self._causal_engine = None

        # v3.4.0: 多跳检索器（延迟初始化，仅在 multihop=True 时加载）
        self._multihop_retriever = None

        # 加载已有数据
        if enable_persistence and storage_path:
            self._load()

    def count(self) -> int:
        """获取记忆总数"""
        return len(self._memories)

    def delete(self, memory_id: str) -> bool:
        """删除指定记忆（按 memory_id）。

        Args:
            memory_id: 目标记忆 ID

        Returns:
            True 表示删除成功；False 表示未找到该记忆。
        """
        with self._lock:
            target = None
            for m in self._memories:
                if m["id"] == memory_id:
                    target = m
                    break
            if target is None:
                return False
            self._memories.remove(target)
            # 清理倒排索引（与 _evict_oldest 一致的清理逻辑）
            for keyword in set(target.get("keywords", [])):
                if keyword in self._index:
                    self._index[keyword].discard(memory_id)
                    self._doc_freq[keyword] = max(0, self._doc_freq.get(keyword, 1) - 1)
                    if not self._index[keyword]:
                        del self._index[keyword]
                        self._doc_freq.pop(keyword, None)
                        self._index_partitions.pop(keyword, None)
            self._total_docs = max(0, self._total_docs - 1)
            # 失效查询缓存（删除改变了结果集）
            self._query_cache.clear()
            return True


    def _get_default_storage_path(self) -> str:
        """
        获取默认存储路径

        优先级：
        1. 环境变量 SU_MEMORY_DATA_DIR
        2. ~/.su_memory/
        3. 当前目录 ./su_memory_data/
        """
        # 1. 检查环境变量
        env_path = os.environ.get("SU_MEMORY_DATA_DIR")
        if env_path:
            return env_path

        # 2. 使用用户目录
        home_path = os.path.expanduser("~")
        default_path = os.path.join(home_path, ".su_memory")

        # 检查是否可写
        try:
            os.makedirs(default_path, exist_ok=True)
            test_file = os.path.join(default_path, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return default_path
        except (OSError, PermissionError):
            pass

        # 3. 使用当前目录
        return os.path.join(os.getcwd(), "su_memory_data")

    def _tokenize(self, text: str) -> list[str]:
        """
        中文分词（简单实现）

        使用N-gram滑动窗口进行简单分词。
        对于生产环境，建议使用jieba等专业分词库。

        v3.1.0: 保留连续数字块（≥2位）和中文字符+数字组合 token，
        解决"第0条"与"第6条"无法区分的问题。
        预编译正则 + 数字快速路径避免不必要的正则开销。

        Args:
            text: 输入文本

        Returns:
            分词结果列表（去重）
        """
        text_lower = text.lower()

        # 去除标点符号
        text_clean = _RE_CLEAN.sub('', text_lower)

        keywords = set()  # 使用set去重

        # v3.1.0: 数字快速路径 — 仅当文本含数字时才执行数字提取
        has_digit = _RE_HAS_DIGIT.search(text_clean) is not None

        if has_digit:
            # 提取中文字符+数字的组合 token（如 "第0"、"0条"、"第6"）
            for match in _RE_CN_DIGIT_COMBO.finditer(text_clean):
                keywords.add(match.group())

            # 提取连续数字块（≥2位）为独立 token
            for d in _RE_DIGIT_BLOCKS.findall(text_clean):
                if len(d) >= 2:
                    keywords.add(d)

        # 中文分词：使用2-4字滑动窗口
        chinese_chars = _RE_STRIP_ALNUM.sub('', text_clean)

        # 处理中文：2-4字词滑动窗口
        for length in [2, 3, 4]:
            for i in range(len(chinese_chars) - length + 1):
                word = chinese_chars[i:i+length]
                if word:
                    keywords.add(word)

        # 处理英文/数字：按空格分割
        english_text = _RE_STRIP_CHINESE.sub(' ', text_lower)
        english_words = english_text.split()
        for w in english_words:
            if len(w) > 1:
                keywords.add(w)

        # 过滤停用词和单字
        result = [
            kw for kw in keywords
            if len(kw) >= 2 and kw not in STOP_WORDS
        ]

        return result

    def _extract_keywords(self, text: str) -> list[str]:
        """
        提取关键词

        使用N-gram分词，返回所有有效关键词。

        Args:
            text: 输入文本

        Returns:
            关键词列表
        """
        return self._tokenize(text)

    def add(self, content: str, metadata: dict[str, Any] = None) -> str:
        """
        添加记忆（优化版）

        Args:
            content: 记忆内容
            metadata: 元数据

        Returns:
            memory_id: 记忆唯一标识
        """
        with self._lock:
            memory_id = f"mem_{uuid.uuid4().hex[:8]}"
            timestamp = metadata.get('timestamp') if metadata else None

            # 添加前先淘汰，确保不超过限制（P0-1修复）
            while len(self._memories) >= self.max_memories:
                self._evict_oldest()

            memory = {
                "id": memory_id,
                "content": content,
                "metadata": metadata or {},
                "keywords": self._extract_keywords(content),
                "timestamp": timestamp
            }

            self._memories.append(memory)

            # 更新索引（使用set去重）
            unique_keywords = set(memory["keywords"])
            for keyword in unique_keywords:
                if keyword not in self._index:
                    self._index[keyword] = set()
                    self._doc_freq[keyword] = 0
                self._index[keyword].add(memory_id)
                self._doc_freq[keyword] += 1

                # v3.3.0: 分段索引 — 高频词自动分片
                df = self._doc_freq[keyword]
                if df >= _PARTITION_DF_THRESHOLD:
                    self._add_to_partition(keyword, memory_id, df)

            self._total_docs += 1

            # 内存限制
            if len(self._memories) > self.max_memories:
                self._evict_oldest()

            # 持久化
            if self.enable_persistence and self.storage_path:
                self._save()

            # 清除缓存（添加新记忆后需要重新查询）
            self._query_cache.clear()

            return memory_id


    def _evict_oldest(self) -> None:
        """
        淘汰最旧的记忆

        v3.2.0: 淘汰的记忆自动下沉到温层（SQLite），避免永久丢失。
        """
        if not self._memories:
            return

        oldest = self._memories.pop(0)

        # v3.2.0: 下沉到温层存储
        if self._tiered_storage is not None:
            self._tiered_storage.add_warm(oldest)

        # 更新索引
        for keyword in set(oldest.get("keywords", [])):
            if keyword in self._index:
                self._index[keyword].discard(oldest["id"])
                self._doc_freq[keyword] = max(0, self._doc_freq.get(keyword, 1) - 1)
                if not self._index[keyword]:
                    del self._index[keyword]
                    self._doc_freq.pop(keyword, None)
                    # v3.3.0: 清理对应分片
                    self._index_partitions.pop(keyword, None)

    # =========================================================================
    # v3.3.0: 分段索引
    # =========================================================================

    def _add_to_partition(self, keyword: str, memory_id: str, df: int) -> None:
        """
        将 memory_id 添加到关键词的分段索引中。
        自动创建新分片或追加到已有分片。
        """
        bucket = df // _PARTITION_BUCKET_SIZE

        if keyword not in self._index_partitions:
            self._index_partitions[keyword] = []

        partitions = self._index_partitions[keyword]

        # 查找已有分片或创建新分片
        for i, (b, ids) in enumerate(partitions):
            if b == bucket:
                ids.add(memory_id)
                return

        # 新分片
        partitions.append((bucket, {memory_id}))
        # 保持分片按 bucket 排序
        partitions.sort(key=lambda x: x[0])

    def _get_partitioned_ids(
        self, keyword: str, max_buckets: int = 5
    ) -> set:
        """
        获取分段索引中的 memory_ids（仅最近 N 个分片）。

        Args:
            keyword: 关键词
            max_buckets: 最多检索的分片数

        Returns:
            合并后的 memory_ids 集合
        """
        if keyword not in self._index_partitions:
            return set()

        partitions = self._index_partitions[keyword]
        # 取最近的 max_buckets 个分片
        recent = partitions[-max_buckets:]
        result: set = set()
        for _, ids in recent:
            result.update(ids)
        return result

    def query(self, query: str, top_k: int = 5, semantic_rerank: bool = False, multihop: bool = False) -> list[dict[str, Any]]:
        """
        查询记忆（TF-IDF优化版 + 缓存 + 倒排索引优化 + v3.2.0语义重排）

        Args:
            query: 查询内容
            top_k: 返回数量
            semantic_rerank: v3.2.0 — 启用 embedding 语义重排（需 sentence-transformers）。
                             默认 False，保持与 v3.1.0 一致的行为。
            multihop: v3.4.0 — 启用多跳推理检索（需 _sys 编码器，较重）。
                      默认 False。启用失败（依赖缺失）时自动降级为普通检索。

        Returns:
            results: 检索结果列表
        """
        with self._lock:
            # v3.4.0: 多跳推理路径（opt-in）。失败时优雅降级为普通检索。
            if multihop and self._memories:
                mh_results = self._multihop_query(query, top_k)
                if mh_results is not None:
                    return mh_results
                # 降级：继续走下面的 TF-IDF 路径

            # 缓存键包含 semantic_rerank 标志
            cache_key = (query, top_k, semantic_rerank)
            if cache_key in self._query_cache:
                self._cache_hits += 1
                # 移动到末尾（LRU）
                self._query_cache.move_to_end(cache_key)
                return self._query_cache[cache_key].copy()

            self._cache_misses += 1
            query_keywords = self._extract_keywords(query)

            if not query_keywords or not self._memories:
                return []

            # P1-2优化：使用倒排索引快速过滤候选集
            # v3.1.0: IDF阈值剪枝 — 出现超过50%的关键词无区分力，跳过
            total = len(self._memories)
            candidate_ids = None
            for keyword in query_keywords:
                if keyword in self._index:
                    df = len(self._index[keyword])
                    if df > total * 0.5:
                        continue  # 高频词跳过，避免候选集膨胀
                    if candidate_ids is None:
                        candidate_ids = self._index[keyword].copy()
                    else:
                        # 交集：只保留包含所有关键词的记忆
                        candidate_ids &= self._index[keyword]

            # 如果没有候选集，使用全部记忆
            if candidate_ids is None:
                candidate_ids = set(m["id"] for m in self._memories)

            scores: dict[str, float] = {}

            # 只对候选集评分（P1-2性能优化）
            if self.enable_tfidf:
                # TF-IDF评分
                for keyword in query_keywords:
                    if keyword in self._index:
                        df = len(self._index[keyword])
                        idf = math.log((self._total_docs + 1) / (df + 1)) + 1

                        # v3.3.0: 分段索引优化 — 高频词只扫描最近分片
                        if keyword in self._index_partitions:
                            partition_ids = self._get_partitioned_ids(keyword)
                            for memory_id in partition_ids:
                                if memory_id in candidate_ids:
                                    scores[memory_id] = scores.get(memory_id, 0) + idf
                        else:
                            for memory_id in self._index[keyword]:
                                if memory_id in candidate_ids:
                                    scores[memory_id] = scores.get(memory_id, 0) + idf
            else:
                # 简单计数
                for keyword in query_keywords:
                    if keyword in self._index:
                        if keyword in self._index_partitions:
                            partition_ids = self._get_partitioned_ids(keyword)
                            for memory_id in partition_ids:
                                if memory_id in candidate_ids:
                                    scores[memory_id] = scores.get(memory_id, 0) + 1
                        else:
                            for memory_id in self._index[keyword]:
                                if memory_id in candidate_ids:
                                    scores[memory_id] = scores.get(memory_id, 0) + 1

            # 归一化
            if scores:
                max_score = max(scores.values())
                if max_score > 0:
                    scores = {k: round(v / max_score, 4) for k, v in scores.items()}

            # v3.2.0: 语义重排路径 — TF-IDF 粗排 (top-N) → embedding 精排 (top-K)
            if semantic_rerank and scores:
                results = self._semantic_rerank_query(
                    query, scores, top_k, cache_key
                )
                return results

            # v3.1.0: heap top-K 替代全排序 — O(n log k) vs O(n log n)
            sorted_ids = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])

            # 返回结果 (按 content 去重, 防止同内容不同 id 的副本重复出现)
            results = []
            seen_contents: set[str] = set()
            memory_map = {m["id"]: m for m in self._memories}

            for memory_id, score in sorted_ids[:top_k]:
                if memory_id in memory_map:
                    memory = memory_map[memory_id]
                    content_key = memory["content"]
                    if content_key in seen_contents:
                        continue
                    seen_contents.add(content_key)
                    results.append({
                        "memory_id": memory_id,
                        "content": memory["content"],
                        "score": score,
                        "metadata": memory["metadata"]
                    })

            # v3.2.0: L0 不足时回退到温层检索
            # 温层回退同样按 content 去重, 且排除热层已返回的内容,
            # 避免 "热层 score=1.0 + 温层 score=0.1" 的重复条目.
            if len(results) < top_k and self._tiered_storage is not None:
                warm_results = self._tiered_storage.query_warm(
                    query_keywords, top_k - len(results)
                )
                for wm in warm_results:
                    content_key = wm["content"]
                    if content_key in seen_contents:
                        continue
                    seen_contents.add(content_key)
                    results.append({
                        "memory_id": wm["id"],
                        "content": wm["content"],
                        "score": 0.1,  # 温层结果分数低于热层
                        "metadata": wm.get("metadata", {}),
                        "tier": "warm",
                    })

            # 保存到缓存（LRU）
            self._query_cache[cache_key] = results
            if len(self._query_cache) > self._cache_size:
                self._query_cache.popitem(last=False)  # 删除最旧的

            return results

    def _semantic_rerank_query(
        self,
        query: str,
        tfidf_scores: dict[str, float],
        top_k: int,
        cache_key: tuple,
    ) -> list[dict[str, Any]]:
        """
        v3.2.0: 语义重排查询路径。

        TF-IDF 粗排取 top-N（4×top_k）→ embedding 余弦相似度精排 → 返回 top-K。
        sentence-transformers 不可用时自动降级回 TF-IDF。
        """
        # 延迟初始化重排器（避免启动时加载模型）
        if self._semantic_reranker is None:
            self._semantic_reranker = SemanticReranker()

        # TF-IDF 粗排取 top-N（4× 召回池，确保足够候选）
        retrieval_pool = min(top_k * 4, len(tfidf_scores))
        tfidf_top = heapq.nlargest(retrieval_pool, tfidf_scores.items(), key=lambda x: x[1])

        # 构建候选列表
        memory_map = {m["id"]: m for m in self._memories}
        candidate_items = []
        for memory_id, tf_score in tfidf_top:
            if memory_id in memory_map:
                candidate_items.append((memory_id, tf_score, memory_map[memory_id]))

        if not candidate_items:
            return []

        # 语义重排
        candidates_text = [item[2]["content"] for item in candidate_items]
        reranked = self._semantic_reranker.rerank(query, candidates_text, top_k)

        # 构建结果（融合 TF-IDF 分数 + 语义分数）
        results = []
        for orig_idx, semantic_score in reranked:
            if orig_idx < len(candidate_items):
                memory_id, tf_score, memory = candidate_items[orig_idx]
                # 综合分数：语义分数为主，TF-IDF 为辅
                combined_score = round(semantic_score * 0.7 + tf_score * 0.3, 4)
                results.append({
                    "memory_id": memory_id,
                    "content": memory["content"],
                    "score": combined_score,
                    "semantic_score": semantic_score,
                    "tfidf_score": tf_score,
                    "metadata": memory["metadata"],
                })

        # 保存到缓存
        self._query_cache[cache_key] = results
        if len(self._query_cache) > self._cache_size:
            self._query_cache.popitem(last=False)

        return results


    # =========================================================================
    # v3.4.0: Multi-hop API
    # =========================================================================

    def _init_multihop_retriever(self):
        """延迟初始化多跳检索器（含 _sys 编码器）。失败返回 None。"""
        if self._multihop_retriever is not None:
            return self._multihop_retriever
        try:
            from su_memory._sys.encoders import EncoderCore, SemanticEncoder
            from su_memory._sys.multi_hop import MultiHopRetriever
            encoder_core = EncoderCore()
            semantic_encoder = SemanticEncoder()
            self._multihop_retriever = MultiHopRetriever(
                encoder_core=encoder_core,
                causal_inference=None,
                semantic_encoder=semantic_encoder,
            )
            return self._multihop_retriever
        except Exception:
            # 依赖缺失或编码器初始化失败 → 降级为普通检索
            self._multihop_retriever = None
            return None

    def _multihop_query(self, query: str, top_k: int):
        """
        v3.4.0: 多跳推理检索。返回结果列表，或 None 表示降级。

        先用 TF-IDF 取候选池（top_k * 4），再用 MultiHopRetriever 做多跳推理重排。
        """
        retriever = self._init_multihop_retriever()
        if retriever is None:
            return None

        # 1) TF-IDF 粗排取候选池
        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            return None

        scores: dict[str, float] = {}
        for keyword in query_keywords:
            if keyword in self._index:
                df = len(self._index[keyword])
                idf = math.log((self._total_docs + 1) / (df + 1)) + 1
                for memory_id in self._index[keyword]:
                    scores[memory_id] = scores.get(memory_id, 0) + idf

        if not scores:
            return None

        pool_size = min(top_k * 4, len(scores))
        ranked = heapq.nlargest(pool_size, scores.items(), key=lambda x: x[1])
        memory_map = {m["id"]: m for m in self._memories}
        candidates = []
        for memory_id, _score in ranked:
            m = memory_map.get(memory_id)
            if m:
                candidates.append({
                    "memory_id": memory_id,
                    "content": m["content"],
                    "memory_type": m.get("metadata", {}).get("memory_type", "fact"),
                    "hexagram_index": 0,
                })

        if not candidates:
            return None

        # 2) 多跳重排
        try:
            hop_results = retriever.retrieve(
                query=query,
                candidates=candidates,
                query_complexity="normal",
                max_hops=2,
                use_vector_sim=False,
            )
        except Exception:
            return None

        # 3) 映射回标准结果结构
        results = []
        for hop in hop_results[:top_k]:
            mid = getattr(hop, "memory_id", None) or getattr(hop, "content", "")
            m = memory_map.get(mid)
            content = m["content"] if m else getattr(hop, "content", "")
            results.append({
                "memory_id": mid,
                "content": content,
                "score": round(getattr(hop, "hop_score", 0.0), 4),
                "metadata": m["metadata"] if m else {},
                "hops": getattr(hop, "hop", None) or getattr(hop, "depth", None) or 1,
            })
        return results if results else None


    # =========================================================================
    # v3.3.0: Causal API
    # =========================================================================

    def find_causal_pairs(self) -> list[tuple[dict, dict, str, float]]:
        """
        查找记忆中的因果关系对。

        Returns:
            [(cause_memory, effect_memory, causal_type, confidence), ...]
            按置信度降序排列。
        """
        if self._causal_engine is None:
            self._causal_engine = CausalEngine()
        return self._causal_engine.find_causal_pairs(list(self._memories))

    def predict_effects(
        self, cause_content: str, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """
        基于历史记忆预测给定原因的效应。

        Args:
            cause_content: 原因文本
            top_k: 返回数量

        Returns:
            [{"memory_id", "content", "confidence", "causal_type"}, ...]
        """
        if self._causal_engine is None:
            self._causal_engine = CausalEngine()
        return self._causal_engine.predict_effects(
            cause_content, list(self._memories), top_k
        )

    def query_causal_chain(
        self, query: str, max_depth: int = 2
    ) -> list[dict[str, Any]]:
        """
        查询因果链：查询 → 直接效应 → 二级效应。

        Args:
            query: 查询文本
            max_depth: 最大因果跳数

        Returns:
            [{"depth", "memory_id", "content", "confidence", ...}, ...]
        """
        if self._causal_engine is None:
            self._causal_engine = CausalEngine()
        return self._causal_engine.query_causal_chain(
            query, list(self._memories), max_depth
        )


    def predict(self, situation: str, action: str) -> dict[str, Any]:
        """
        预测（优化版）

        Args:
            situation: 当前情境
            action: 拟采取行动

        Returns:
            prediction: 预测结果
        """
        with self._lock:
            situation_keywords = self._extract_keywords(situation)
            action_keywords = self._extract_keywords(action)

            # 检索相关记忆
            related = set()
            for keyword in situation_keywords:
                if keyword in self._index:
                    related.update(self._index[keyword])

            related_count = len(related)

            # 检查是否有相似行动的历史
            similar_actions = 0
            for keyword in action_keywords:
                if keyword in self._index:
                    similar_actions += len(self._index[keyword])

            # 计算置信度
            if related_count == 0:
                confidence = 0.1
            else:
                confidence = min(related_count * 0.05 + similar_actions * 0.02, 0.95)

            return {
                "outcome": "基于TF-IDF相似度预测",
                "confidence": round(confidence, 4),
                "related_memories": related_count,
                "similar_actions": similar_actions,
                "mode": "lite_tfidf"
            }


    def get_stats(self) -> dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计字典
        """
        with self._lock:
            cache_total = self._cache_hits + self._cache_misses
            cache_hit_rate = (
                self._cache_hits / cache_total * 100
                if cache_total > 0 else 0
            )

            return {
                "total_memories": len(self._memories),
                "max_memories": self.max_memories,
                "index_size": len(self._index),
                "total_docs": self._total_docs,
                "tfidf_enabled": self.enable_tfidf,
                "persistence_enabled": self.enable_persistence,
                "storage_path": self.storage_path,
                "cache_size": len(self._query_cache),
                "cache_max_size": self._cache_size,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": round(cache_hit_rate, 2)
            }


    def _get_storage_file(self) -> str | None:
        """
        获取存储文件路径
        """
        if self.storage_path:
            return os.path.join(self.storage_path, "su_memory_lite.json")
        return None

    def _save(self) -> bool:
        """
        保存记忆到磁盘（P2-3优化：使用紧凑JSON格式）

        Returns:
            是否保存成功
        """
        storage_file = self._get_storage_file()
        if not storage_file:
            return False

        try:
            os.makedirs(os.path.dirname(storage_file), exist_ok=True)
            data = {
                "memories": self._memories,
                "index": {k: list(v) for k, v in self._index.items()},
                "doc_freq": self._doc_freq,
                "total_docs": self._total_docs
            }
            # 使用紧凑格式（无缩进）减少文件大小
            with open(storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Save failed: {e}")
            return False

    def _load(self) -> bool:
        """
        从磁盘加载记忆

        Returns:
            是否加载成功
        """
        storage_file = self._get_storage_file()
        if not storage_file or not os.path.exists(storage_file):
            return False

        try:
            with open(storage_file, encoding='utf-8') as f:
                data = json.load(f)

            self._memories = data.get("memories", [])
            self._index = {k: set(v) for k, v in data.get("index", {}).items()}
            self._doc_freq = data.get("doc_freq", {})
            self._total_docs = data.get("total_docs", len(self._memories))

            # 加载后裁剪到最大限制（P0-1修复）
            self._trim_to_max()

            # v3.3.0: 预热查询缓存（热点记忆的关键词预查询）
            self._warm_cache()

            return True
        except Exception as e:
            print(f"Load failed: {e}")
            return False

    def _trim_to_max(self) -> None:
        """
        裁剪到最大限制
        """
        while len(self._memories) > self.max_memories:
            self._evict_oldest()

    def _warm_cache(self, top_n: int = 20) -> int:
        """
        v3.3.0: 预热查询缓存。

        持久化加载后，提取最近 top-N 条记忆的高区分度关键词，
        预执行查询填充缓存，使首次真实查询命中。

        Args:
            top_n: 预热的记忆数量

        Returns:
            预热的查询数量
        """
        if not self._memories:
            return 0

        # 1. 取最近 top-N 条记忆（按插入顺序，最后插入的最新）
        recent = self._memories[-top_n:]

        # 2. 收集高区分度关键词：取每条记忆最长的 2 个关键词
        warm_queries: set = set()
        for memory in recent:
            keywords = memory.get("keywords", [])
            if not keywords:
                continue
            # 按长度降序，优先长关键词（区分度更高）
            sorted_kw = sorted(keywords, key=len, reverse=True)
            # 取前 2 个最长的，构造候选查询
            for kw in sorted_kw[:2]:
                if len(kw) >= 3:  # 至少 3 字
                    warm_queries.add(kw)

        # 3. 对每个候选关键词执行查询（结果自动缓存）
        warmed = 0
        for query_text in list(warm_queries)[:top_n]:  # 最多 top_n 个预热查询
            try:
                self.query(query_text, top_k=3)
                warmed += 1
            except Exception:
                pass

        return warmed

    def clear(self) -> None:
        """
        清空所有记忆
        """
        with self._lock:
            self._memories.clear()
            self._index.clear()
            self._doc_freq.clear()
            self._total_docs = 0

            # 删除存储文件
            storage_file = self._get_storage_file()
            if storage_file and os.path.exists(storage_file):
                os.remove(storage_file)

    # =========================================================================
    # v3.0.0: 分布式存储后端支持
    # =========================================================================

    def _init_storage_backend(self, backend_type: str) -> None:
        """
        初始化分布式存储后端 (委托共享模块)。

        Args:
            backend_type: 后端类型 ("sqlite" / "postgresql" / "redis" / "auto")
        """
        from su_memory.sdk._storage_helpers import init_storage_backend
        init_storage_backend(self, backend_type, self.storage_path, label="SuMemoryLite")

    def get_storage_backend(self):
        """
        获取当前存储后端实例。

        Returns:
            StorageBackend 或 None（使用默认 JSON 持久化时）
        """
        return self._storage_backend

    @property
    def storage_backend_type(self) -> str:
        """当前存储后端类型"""
        return self._storage_backend_type


    def __len__(self) -> int:
        """
        记忆数量
        """
        return len(self._memories)

    def __bool__(self) -> bool:
        """
        始终返回True，确保client对象在布尔上下文中为真
        """
        return True

    def __enter__(self):
        """
        上下文管理器入口
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口
        """
        if self.enable_persistence and self.storage_path:
            self._save()
