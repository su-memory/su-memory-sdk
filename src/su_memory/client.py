"""
SuMemory Client — SDK 一行API
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
import os

if TYPE_CHECKING:
    from su_memory.encoding import MemoryEncoding


class MemoryResult:
    """记忆检索结果"""
    __slots__ = ("memory_id", "content", "score", "encoding", "metadata")

    def __init__(
        self,
        memory_id: str,
        content: str,
        score: float,
        encoding: "MemoryEncoding",
        metadata: Dict[str, Any],
    ):
        self.memory_id = memory_id
        self.content = content
        self.score = score
        self.encoding = encoding
        self.metadata = metadata


class SuMemory:
    """
    Semantic Memory Engine SDK Client

    一行代码初始化，本地运行，无需服务器。

    Example:
        >>> client = SuMemory()
        >>> mid = client.add("项目ROI增长了25%", metadata={"source": "finance"})
        >>> results = client.query("投资回报")
        >>> print(results[0].encoding.category)  # creative
    """

    def __init__(
        self,
        mode: str = "local",
        storage: str = "sqlite",
        persist_dir: str = None,
        embedder = None,
    ):
        self.mode = mode
        self.storage = storage
        self.persist_dir = persist_dir or self._detect_default_dir()
        self._embedder = embedder
        self._embedding_dim = None  # V3.16: 兼容 OpenClaw 检测

        self._init_engine()

    @property
    def embedding_dim(self) -> int:
        """返回当前嵌入维度（OpenClaw 兼容接口）"""
        if self._embedding_dim is not None:
            return self._embedding_dim
        emb = self._auto_detect_embedder()
        if emb and hasattr(emb, 'dims'):
            self._embedding_dim = emb.dims
        return self._embedding_dim

    @staticmethod
    def _detect_default_dir() -> str:
        """检测默认存储目录（OpenClaw兼容）"""
        import os as _os
        home = _os.path.expanduser("~")
        # OpenClaw 环境
        openclaw_dir = _os.environ.get("OPENCLAW_DIR", _os.path.join(home, ".openclaw"))
        if _os.path.exists(openclaw_dir):
            return _os.path.join(openclaw_dir, "su_memory_data")
        # 默认
        return _os.path.join(home, ".su_memory")

    def _init_engine(self):
        """初始化引擎（含 Phase 1&2 增强模块）"""
        from su_memory._sys.encoders import SemanticEncoder, EncoderCore
        from su_memory._sys.causal import CausalChain, CausalInference
        from su_memory._sys.codec import SuCompressor
        from su_memory._sys.chrono import TemporalSystem
        from su_memory._sys.intent_classifier import IntentClassifier, ProgressiveDisclosure
        from su_memory._sys.session_bridge import SessionBridge
        from su_memory._sys.recency_feedback import RecencyFeedbackSystem
        from su_memory._sys.wiki_linker import WikiLinker
        from su_memory._sys.multi_hop import MultiHopRetriever
        from su_memory._sys.recall_trigger import RecallTrigger
        # 原有核心
        self._causal = CausalChain()
        self._codec = SuCompressor()
        self._embedder = getattr(self, '_embedder', None)  # 来自 __init__ 注入

        self._encoder = SemanticEncoder()
        self._encoder_core = EncoderCore()
        self._causal_inference = CausalInference()
        self._temporal = TemporalSystem()
        # Phase 1&2 增强模块
        self._intent_classifier = IntentClassifier()
        self._session_bridge = SessionBridge(
            persist_path=os.path.join(self.persist_dir, "sessions.jsonl"))
        self._recency_feedback = RecencyFeedbackSystem(self._temporal)
        self._wiki_linker = WikiLinker()
        self._multi_hop = MultiHopRetriever(
            self._encoder_core, self._causal_inference, self._encoder)
        self._recall_trigger = RecallTrigger(
            intent_classifier=self._intent_classifier,
            session_bridge=self._session_bridge,
            wiki_linker=self._wiki_linker,
            semantic_encoder=self._encoder,
            encoder_core=self._encoder_core,
            memory_store=self)
        self._disclosure = ProgressiveDisclosure()
        # 存储结构
        self._memories: List[Dict] = []
        self._next_id = 1
        self._semantic_index: Dict[str, List[int]] = {}
        self._energy_index: Dict[str, List[int]] = {}
        self._vectors: List[Optional[List[float]]] = []
        # _load() 由外层按需调用

    def _auto_detect_embedder(self):
        """自动检测并初始化嵌入器（四级fallback，永不返回None）"""
        if self._embedder is not None:
            return self._embedder

        # 1. Ollama (本地离线)
        try:
            import urllib.request
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                import json as _json
                models = [m['name'] for m in _json.loads(resp.read()).get('models', [])]
                has_embed = any('bge' in m.lower() or 'embed' in m.lower() for m in models)
                if has_embed or models:
                    from su_memory.sdk.embedding import OllamaEmbedding
                    self._embedder = OllamaEmbedding()
                    self._embedding_dim = self._embedder.dims
                    return self._embedder
        except Exception:
            pass

        # 2. sentence-transformers (内置依赖)
        try:
            import sentence_transformers
            model_name = os.environ.get("SU_MEMORY_EMBEDDING_MODEL",
                "paraphrase-multilingual-MiniLM-L12-v2")
            model = sentence_transformers.SentenceTransformer(model_name)
            dims = model.get_sentence_embedding_dimension()
            class _STWrapper:
                def __init__(self, st_model, ndim):
                    self._model = st_model
                    self.dims = ndim
                def encode(self, text):
                    return self._model.encode([text], convert_to_numpy=True)[0].tolist()
            self._embedder = _STWrapper(model, dims)
            self._embedding_dim = dims
            return self._embedder
        except Exception:
            pass

        # 3. TF-IDF fallback
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import hashlib, struct
            class _TfidfWrapper:
                def __init__(self):
                    self.dims = 256
                    self._corpus = []
                    self._fitted = False
                    self._vectorizer = None
                def encode(self, text):
                    if not self._fitted or len(self._corpus) < 50:
                        self._corpus.append(text)
                        return self._hash_vec(text)
                    if self._vectorizer is None:
                        self._vectorizer = TfidfVectorizer(
                            max_features=self.dims, analyzer='char_wb', ngram_range=(2,4))
                        self._vectorizer.fit(self._corpus + [text])
                    try:
                        v = self._vectorizer.transform([text]).toarray()[0]
                        vec = list(v[:self.dims])
                        if len(vec) < self.dims:
                            vec += [0.0] * (self.dims - len(vec))
                        norm = (sum(x*x for x in vec)) ** 0.5
                        if norm > 0: vec = [x/norm for x in vec]
                        return vec
                    except Exception:
                        return self._hash_vec(text)
                def _hash_vec(self, text):
                    vec = [0.0] * self.dims
                    for i,ch in enumerate(text):
                        h = hashlib.sha256(f"{i}:{ch}".encode()).digest()[:2]
                        idx = struct.unpack('<H', h)[0] % self.dims
                        vec[idx] += 1.0
                    norm = (sum(v*v for v in vec)) ** 0.5
                    if norm > 0: vec = [v/norm for v in vec]
                    return vec
            self._embedder = _TfidfWrapper()
            self._embedding_dim = 256
            return self._embedder
        except Exception:
            pass

        # 4. 最终兜底 Hash (dim=128)
        class _HashFallback:
            def __init__(self):
                self.dims = 128
            def encode(self, text):
                import hashlib, struct
                vec = [0.0] * self.dims
                for i,ch in enumerate(text):
                    h = hashlib.sha256(f"{i}:{ch}".encode()).digest()[:2]
                    idx = struct.unpack('<H', h)[0] % self.dims
                    vec[idx] += 1.0
                norm = (sum(v*v for v in vec)) ** 0.5
                if norm > 0: vec = [v/norm for v in vec]
                return vec
        self._embedder = _HashFallback()
        self._embedding_dim = 128
        return self._embedder

    def add(self, content: str, metadata: Optional[Dict] = None) -> str:
        """
        添加一条记忆

        Args:
            content: 记忆内容
            metadata: 可选元数据

        Returns:
            memory_id: 记忆唯一ID
        """
        enc = self._codec.compress(content)
        category = enc.get("category", "receptive")
        energy_type = enc.get("energy_type", "earth")
        energy = enc.get("energy", 1.0)

        memory_id = f"mem_{self._next_id}"
        self._next_id += 1

        memory = {
            "id": memory_id,
            "content": content,
            "category": category,
            "energy_type": energy_type,
            "energy": energy,
            "metadata": metadata or {},
        }

        self._memories.append(memory)
        self._causal.add(memory_id, category=category, energy_type=energy_type)

        return memory_id

    def query(self, text: str, top_k: int = 5) -> List[MemoryResult]:
        """
        语义检索记忆

        Args:
            text: 查询文本
            top_k: 返回数量

        Returns:
            按相关度排序的记忆列表
        """
        from su_memory.encoding import MemoryEncoding

        enc = self._codec.compress(text)
        query_category = enc.get("category", "receptive")
        query_energy_type = enc.get("energy_type", "earth")

        # 尝试使用向量检索
        vector_scores = {}
        if self._embedder or self._auto_detect_embedder():
            try:
                query_vec = self._embedder.encode(text)
                if query_vec:
                    for i, m in enumerate(self._memories):
                        if i < len(self._vectors) and self._vectors[i]:
                            vec = self._vectors[i]
                            # 计算余弦相似度
                            dot = sum(a * b for a, b in zip(query_vec, vec))
                            norm_q = sum(a * a for a in query_vec) ** 0.5
                            norm_m = sum(a * a for a in vec) ** 0.5
                            if norm_q > 0 and norm_m > 0:
                                vector_scores[m["id"]] = dot / (norm_q * norm_m)
            except Exception:
                pass

        results: List[MemoryResult] = []
        for m in self._memories:
            score = 0.0

            # 向量相似度（权重最高）
            if m["id"] in vector_scores:
                score += vector_scores[m["id"]] * 0.8

            # 类别匹配
            if m["category"] == query_category:
                score += 0.1

            # 能量类型匹配
            if m["energy_type"] == query_energy_type:
                score += 0.05

            # 内容包含关键词
            if any(w in m["content"] for w in text if len(w) > 1):
                score += 0.05

            if score > 0:
                results.append(MemoryResult(
                    memory_id=m["id"],
                    content=m["content"],
                    score=score,
                    encoding=MemoryEncoding(
                        category=m["category"],
                        energy=m["energy_type"],
                        pattern=0,
                        intensity=1.0,
                        time_stem="",
                        time_branch="",
                        causal_depth=0,
                    ),
                    metadata=m["metadata"],
                ))

        results.sort(key=lambda x: -x.score)
        return results[:top_k]

    def link(self, parent_id: str, child_id: str) -> bool:
        """建立两条记忆的因果关联"""
        return self._causal.link(parent_id, child_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        category_count: Dict[str, int] = {}
        energy_count: Dict[str, int] = {}
        for m in self._memories:
            category_count[m["category"]] = category_count.get(m["category"], 0) + 1
            energy_count[m["energy_type"]] = energy_count.get(m["energy_type"], 0) + 1

        return {
            "total_memories": len(self._memories),
            "category_distribution": category_count,
            "energy_distribution": energy_count,
        }

    # ── 记忆生命周期管理 ────────────────────────────────────────────

    def forget(self, memory_id: str) -> bool:
        """
        删除单条记忆

        Args:
            memory_id: 记忆ID

        Returns:
            bool: 是否成功删除

        Example:
            >>> client.forget("mem_1")
            True
        """
        for i, m in enumerate(self._memories):
            if m.get("id") == memory_id:
                self._memories.pop(i)
                # 同步删除向量
                if i < len(self._vectors):
                    self._vectors.pop(i)
                # 从因果图中移除
                self._causal.remove(memory_id)
                return True
        return False

    def decay(self, days: int = 30) -> Dict[str, int]:
        """
        时间衰减：归档超过指定天数的旧记忆

        降低旧记忆的能量值，保留但不优先检索。

        Args:
            days: 超过多少天视为旧记忆

        Returns:
            归档统计 {"archived": N, "unchanged": M}
        """
        import time
        now = time.time()
        threshold = days * 24 * 3600  # 转换为秒

        archived = 0
        unchanged = 0

        for m in self._memories:
            timestamp = m.get("timestamp", 0)
            if timestamp > 0:
                age = now - timestamp
                if age > threshold:
                    # 能量衰减到10%
                    m["energy"] = max(0.1, m.get("energy", 1.0) * 0.9)
                    m["archived"] = True
                    archived += 1
                else:
                    unchanged += 1

        return {"archived": archived, "unchanged": unchanged}

    def summarize(self, topic: str = None, max_memories: int = 10) -> str:
        """
        压缩多条记忆为单条摘要

        Args:
            topic: 可选，限定主题
            max_memories: 最多压缩的记忆条数

        Returns:
            摘要文本

        Example:
            >>> summary = client.summarize("项目进展")
            >>> print(summary)
        """
        candidates = self._memories

        # 按主题过滤
        if topic:
            results = self.query(topic, top_k=max_memories)
            candidates = [r.memory for r in results]

        if not candidates:
            return ""

        # 简单摘要：取最重要的几条拼接
        contents = [m.get("content", "") for m in candidates[:max_memories]]

        # 按时间排序
        sorted_contents = sorted(contents, key=lambda x: len(x), reverse=True)

        return f"摘要（共{len(sorted_contents)}条记忆）：" + " | ".join(sorted_contents[:3])

    def conflict_resolution(self, threshold: float = 0.7) -> List[Dict]:
        """
        检测矛盾记忆

        查找在同一主题上存在矛盾信息的记忆对。

        Args:
            threshold: 相似度阈值，超过此值认为可能矛盾

        Returns:
            矛盾记忆列表，每项包含 memory_a, memory_b, reason
        """
        conflicts = []

        for i, m1 in enumerate(self._memories):
            for m2 in self._memories[i+1:]:
                # 检查是否存在时间或事实上的矛盾
                # 例如：一个说"完成"，另一个说"未开始"
                contradiction_markers = [
                    ("完成", "未完成"), ("成功", "失败"), ("是", "否"),
                    ("同意", "拒绝"), ("存在", "不存在"), ("开始", "结束")
                ]

                content1 = m1.get("content", "")
                content2 = m2.get("content", "")

                for pos, neg in contradiction_markers:
                    if (pos in content1 and neg in content2) or \
                       (neg in content1 and pos in content2):
                        conflicts.append({
                            "memory_a": m1.get("id"),
                            "memory_b": m2.get("id"),
                            "content_a": content1[:50],
                            "content_b": content2[:50],
                            "reason": f"包含矛盾标记: '{pos}' vs '{neg}'",
                            "type": "factual_conflict"
                        })

        return conflicts

    def clear(self) -> int:
        """清空所有记忆，返回清空数量"""
        count = len(self._memories)
        self._memories.clear()
        self._vectors.clear()
        self._semantic_index.clear()
        self._energy_index.clear()
        self._causal.clear()
        self._next_id = 1
        return count

    # ── Phase 1&2: 新模块辅助方法 ─────────────────────────────────

    def get_all_memories(self) -> List[Dict]:
        """返回所有记忆（供 RecallTrigger 等使用）"""
        return self._memories

    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """按 ID 获取单条记忆"""
        for m in self._memories:
            if m.get("id") == memory_id:
                return m
        return None

    def _classify_intent(self, query: str):
        """意图分类"""
        return self._intent_classifier.classify(query)

    def _build_candidates(self) -> List[Dict]:
        """构建检索候选集"""
        return [{
            "memory_id": m.get("id"),
            "content": m.get("content"),
            "memory_type": m.get("metadata", {}).get("type", "fact"),
            "hexagram_index": m.get("hexagram_index", 0),
            "hexagram_name": m.get("hexagram_name", ""),
            "energy_type": m.get("energy_type", ""),
            "category_probs": m.get("category_probs"),
            "energy_scores": m.get("energy_scores"),
            "vector": self._vectors[i] if i < len(self._vectors) else None,
            "timestamp": m.get("timestamp", 0),
        } for i, m in enumerate(self._memories)]

    def record_feedback(self, memory_id: str, was_useful: bool) -> None:
        """记录用户反馈"""
        self._recency_feedback.record_feedback(memory_id, was_useful)
        self._session_bridge.record_memory_access(
            memory_id, relevance_score=1.0 if was_useful else 0.0)

    def on_query(self, query: str) -> None:
        """每次查询前调用"""
        intent = self._intent_classifier.classify(query)
        self._session_bridge.record_query(query, intent.name)
        self._disclosure.record_query(query, intent.name)

    def get_next_disclosure_results(self, positive: bool) -> List["MemoryResult"]:
        """正反馈后获取更深阶段结果"""
        self._disclosure.get_next_stage(feedback="positive" if positive else "negative")
        resp = self._recall_trigger.last_response
        if not resp:
            return []
        return resp.results[:self._disclosure.current_stage.max_items]

    def __len__(self) -> int:
        return len(self._memories)

    # ── 批量操作 ──────────────────────────────────────────────────

    def add_batch(self, items: List[Dict[str, Any]]) -> List[str]:
        """
        批量添加记忆（同步优化版）

        比逐条add快10x以上，适合批量导入场景。

        Args:
            items: 记忆列表，每个元素包含:
                - content: 记忆内容
                - metadata: 可选元数据

        Returns:
            memory_ids: 添加的记忆ID列表

        Example:
            >>> client.add_batch([
            ...     {"content": "记忆1", "metadata": {"source": "doc"}},
            ...     {"content": "记忆2"},
            ... ])
            ['mem_1', 'mem_2']
        """
        memory_ids = []
        for item in items:
            content = item.get("content", "")
            metadata = item.get("metadata")
            memory_id = self.add(content, metadata)
            memory_ids.append(memory_id)
        return memory_ids

    async def aadd_batch(self, items: List[Dict[str, Any]]) -> List[str]:
        """
        异步批量添加记忆

        真正的异步版本，使用协程并发处理。

        Args:
            items: 记忆列表

        Returns:
            memory_ids: 添加的记忆ID列表

        Example:
            >>> import asyncio
            >>> await client.aadd_batch([{"content": "记忆1"}, {"content": "记忆2"}])
        """
        import asyncio

        async def _add_one(item: Dict[str, Any]) -> str:
            # 模拟异步IO，实际使用线程池
            content = item.get("content", "")
            metadata = item.get("metadata")
            # 使用线程池执行同步代码
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.add, content, metadata)

        # 并发执行
        tasks = [_add_one(item) for item in items]
        return await asyncio.gather(*tasks)

    async def astream_query(self, query: str, top_k: int = 5):
        """
        异步流式查询

        返回异步生成器，支持流式处理结果。

        Args:
            query: 查询文本
            top_k: 返回数量

        Yields:
            MemoryResult: 逐条返回检索结果

        Example:
            >>> async for result in client.astream_query("项目"):
            ...     print(result.content)
        """
        import asyncio

        async def _query():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.query, query, top_k)

        results = await _query()
        for result in results:
            yield result

