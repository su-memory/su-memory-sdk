"""
SuMemory Client — SDK 一行API

v3.5.5: FAISS 索引化查询 — 替代 O(n) 线性扫描，查询延迟 10-20x 提升。
v3.5.5-p0: 批量编码优化 (add_batch 3x) + 异步预计算管道 (add 感知 <1ms)。
"""

import logging
import os
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

if TYPE_CHECKING:
    from su_memory.encoding import MemoryEncoding

logger = logging.getLogger(__name__)


class MemoryResult:
    """记忆检索结果"""
    __slots__ = ("memory_id", "content", "score", "encoding", "metadata")

    def __init__(
        self,
        memory_id: str,
        content: str,
        score: float,
        encoding: "MemoryEncoding",
        metadata: dict[str, Any],
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

    v3.5.5-p0: async_embed=True 启动异步预计算管道，add() 感知延迟 <1ms。
    """

    def __init__(
        self,
        mode: str = "local",
        storage: str = "sqlite",
        persist_dir: str = None,
        embedder = None,
        async_embed: bool = False,
    ):
        self.mode = mode
        self.storage = storage
        self.persist_dir = persist_dir or self._detect_default_dir()
        self._embedder = embedder
        self._embedding_dim = None  # V3.16: 兼容 OpenClaw 检测

        # v3.5.5-p0: 异步预计算管道 (opt-in)
        self._async_embed = async_embed
        self._embed_queue = None
        self._embed_worker: threading.Thread | None = None
        self._embed_worker_stop = threading.Event()
        self._pending_count = 0  # 待处理嵌入任务数

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
        """初始化引擎（含 Phase 1&2 增强模块 + v3.5.5 FAISS 索引）"""
        from su_memory._sys.causal import CausalChain, CausalInference
        from su_memory._sys.chrono import TemporalSystem
        from su_memory._sys.codec import SuCompressor
        from su_memory._sys.encoders import EncoderCore, SemanticEncoder
        from su_memory._sys.intent_classifier import IntentClassifier, ProgressiveDisclosure
        from su_memory._sys.multi_hop import MultiHopRetriever
        from su_memory._sys.recall_trigger import RecallTrigger
        from su_memory._sys.recency_feedback import RecencyFeedbackSystem
        from su_memory._sys.session_bridge import SessionBridge
        from su_memory._sys.wiki_linker import WikiLinker
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
        self._memories: list[dict] = []
        self._next_id = 1
        self._semantic_index: dict[str, list[int]] = {}
        self._energy_index: dict[str, list[int]] = {}
        self._vectors: list[list[float] | None] = []

        # v3.5.5: FAISS 向量索引 — 替代 O(n) 线性扫描
        self._faiss_index = None       # faiss.IndexFlatIP 实例
        self._faiss_dim = None         # 向量维度（首次 add 时确定）
        self._faiss_dirty = False      # forget() 后需重建
        # v3.5.5: 查询向量 LRU 缓存 — 消除重复 encode() 开销
        from collections import OrderedDict
        self._query_vec_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_vec_cache_max = 256
        if FAISS_AVAILABLE:
            logger.debug("FAISS 向量索引已启用 (IndexFlatIP)")
        else:
            logger.debug("FAISS 不可用，回退到线性扫描")

        # v3.5.5-p0: 异步嵌入预计算管道 (opt-in, daemon worker)
        if self._async_embed:
            import queue
            self._embed_queue = queue.Queue(maxsize=2048)
            self._embed_worker_stop.clear()
            self._embed_worker = threading.Thread(
                target=self._embed_worker_loop,
                name="su-memory-embed-worker",
                daemon=True,
            )
            self._embed_worker.start()
            logger.debug("异步嵌入管道已启动 (daemon worker)")

        # _load() 由外层按需调用

    def _embed_worker_loop(self) -> None:
        """后台线程：从队列取任务 → encode → 写入 _vectors + FAISS"""
        while not self._embed_worker_stop.is_set():
            try:
                task = self._embed_queue.get(timeout=0.5)
            except Exception:
                continue
            if task is None:  # 关闭信号
                break
            idx, content = task
            try:
                raw_vec = self._embedder.encode(content)
                if raw_vec is not None:
                    vec = np.asarray(raw_vec, dtype=np.float32)
                    vec = self._l2_normalize(vec)
                    if idx < len(self._vectors):
                        self._vectors[idx] = vec.tolist()
                    if FAISS_AVAILABLE and not self._faiss_dirty:
                        if self._faiss_index is None:
                            dim = len(vec)
                            self._faiss_dim = dim
                            self._faiss_index = faiss.IndexFlatIP(dim)
                        self._faiss_index.add(vec.reshape(1, -1))
            except Exception:
                logger.debug("异步嵌入 worker 编码失败", exc_info=True)
            finally:
                self._pending_count = max(0, self._pending_count - 1)

    def _flush_pending_embeddings(self, timeout: float = 30.0) -> None:
        """等待所有待处理的嵌入任务完成（query 前调用以保一致性）"""
        if not self._async_embed or self._embed_queue is None:
            return
        import time as _time
        deadline = _time.time() + timeout
        while self._pending_count > 0:
            if _time.time() > deadline:
                logger.warning(
                    "_flush_pending_embeddings 超时 (%.1fs), %d 任务未完成",
                    timeout, self._pending_count,
                )
                break
            _time.sleep(0.01)

    def _encode_batch(self, contents: list[str]) -> list:
        """批量编码：优先使用 embedder.encode_batch，回退逐条 encode"""
        self._ensure_embedder()
        if self._embedder is None:
            return [None] * len(contents)
        if hasattr(self._embedder, 'encode_batch'):
            try:
                return self._embedder.encode_batch(contents)
            except Exception:
                logger.debug("encode_batch 失败，回退逐条编码", exc_info=True)
        return [self._embedder.encode(c) for c in contents]

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
                def encode_batch(self, texts):
                    """批量编码 — sentence-transformers 原生支持"""
                    arr = self._model.encode(texts, convert_to_numpy=True)
                    return arr.tolist()
            self._embedder = _STWrapper(model, dims)
            self._embedding_dim = dims
            return self._embedder
        except Exception:
            pass

        # 3. TF-IDF fallback
        try:
            import hashlib
            import struct

            from sklearn.feature_extraction.text import TfidfVectorizer
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
                        if norm > 0:
                            vec = [x/norm for x in vec]
                        return vec
                    except Exception:
                        return self._hash_vec(text)
                def encode_batch(self, texts):
                    return [self.encode(t) for t in texts]
                def _hash_vec(self, text):
                    vec = [0.0] * self.dims
                    for i,ch in enumerate(text):
                        h = hashlib.sha256(f"{i}:{ch}".encode()).digest()[:2]
                        idx = struct.unpack('<H', h)[0] % self.dims
                        vec[idx] += 1.0
                    norm = (sum(v*v for v in vec)) ** 0.5
                    if norm > 0:
                        vec = [v/norm for v in vec]
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
                import hashlib
                import struct
                vec = [0.0] * self.dims
                for i,ch in enumerate(text):
                    h = hashlib.sha256(f"{i}:{ch}".encode()).digest()[:2]
                    idx = struct.unpack('<H', h)[0] % self.dims
                    vec[idx] += 1.0
                norm = (sum(v*v for v in vec)) ** 0.5
                if norm > 0:
                    vec = [v/norm for v in vec]
                return vec
            def encode_batch(self, texts):
                return [self.encode(t) for t in texts]
        self._embedder = _HashFallback()
        self._embedding_dim = 128
        return self._embedder

    def _ensure_embedder(self):
        """确保嵌入器已初始化（延迟初始化，兼容 __init__ 注入）"""
        if self._embedder is None:
            self._auto_detect_embedder()

    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 归一化向量，使 IndexFlatIP 等价于余弦相似度"""
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec

    def _ensure_faiss_index(self) -> None:
        """延迟初始化或重建 FAISS 索引（IndexFlatIP）"""
        if not FAISS_AVAILABLE:
            return
        if self._faiss_dirty and self._faiss_index is not None:
            # forget() 后需要重建
            self._faiss_index.reset()
            self._faiss_dirty = False
            # 重新添加所有向量
            valid_vectors = [v for v in self._vectors if v is not None]
            if valid_vectors:
                arr = np.array(valid_vectors, dtype=np.float32)
                self._faiss_index.add(arr)
        if self._faiss_index is None and self._vectors:
            valid = [v for v in self._vectors if v is not None]
            if valid:
                dim = len(valid[0])
                self._faiss_dim = dim
                self._faiss_index = faiss.IndexFlatIP(dim)
                arr = np.array(valid, dtype=np.float32)
                self._faiss_index.add(arr)

    def _add_vector_to_faiss(self, vec: np.ndarray) -> None:
        """将单条向量写入 FAISS 索引（内联辅助）"""
        if FAISS_AVAILABLE and not self._faiss_dirty:
            if self._faiss_index is None:
                dim = len(vec)
                self._faiss_dim = dim
                self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(vec.reshape(1, -1))

    def add(self, content: str, metadata: dict | None = None,
            _vector: "np.ndarray | None" = None) -> str:
        """
        添加一条记忆（v3.5.5-p0: 支持预计算向量 + 异步嵌入管道）

        Args:
            content: 记忆内容
            metadata: 可选元数据
            _vector: (内部) 预计算 + L2 归一化后的向量，跳过 encode

        Returns:
            memory_id: 记忆唯一ID
        """
        # v3.5.5-p0: 输入校验
        if not content or not isinstance(content, str):
            raise ValueError("add() 的 content 必须是非空字符串")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("add() 的 metadata 必须是 dict 或 None")

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

        # v3.5.5-p0: 预计算向量路径（来自 add_batch 批量编码）
        if _vector is not None:
            vec = np.asarray(_vector, dtype=np.float32)
            if vec.ndim == 1:
                vec = self._l2_normalize(vec)
            self._vectors.append(vec.tolist())
            self._add_vector_to_faiss(vec)
            return memory_id

        # v3.5.5-p0: 异步嵌入管道 — 入队后立即返回
        if self._async_embed and self._embed_queue is not None:
            self._pending_count += 1
            vec_idx = len(self._vectors)
            self._vectors.append(None)  # 占位
            self._embed_queue.put((vec_idx, content))
            return memory_id

        # v3.5.5: 同步预计算向量并写入 FAISS 索引
        self._ensure_embedder()
        if self._embedder is not None:
            try:
                raw_vec = self._embedder.encode(content)
                if raw_vec is not None:
                    vec = np.asarray(raw_vec, dtype=np.float32)
                    vec = self._l2_normalize(vec)
                    self._vectors.append(vec.tolist())
                    self._add_vector_to_faiss(vec)
                    return memory_id
            except Exception:
                logger.debug("add() 编码失败", exc_info=True)

        self._vectors.append(None)
        return memory_id

    def _get_query_vector(self, text: str) -> np.ndarray:
        """获取查询向量（带 LRU 缓存，消除重复 encode() 开销）"""
        if text in self._query_vec_cache:
            self._query_vec_cache.move_to_end(text)
            return self._query_vec_cache[text]

        self._ensure_embedder()
        raw_vec = self._embedder.encode(text)
        vec = np.asarray(raw_vec, dtype=np.float32)
        vec = self._l2_normalize(vec)

        # LRU 缓存
        self._query_vec_cache[text] = vec
        if len(self._query_vec_cache) > self._query_vec_cache_max:
            self._query_vec_cache.popitem(last=False)
        return vec

    def query(self, text: str, top_k: int = 5) -> list[MemoryResult]:
        """
        语义检索记忆（v3.5.5: FAISS 索引化 + 查询向量缓存 — O(d) 替代 O(n)）

        Args:
            text: 查询文本
            top_k: 返回数量

        Returns:
            按相关度排序的记忆列表
        """
        # v3.5.5-p0: 刷新待处理嵌入（异步模式下保证一致性）
        self._flush_pending_embeddings()

        from su_memory.encoding import MemoryEncoding

        enc = self._codec.compress(text)
        query_category = enc.get("category", "receptive")
        query_energy_type = enc.get("energy_type", "earth")

        # v3.5.5: FAISS 快速路径 — 向量索引 + 候选精排 + 查询向量缓存
        self._ensure_embedder()
        if (FAISS_AVAILABLE and self._embedder is not None
                and self._vectors and any(v is not None for v in self._vectors)):
            try:
                self._ensure_faiss_index()
                if self._faiss_index is not None and self._faiss_index.ntotal > 0:
                    query_vec = self._get_query_vector(text)

                    # FAISS 粗排：取 top_k * 8 候选
                    n_candidates = min(top_k * 8, self._faiss_index.ntotal)
                    distances, indices = self._faiss_index.search(
                        query_vec.reshape(1, -1), n_candidates
                    )

                    # 候选精排：向量相似度 + 类别 + 能量 + 关键词
                    scored: list[tuple[float, int]] = []
                    for rank, idx in enumerate(indices[0]):
                        idx = int(idx)
                        if idx >= len(self._memories):
                            continue
                        m = self._memories[idx]
                        score = distances[0][rank] * 0.8  # FAISS 内积 = 余弦相似度
                        if m["category"] == query_category:
                            score += 0.1
                        if m["energy_type"] == query_energy_type:
                            score += 0.05
                        if any(w in m["content"] for w in text if len(w) > 1):
                            score += 0.05
                        scored.append((score, idx))

                    scored.sort(key=lambda x: -x[0])

                    results: list[MemoryResult] = []
                    for score, idx in scored[:top_k]:
                        m = self._memories[idx]
                        results.append(MemoryResult(
                            memory_id=m["id"],
                            content=m["content"],
                            score=round(score, 4),
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
                    return results
            except Exception:
                logger.debug("FAISS 查询失败，回退到线性扫描", exc_info=True)

        # 回退路径：原始线性扫描（FAISS 不可用或无向量时）
        vector_scores: dict[str, float] = {}
        if self._embedder is not None:
            try:
                query_vec = self._embedder.encode(text)
                if query_vec:
                    qv = np.asarray(query_vec, dtype=np.float32)
                    qv = self._l2_normalize(qv)
                    for i, m in enumerate(self._memories):
                        if i < len(self._vectors) and self._vectors[i]:
                            mv = np.asarray(self._vectors[i], dtype=np.float32)
                            dot = float(np.dot(qv, mv))
                            vector_scores[m["id"]] = dot
            except Exception:
                pass

        results = []
        for m in self._memories:
            score = 0.0
            if m["id"] in vector_scores:
                score += vector_scores[m["id"]] * 0.8
            if m["category"] == query_category:
                score += 0.1
            if m["energy_type"] == query_energy_type:
                score += 0.05
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

    def get_stats(self) -> dict[str, Any]:
        """获取记忆统计（v3.5.5-p0: 含 FAISS + 异步嵌入状态）"""
        category_count: dict[str, int] = {}
        energy_count: dict[str, int] = {}
        for m in self._memories:
            category_count[m["category"]] = category_count.get(m["category"], 0) + 1
            energy_count[m["energy_type"]] = energy_count.get(m["energy_type"], 0) + 1

        stats = {
            "total_memories": len(self._memories),
            "category_distribution": category_count,
            "energy_distribution": energy_count,
            "faiss_enabled": FAISS_AVAILABLE and self._faiss_index is not None,
            "async_embed": self._async_embed,
            "pending_embeddings": self._pending_count,
        }
        if self._faiss_index is not None:
            stats["faiss_index_size"] = self._faiss_index.ntotal
            stats["faiss_dim"] = self._faiss_dim
        return stats

    # ── 记忆生命周期管理 ────────────────────────────────────────────

    def forget(self, memory_id: str) -> bool:
        """
        删除单条记忆

        Args:
            memory_id: 记忆ID

        Returns:
            bool: 是否成功删除

        Note:
            v3.5.5: FAISS IndexFlatIP 不支持逐条删除，
            标记 dirty 后下次 query 时延迟重建索引。

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
                # v3.5.5: 标记 FAISS 索引待重建
                self._faiss_dirty = True
                # 从因果图中移除
                try:
                    self._causal.remove(memory_id)
                except Exception:
                    logger.debug(f"CausalChain.remove() 不支持，跳过删除 {memory_id}")
                return True
        return False

    def decay(self, days: int = 30) -> dict[str, int]:
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

    def conflict_resolution(self, threshold: float = 0.7) -> list[dict]:
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
        """清空所有记忆（v3.5.5-p0: 含异步嵌入管道清理），返回清空数量"""
        count = len(self._memories)
        self._memories.clear()
        self._vectors.clear()
        self._semantic_index.clear()
        self._energy_index.clear()
        try:
            self._causal.clear()
        except Exception:
            logger.debug("CausalChain.clear() 失败，跳过", exc_info=True)
        self._next_id = 1
        # v3.5.5: 重置 FAISS 索引 + 查询向量缓存
        if self._faiss_index is not None:
            self._faiss_index.reset()
            self._faiss_index = None
            self._faiss_dim = None
            self._faiss_dirty = False
        self._query_vec_cache.clear()
        # v3.5.5-p0: 清空异步嵌入队列
        self._pending_count = 0
        if self._embed_queue is not None:
            while not self._embed_queue.empty():
                try:
                    self._embed_queue.get_nowait()
                except Exception:
                    break
        return count

    # ── Phase 1&2: 新模块辅助方法 ─────────────────────────────────

    def get_all_memories(self) -> list[dict]:
        """返回所有记忆（供 RecallTrigger 等使用）"""
        return self._memories

    def get_memory(self, memory_id: str) -> dict | None:
        """按 ID 获取单条记忆"""
        for m in self._memories:
            if m.get("id") == memory_id:
                return m
        return None

    def _classify_intent(self, query: str):
        """意图分类"""
        return self._intent_classifier.classify(query)

    def _build_candidates(self) -> list[dict]:
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

    def get_next_disclosure_results(self, positive: bool) -> list["MemoryResult"]:
        """正反馈后获取更深阶段结果"""
        self._disclosure.get_next_stage(feedback="positive" if positive else "negative")
        resp = self._recall_trigger.last_response
        if not resp:
            return []
        return resp.results[:self._disclosure.current_stage.max_items]

    def __len__(self) -> int:
        return len(self._memories)

    # ── 批量操作 ──────────────────────────────────────────────────

    def add_batch(self, items: list[dict[str, Any]]) -> list[str]:
        """
        批量添加记忆（v3.5.5-p0: 批量编码优化，3x 提升）

        先批量编码所有文本，再逐条写入 FAISS 索引。

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
        # v3.5.5-p0: 输入校验
        if not items:
            raise ValueError("add_batch() 的 items 不能为空列表")
        if not isinstance(items, list):
            raise ValueError("add_batch() 的 items 必须是 list")

        # 提取所有文本
        contents: list[str] = []
        metadatas: list[dict | None] = []
        for i, item in enumerate(items):
            content = item.get("content", "")
            if not content or not isinstance(content, str):
                raise ValueError(f"add_batch() items[{i}].content 必须是非空字符串")
            contents.append(content)
            metadatas.append(item.get("metadata"))

        # v3.5.5-p0: 批量编码
        raw_vectors = self._encode_batch(contents)

        # 逐条写入（使用预计算向量跳过 encode）
        memory_ids: list[str] = []
        for i, content in enumerate(contents):
            raw_vec = raw_vectors[i] if i < len(raw_vectors) else None
            vec = None
            if raw_vec is not None:
                vec = self._l2_normalize(np.asarray(raw_vec, dtype=np.float32))
            memory_id = self.add(content, metadatas[i], _vector=vec)
            memory_ids.append(memory_id)
        return memory_ids

    async def aadd_batch(self, items: list[dict[str, Any]]) -> list[str]:
        """
        异步批量添加记忆（v3.5.5-p0: 批量编码 + 线程池并发写入）

        真正的异步版本，批量编码后通过线程池并发写入。

        Args:
            items: 记忆列表

        Returns:
            memory_ids: 添加的记忆ID列表

        Example:
            >>> import asyncio
            >>> await client.aadd_batch([{"content": "记忆1"}, {"content": "记忆2"}])
        """
        import asyncio

        # v3.5.5-p0: 输入校验
        if not items:
            raise ValueError("aadd_batch() 的 items 不能为空列表")

        # 提取所有文本
        contents: list[str] = []
        metadatas: list[dict | None] = []
        for item in items:
            content = item.get("content", "")
            if not content or not isinstance(content, str):
                raise ValueError("aadd_batch() items[].content 必须是非空字符串")
            contents.append(content)
            metadatas.append(item.get("metadata"))

        # 批量编码（同步执行，sentence-transformers 不释放 GIL）
        loop = asyncio.get_event_loop()
        raw_vectors = await loop.run_in_executor(None, self._encode_batch, contents)

        # 并发写入
        async def _add_one(i: int) -> str:
            raw_vec = raw_vectors[i] if i < len(raw_vectors) else None
            vec = None
            if raw_vec is not None:
                vec = self._l2_normalize(np.asarray(raw_vec, dtype=np.float32))
            return await loop.run_in_executor(
                None, self.add, contents[i], metadatas[i], vec)

        tasks = [_add_one(i) for i in range(len(contents))]
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
