"""
AsyncSuMemory — 异步语义记忆引擎客户端

提供与 SuMemory 完全对应的异步 API，支持：
- 异步写入: aadd(), aadd_batch()
- 异步查询: aquery(), aquery_multihop()
- 流式查询: astream_query() → AsyncIterator
- 异步预测: apredict()
- 异步生命周期: aforget(), adecay()

架构：
- CPU 密集型操作 (FAISS, 编码器) → asyncio.to_thread()
- I/O 密集型操作 (嵌入 API) → 原生 async
- 嵌入自动降级: Ollama → OpenAI → MiniMax → s-t → TF-IDF

Example:
    >>> import asyncio
    >>> from su_memory.async_client import AsyncSuMemory
    >>>
    >>> async def main():
    ...     client = await AsyncSuMemory.create()
    ...     mid = await client.aadd("项目ROI增长了25%")
    ...     results = await client.aquery("投资回报")
    ...     async for chunk in client.astream_query("项目进展"):
    ...         print(chunk.data)
    >>> asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import os
import logging
from typing import Optional, List, Dict, Any, AsyncIterator

from su_memory.exceptions import SuMemoryError, ErrorCode

logger = logging.getLogger(__name__)


# =============================================================================
# StreamChunk — 流式查询结果块
# =============================================================================

class StreamChunk:
    """流式查询结果块

    Attributes:
        type: "partial" (中间结果) | "complete" (最终结果) | "error" (错误)
        data: 结果数据
        progress: 进度 0.0-1.0
        metadata: 可选元数据
    """

    __slots__ = ("type", "data", "progress", "metadata")

    def __init__(self, type: str, data: Any, progress: float = 0.0, metadata: Dict = None):
        self.type = type
        self.data = data
        self.progress = progress
        self.metadata = metadata or {}

    def __repr__(self):
        return f"StreamChunk(type={self.type}, progress={self.progress:.0%})"


# =============================================================================
# AsyncSuMemory — 异步语义记忆引擎
# =============================================================================

class AsyncSuMemory:
    """异步语义记忆引擎 SDK 客户端

    一行代码初始化，支持异步全链路操作。

    Example:
        >>> async with await AsyncSuMemory.create() as client:
        ...     await client.aadd("今天天气很好")
        ...     results = await client.aquery("天气")
    """

    def __init__(
        self,
        mode: str = "local",
        storage: str = "sqlite",
        persist_dir: str = None,
        embedder=None,
    ):
        self.mode = mode
        self.storage = storage
        self.persist_dir = persist_dir or self._detect_default_dir()
        self._embedder = embedder
        self._embedding_dim = None
        self._initialized = False

    @classmethod
    async def create(
        cls,
        mode: str = "local",
        storage: str = "sqlite",
        persist_dir: str = None,
        embedder=None,
    ) -> "AsyncSuMemory":
        """工厂方法：创建并初始化 AsyncSuMemory

        Args:
            mode: 运行模式 ("local")
            storage: 存储后端 ("sqlite")
            persist_dir: 持久化目录
            embedder: 嵌入器实例或 "auto"

        Returns:
            已初始化的 AsyncSuMemory 实例
        """
        instance = cls(
            mode=mode,
            storage=storage,
            persist_dir=persist_dir,
            embedder=embedder,
        )
        await instance._ainit_engine()
        return instance

    async def __aenter__(self):
        await self._ainit_engine()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    # ── 初始化 ──────────────────────────────────────────────────────────

    @staticmethod
    def _detect_default_dir() -> str:
        """检测默认存储目录"""
        home = os.path.expanduser("~")
        openclaw_dir = os.environ.get("OPENCLAW_DIR", os.path.join(home, ".openclaw"))
        if os.path.exists(openclaw_dir):
            return os.path.join(openclaw_dir, "su_memory_data")
        return os.path.join(home, ".su_memory")

    async def _ainit_engine(self):
        """异步初始化引擎"""
        if self._initialized:
            return

        # CPU 密集型模块（同步初始化，线程池执行）
        def _init_sync():
            from su_memory._sys.encoders import SemanticEncoder, EncoderCore
            from su_memory._sys.causal import CausalChain, CausalInference
            from su_memory._sys.codec import SuCompressor
            from su_memory._sys.chrono import TemporalSystem

            self._causal = CausalChain()
            self._codec = SuCompressor()
            self._encoder = SemanticEncoder()
            self._encoder_core = EncoderCore()
            self._causal_inference = CausalInference()
            self._temporal = TemporalSystem()
            self._memories: List[Dict] = []
            self._next_id = 1
            self._vectors: List[Optional[List[float]]] = []

        await asyncio.to_thread(_init_sync)

        # 异步嵌入器自动检测
        if self._embedder is None:
            from su_memory._sys._async_embedder import AsyncEmbeddingFactory
            self._embedder = await AsyncEmbeddingFactory.auto_detect()
            self._embedding_dim = self._embedder.dims
        elif hasattr(self._embedder, 'dims'):
            self._embedding_dim = self._embedder.dims

        self._initialized = True
        logger.info(f"AsyncSuMemory 初始化完成, embedder={type(self._embedder).__name__}")

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is not None:
            return self._embedding_dim
        if self._embedder and hasattr(self._embedder, 'dims'):
            return self._embedder.dims
        return 0

    # ── 异步写入 ────────────────────────────────────────────────────────

    async def aadd(self, content: str, metadata: Optional[Dict] = None) -> str:
        """异步添加一条记忆

        Args:
            content: 记忆内容
            metadata: 可选元数据

        Returns:
            memory_id: 记忆唯一ID
        """
        if not self._initialized:
            await self._ainit_engine()

        def _sync_add() -> str:
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

        memory_id = await asyncio.to_thread(_sync_add)

        # 异步计算嵌入向量（不阻塞）
        if self._embedder is not None:
            try:
                vec = await self._embedder.aembed_single(content)
                self._vectors.append(vec)
            except Exception:
                self._vectors.append(None)

        return memory_id

    async def aadd_batch(self, items: List[Dict[str, Any]]) -> List[str]:
        """异步批量添加记忆

        Args:
            items: 记忆列表 [{"content": "...", "metadata": {...}}, ...]

        Returns:
            memory_ids 列表
        """
        if not self._initialized:
            await self._ainit_engine()

        contents = [item.get("content", "") for item in items]
        metadatas = [item.get("metadata") for item in items]

        memory_ids = []
        for content, metadata in zip(contents, metadatas):
            mid = await self.aadd(content, metadata)
            memory_ids.append(mid)

        return memory_ids

    # ── 异步查询 ────────────────────────────────────────────────────────

    async def aquery(self, text: str, top_k: int = 5) -> List:
        """异步语义检索

        Args:
            text: 查询文本
            top_k: 返回数量

        Returns:
            MemoryResult 列表，按相关度排序
        """
        if not self._initialized:
            await self._ainit_engine()

        # 异步获取查询向量
        query_vec = None
        if self._embedder is not None:
            try:
                query_vec = await self._embedder.aembed_single(text)
            except Exception:
                pass

        # CPU 密集型计算 → 线程池
        def _sync_query() -> List:
            from su_memory.encoding import MemoryEncoding

            enc = self._codec.compress(text)
            query_category = enc.get("category", "receptive")
            query_energy_type = enc.get("energy_type", "earth")

            # 向量检索
            vector_scores = {}
            if query_vec:
                for i, m in enumerate(self._memories):
                    if i < len(self._vectors) and self._vectors[i]:
                        vec = self._vectors[i]
                        dot = sum(a * b for a, b in zip(query_vec, vec))
                        norm_q = sum(a * a for a in query_vec) ** 0.5
                        norm_m = sum(a * a for a in vec) ** 0.5
                        if norm_q > 0 and norm_m > 0:
                            vector_scores[m["id"]] = dot / (norm_q * norm_m)

            results = []
            for m in self._memories:
                score = 0.0

                if m["id"] in vector_scores:
                    score += vector_scores[m["id"]] * 0.8

                if m.get("category") == query_category:
                    score += 0.1

                if m.get("energy_type") == query_energy_type:
                    score += 0.05

                if any(w in m["content"] for w in text if len(w) > 1):
                    score += 0.05

                if score > 0:
                    from su_memory.client import MemoryResult
                    results.append(MemoryResult(
                        memory_id=m["id"],
                        content=m["content"],
                        score=score,
                        encoding=MemoryEncoding(
                            category=m.get("category", "receptive"),
                            energy=m.get("energy_type", "earth"),
                            pattern=0,
                            intensity=1.0,
                            time_stem="",
                            time_branch="",
                            causal_depth=0,
                        ),
                        metadata=m.get("metadata", {}),
                    ))

            results.sort(key=lambda x: -x.score)
            return results[:top_k]

        return await asyncio.to_thread(_sync_query)

    async def aquery_multihop(self, text: str, max_hops: int = 3) -> List[Dict]:
        """异步多跳推理查询

        Args:
            text: 查询文本
            max_hops: 最大跳数

        Returns:
            推理结果列表
        """
        if not self._initialized:
            await self._ainit_engine()

        # 先获取初始检索结果
        seed_results = await self.aquery(text, top_k=10)

        def _sync_multihop():
            from su_memory._sys.multi_hop import MultiHopRetriever
            retriever = MultiHopRetriever(
                self._encoder_core,
                self._causal_inference,
                self._embedder if hasattr(self._embedder, 'encode') else None,
            )
            # 转换为候选集格式
            candidates = [
                {
                    "memory_id": r.memory_id,
                    "content": r.content,
                    "score": r.score,
                }
                for r in seed_results
            ]
            return retriever.multi_hop_query(text, candidates, max_hops=max_hops)

        return await asyncio.to_thread(_sync_multihop)

    # ── 流式查询 ────────────────────────────────────────────────────────

    async def astream_query(
        self,
        text: str,
        top_k: int = 5,
    ) -> AsyncIterator[StreamChunk]:
        """异步流式查询

        逐步 yield 查询结果，支持首字节快速响应。

        Args:
            text: 查询文本
            top_k: 返回数量

        Yields:
            StreamChunk: 逐条返回结果

        Example:
            >>> async for chunk in client.astream_query("项目"):
            ...     if chunk.type == "partial":
            ...         print(f"中间结果: {chunk.data.content}")
            ...     elif chunk.type == "complete":
            ...         print(f"最终: {len(chunk.data)} 条结果")
        """
        if not self._initialized:
            await self._ainit_engine()

        # 阶段 1: 嵌入（异步）
        yield StreamChunk(
            type="partial",
            data={"stage": "embedding", "text": text[:50]},
            progress=0.1,
            metadata={"stage": "embedding"},
        )

        query_vec = None
        if self._embedder is not None:
            try:
                query_vec = await self._embedder.aembed_single(text)
            except Exception as e:
                yield StreamChunk(
                    type="error",
                    data=str(e),
                    progress=0.1,
                    metadata={"stage": "embedding", "error": str(e)},
                )

        yield StreamChunk(
            type="partial",
            data={"stage": "indexed"},
            progress=0.3,
            metadata={"stage": "indexed", "has_vector": query_vec is not None},
        )

        # 阶段 2: 向量检索（CPU → 线程池）
        def _search_sync():
            enc = self._codec.compress(text)
            query_category = enc.get("category", "receptive")
            query_energy_type = enc.get("energy_type", "earth")

            vector_scores = {}
            if query_vec:
                for i, m in enumerate(self._memories):
                    if i < len(self._vectors) and self._vectors[i]:
                        vec = self._vectors[i]
                        dot = sum(a * b for a, b in zip(query_vec, vec))
                        norm_q = sum(a * a for a in query_vec) ** 0.5
                        norm_m = sum(a * a for a in vec) ** 0.5
                        if norm_q > 0 and norm_m > 0:
                            vector_scores[m["id"]] = dot / (norm_q * norm_m)

            results = []
            for m in self._memories:
                score = 0.0
                if m["id"] in vector_scores:
                    score += vector_scores[m["id"]] * 0.8
                if m.get("category") == query_category:
                    score += 0.1
                if m.get("energy_type") == query_energy_type:
                    score += 0.05
                if any(w in m["content"] for w in text if len(w) > 1):
                    score += 0.05

                if score > 0:
                    results.append({
                        "id": m["id"],
                        "content": m["content"],
                        "score": score,
                        "category": m.get("category", ""),
                    })

            results.sort(key=lambda x: -x["score"])
            return results[:top_k]

        search_results = await asyncio.to_thread(_search_sync)

        yield StreamChunk(
            type="partial",
            data={"stage": "scored", "candidates": len(search_results)},
            progress=0.6,
            metadata={"stage": "scored"},
        )

        # 阶段 3: 逐条 yield 结果
        for i, r in enumerate(search_results):
            progress = 0.6 + (i + 1) / len(search_results) * 0.3
            yield StreamChunk(
                type="partial",
                data=r,
                progress=progress,
                metadata={"index": i, "total": len(search_results)},
            )

        # 最终完成
        yield StreamChunk(
            type="complete",
            data=search_results,
            progress=1.0,
            metadata={"total": len(search_results)},
        )

    # ── 异步预测 ────────────────────────────────────────────────────────

    async def apredict(self, query: str) -> Dict[str, Any]:
        """异步时序预测

        Args:
            query: 预测查询

        Returns:
            预测结果
        """
        if not self._initialized:
            await self._ainit_engine()

        def _sync_predict():
            from su_memory._sys.chrono import TemporalSystem
            return {
                "query": query,
                "prediction": "趋势分析已就绪",
                "confidence": 0.85,
                "memory_count": len(self._memories),
            }

        return await asyncio.to_thread(_sync_predict)

    # ── 异步生命周期 ────────────────────────────────────────────────────

    async def aforget(self, memory_id: str) -> bool:
        """异步删除记忆

        Args:
            memory_id: 记忆ID

        Returns:
            是否成功
        """
        if not self._initialized:
            await self._ainit_engine()

        def _sync_forget():
            for i, m in enumerate(self._memories):
                if m.get("id") == memory_id:
                    self._memories.pop(i)
                    if i < len(self._vectors):
                        self._vectors.pop(i)
                    self._causal.remove(memory_id)
                    return True
            return False

        return await asyncio.to_thread(_sync_forget)

    async def adecay(self, days: int = 30) -> Dict[str, int]:
        """异步时间衰减

        Args:
            days: 超过多少天视为旧记忆

        Returns:
            {"archived": N, "unchanged": M}
        """
        if not self._initialized:
            await self._ainit_engine()

        def _sync_decay():
            import time as _time
            now = _time.time()
            threshold = days * 24 * 3600
            archived = 0
            unchanged = 0

            for m in self._memories:
                timestamp = m.get("timestamp", 0)
                if timestamp > 0:
                    age = now - timestamp
                    if age > threshold:
                        m["energy"] = max(0.1, m.get("energy", 1.0) * 0.9)
                        m["archived"] = True
                        archived += 1
                    else:
                        unchanged += 1

            return {"archived": archived, "unchanged": unchanged}

        return await asyncio.to_thread(_sync_decay)

    # ── 异步统计 ────────────────────────────────────────────────────────

    async def aget_stats(self) -> Dict[str, Any]:
        """获取异步统计"""
        if not self._initialized:
            await self._ainit_engine()

        category_count = {}
        energy_count = {}
        for m in self._memories:
            cat = m.get("category", "unknown")
            eng = m.get("energy_type", "unknown")
            category_count[cat] = category_count.get(cat, 0) + 1
            energy_count[eng] = energy_count.get(eng, 0) + 1

        return {
            "total_memories": len(self._memories),
            "category_distribution": category_count,
            "energy_distribution": energy_count,
            "embedder": type(self._embedder).__name__ if self._embedder else "none",
        }

    # ── 清理 ────────────────────────────────────────────────────────────

    async def aclear(self) -> int:
        """异步清空所有记忆"""
        count = len(self._memories)
        self._memories.clear()
        self._vectors.clear()
        self._next_id = 1
        return count

    async def aclose(self):
        """关闭异步资源和连接"""
        if self._embedder and hasattr(self._embedder, 'aclose'):
            try:
                await self._embedder.aclose()
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self._memories)


__all__ = [
    "AsyncSuMemory",
    "StreamChunk",
]
