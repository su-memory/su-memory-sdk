"""
su-memory 异步嵌入服务层

提供与同步 EmbeddingProvider 对应的异步接口，支持：
- Ollama (本地异步 HTTP)
- OpenAI (AsyncOpenAI)
- MiniMax (AsyncOpenAI + MiniMax base_url)
- sentence-transformers (CPU → asyncio.to_thread)
- TF-IDF / Hash fallback (最终兜底)

架构：
- AsyncEmbeddingProvider: 异步嵌入抽象基类
- AsyncEmbeddingFactory: 自动检测可用后端
- 嵌入缓存: 复用 EmbeddingCache，通过 asyncio.to_thread 调用
"""

from __future__ import annotations

import asyncio
import os
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from su_memory.exceptions import SuMemoryError, ErrorCode

logger = logging.getLogger(__name__)


# =============================================================================
# AsyncEmbeddingProvider — 异步嵌入抽象基类
# =============================================================================

class AsyncEmbeddingProvider(ABC):
    """异步嵌入服务抽象基类

    与同步 EmbeddingProvider 保持相同的语义，
    但所有方法均为 async。
    """

    @abstractmethod
    async def aembed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """批量异步生成嵌入向量

        Args:
            texts: 文本列表
            model: 模型名称（可选）

        Returns:
            向量列表，顺序与 texts 一致
        """
        pass

    @abstractmethod
    async def aembed_single(self, text: str, model: Optional[str] = None) -> List[float]:
        """单条文本异步嵌入

        Args:
            text: 文本
            model: 模型名称（可选）

        Returns:
            嵌入向量
        """
        pass

    @abstractmethod
    async def ais_available(self) -> bool:
        """异步检查服务是否可用"""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """获取默认模型名称"""
        pass

    @property
    @abstractmethod
    def dims(self) -> int:
        """向量维度"""
        pass


# =============================================================================
# OllamaAsyncEmbedder — httpx.AsyncClient 本地异步嵌入
# =============================================================================

class OllamaAsyncEmbedder(AsyncEmbeddingProvider):
    """Ollama 异步嵌入服务 (httpx.AsyncClient)"""

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_DIMS = 768  # nomic-embed-text 默认维度

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self._model = model or self.DEFAULT_MODEL
        self._timeout = timeout
        self._client: Optional[Any] = None

    def _get_client(self):
        """获取或创建 httpx.AsyncClient"""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self._timeout,
                )
            except ImportError:
                raise SuMemoryError(
                    ErrorCode.EMBED_UNAVAILABLE,
                    detail="请安装 httpx: pip install httpx",
                ) from None
        return self._client

    async def aembed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        if not texts:
            return []

        model = model or self._model
        client = self._get_client()

        results = []
        for text in texts:
            response = await client.post("/api/embeddings", json={
                "model": model,
                "prompt": text,
            })
            response.raise_for_status()
            data = response.json()
            results.append(data["embedding"])

        return results

    async def aembed_single(self, text: str, model: Optional[str] = None) -> List[float]:
        results = await self.aembed([text], model)
        return results[0] if results else []

    async def ais_available(self) -> bool:
        try:
            client = self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def get_default_model(self) -> str:
        return self._model

    @property
    def dims(self) -> int:
        return self.DEFAULT_DIMS

    async def aclose(self):
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# OpenAIAsyncEmbedder — openai.AsyncOpenAI
# =============================================================================

class OpenAIAsyncEmbedder(AsyncEmbeddingProvider):
    """OpenAI 异步嵌入服务 (AsyncOpenAI)"""

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMS = 1536

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get(
            "OPENAI_API_BASE_URL", "https://api.openai.com/v1"
        )
        self._model = model or self.DEFAULT_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise SuMemoryError(
                    ErrorCode.EMBED_UNAVAILABLE,
                    detail="请安装 openai: pip install openai",
                ) from None
        return self._client

    async def aembed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        if not texts:
            return []

        model = model or self._model
        client = self._get_client()

        response = await client.embeddings.create(
            model=model,
            input=texts,
        )

        return [d.embedding for d in response.data]

    async def aembed_single(self, text: str, model: Optional[str] = None) -> List[float]:
        results = await self.aembed([text], model)
        return results[0] if results else []

    async def ais_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            await client.embeddings.create(
                model=self._model,
                input=["test"],
            )
            return True
        except Exception:
            return False

    def get_default_model(self) -> str:
        return self._model

    @property
    def dims(self) -> int:
        return self.DEFAULT_DIMS

    async def aclose(self):
        if self._client:
            await self._client.close()
            self._client = None


# =============================================================================
# MiniMaxAsyncEmbedder — AsyncOpenAI + MiniMax base_url
# =============================================================================

class MiniMaxAsyncEmbedder(AsyncEmbeddingProvider):
    """MiniMax 异步嵌入服务 (AsyncOpenAI + MiniMax API)"""

    DEFAULT_MODEL = "embo-01"
    DEFAULT_DIMS = 1536

    def __init__(
        self,
        api_key: Optional[str] = None,
        group_id: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        self.group_id = group_id or os.environ.get("MINIMAX_GROUP_ID")
        self.base_url = "https://api.minimax.chat/v1"
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise SuMemoryError(
                    ErrorCode.EMBED_UNAVAILABLE,
                    detail="请安装 openai: pip install openai",
                ) from None
        return self._client

    async def aembed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        if not texts:
            return []

        model = model or self.DEFAULT_MODEL
        client = self._get_client()

        response = await client.embeddings.create(
            model=model,
            input=texts,
            extra_body={"dimension": self.DEFAULT_DIMS},
        )

        return [d.embedding for d in response.data]

    async def aembed_single(self, text: str, model: Optional[str] = None) -> List[float]:
        results = await self.aembed([text], model)
        return results[0] if results else []

    async def ais_available(self) -> bool:
        if not self.api_key or not self.group_id:
            return False
        try:
            client = self._get_client()
            await client.embeddings.create(
                model=self.DEFAULT_MODEL,
                input=["test"],
            )
            return True
        except Exception:
            return False

    def get_default_model(self) -> str:
        return self.DEFAULT_MODEL

    @property
    def dims(self) -> int:
        return self.DEFAULT_DIMS

    async def aclose(self):
        if self._client:
            await self._client.close()
            self._client = None


# =============================================================================
# SentenceTransformersAsyncEmbedder — CPU 密集型 → asyncio.to_thread
# =============================================================================

class SentenceTransformersAsyncEmbedder(AsyncEmbeddingProvider):
    """sentence-transformers 异步嵌入 (CPU → asyncio.to_thread)"""

    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    DEFAULT_DIMS = 384

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or os.environ.get(
            "SU_MEMORY_EMBEDDING_MODEL", self.DEFAULT_MODEL
        )
        self._model = None
        self._dims = self.DEFAULT_DIMS

    def _ensure_model(self):
        if self._model is None:
            try:
                import sentence_transformers
                self._model = sentence_transformers.SentenceTransformer(self._model_name)
                self._dims = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise SuMemoryError(
                    ErrorCode.EMBED_UNAVAILABLE,
                    detail="请安装 sentence-transformers: pip install sentence-transformers",
                ) from None
        return self._model

    async def aembed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        if not texts:
            return []

        st_model = self._ensure_model()

        def _encode_sync():
            return st_model.encode(texts, convert_to_numpy=True).tolist()

        embeddings = await asyncio.to_thread(_encode_sync)
        return embeddings

    async def aembed_single(self, text: str, model: Optional[str] = None) -> List[float]:
        results = await self.aembed([text], model)
        return results[0] if results else []

    async def ais_available(self) -> bool:
        try:
            self._ensure_model()
            return True
        except Exception:
            return False

    def get_default_model(self) -> str:
        return self._model_name

    @property
    def dims(self) -> int:
        return self._dims


# =============================================================================
# TF-IDF Async Fallback — CPU → asyncio.to_thread
# =============================================================================

class TfidfAsyncEmbedder(AsyncEmbeddingProvider):
    """TF-IDF 异步嵌入回退 (CPU → asyncio.to_thread)"""

    DEFAULT_DIMS = 256

    def __init__(self):
        self._dims = self.DEFAULT_DIMS
        self._corpus: List[str] = []
        self._vectorizer = None
        self._fitted = False

    def _build_vectorizer(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                max_features=self._dims,
                analyzer='char_wb',
                ngram_range=(2, 4),
            )
        except ImportError:
            raise SuMemoryError(
                ErrorCode.EMBED_UNAVAILABLE,
                detail="请安装 scikit-learn: pip install scikit-learn",
            ) from None

    async def aembed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        if not texts:
            return []

        import hashlib
        import struct

        def _encode_sync():
            results = []
            for text in texts:
                if not self._fitted or len(self._corpus) < 50:
                    self._corpus.append(text)
                    vec = self._hash_vec(text)
                    results.append(vec)
                    continue

                if self._vectorizer is None:
                    self._build_vectorizer()
                    self._vectorizer.fit(self._corpus + texts)

                try:
                    v = self._vectorizer.transform([text]).toarray()[0]
                    vec = list(v[:self._dims])
                    if len(vec) < self._dims:
                        vec += [0.0] * (self._dims - len(vec))
                    norm = (sum(x * x for x in vec)) ** 0.5
                    if norm > 0:
                        vec = [x / norm for x in vec]
                    results.append(vec)
                except Exception:
                    results.append(self._hash_vec(text))
            return results

        return await asyncio.to_thread(_encode_sync)

    def _hash_vec(self, text: str) -> List[float]:
        import hashlib
        import struct
        vec = [0.0] * self._dims
        for i, ch in enumerate(text):
            h = hashlib.sha256(f"{i}:{ch}".encode()).digest()[:2]
            idx = struct.unpack('<H', h)[0] % self._dims
            vec[idx] += 1.0
        norm = (sum(v * v for v in vec)) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    async def aembed_single(self, text: str, model: Optional[str] = None) -> List[float]:
        results = await self.aembed([text], model)
        return results[0] if results else []

    async def ais_available(self) -> bool:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            return True
        except ImportError:
            return False

    def get_default_model(self) -> str:
        return "tfidf-256"

    @property
    def dims(self) -> int:
        return self._dims


# =============================================================================
# AsyncEmbeddingFactory — 自动检测异步可用后端
# =============================================================================

class AsyncEmbeddingFactory:
    """异步嵌入服务工厂

    按优先级自动检测可用后端：Ollama → OpenAI → MiniMax → s-t → TF-IDF
    """

    _providers: Dict[str, type] = {
        "ollama": OllamaAsyncEmbedder,
        "openai": OpenAIAsyncEmbedder,
        "minimax": MiniMaxAsyncEmbedder,
        "sentence_transformers": SentenceTransformersAsyncEmbedder,
    }

    @classmethod
    async def create(cls, provider: str = "auto", **kwargs) -> AsyncEmbeddingProvider:
        """创建异步嵌入服务

        Args:
            provider: 提供商名称，"auto" 表示自动检测
            **kwargs: 提供商配置参数

        Returns:
            异步嵌入服务实例
        """
        if provider == "auto":
            return await cls.auto_detect(**kwargs)

        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise SuMemoryError(
                ErrorCode.CONFIG_INVALID_PARAM,
                param="provider",
                value=provider,
                reason=f"可用选项: {available}",
            )

        instance = cls._providers[provider](**kwargs)
        if not await instance.ais_available():
            logger.warning(f"异步嵌入服务 {provider} 不可用，回退到 TF-IDF")
            return TfidfAsyncEmbedder()
        return instance

    @classmethod
    async def auto_detect(cls, preferred: Optional[List[str]] = None) -> AsyncEmbeddingProvider:
        """自动检测可用异步嵌入服务

        按优先级尝试各服务，返回第一个可用的。

        Args:
            preferred: 优先尝试的服务列表

        Returns:
            可用的异步嵌入服务实例
        """
        default_order = ["ollama", "openai", "minimax", "sentence_transformers"]
        order = preferred or default_order

        errors = []
        for name in order:
            try:
                provider_class = cls._providers.get(name)
                if not provider_class:
                    continue

                provider = provider_class()
                if await provider.ais_available():
                    logger.info(f"✅ 自动选择异步嵌入服务: {name}")
                    return provider
            except Exception as e:
                errors.append(f"{name}: {str(e)}")

        logger.warning(
            f"所有异步嵌入后端不可用，回退到 TF-IDF。错误:\n" +
            "\n".join(errors)
        )
        return TfidfAsyncEmbedder()

    @classmethod
    def list_providers(cls) -> List[str]:
        """列出所有支持的异步提供商"""
        return list(cls._providers.keys()) + ["tfidf"]


# =============================================================================
# 便捷函数
# =============================================================================

async def get_async_embedder(provider: str = "auto", **kwargs) -> AsyncEmbeddingProvider:
    """获取异步嵌入服务"""
    return await AsyncEmbeddingFactory.create(provider, **kwargs)


# =============================================================================
# 异步嵌入缓存适配器
# =============================================================================

class AsyncEmbeddingCache:
    """异步嵌入缓存包装器

    将同步 EmbeddingCache 适配为异步接口，通过 asyncio.to_thread 调用。
    """

    def __init__(self, max_entries: int = 10000, ttl_seconds: int = 3600):
        from su_memory._sys._embedding_cache import EmbeddingCache
        self._cache = EmbeddingCache(max_entries=max_entries, ttl_seconds=ttl_seconds)

    async def aget(self, key: str):
        """异步获取缓存"""
        return await asyncio.to_thread(self._cache.get, key)

    async def aset(self, key: str, value: List[float]):
        """异步设置缓存"""
        return await asyncio.to_thread(self._cache.set, key, value)

    async def aget_or_compute(
        self,
        key: str,
        compute_fn,
        *args,
        **kwargs,
    ) -> List[float]:
        """异步获取或计算缓存

        Args:
            key: 缓存键
            compute_fn: 异步计算函数 async def fn(*args, **kwargs) -> List[float]
            *args, **kwargs: 传递给 compute_fn

        Returns:
            嵌入向量
        """
        cached = await self.aget(key)
        if cached is not None:
            return cached

        result = await compute_fn(*args, **kwargs)
        await self.aset(key, result)
        return result

    async def aget_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return await asyncio.to_thread(self._cache.get_stats)

    async def aclear(self):
        """清空缓存"""
        await asyncio.to_thread(self._cache.clear)


__all__ = [
    "AsyncEmbeddingProvider",
    "OllamaAsyncEmbedder",
    "OpenAIAsyncEmbedder",
    "MiniMaxAsyncEmbedder",
    "SentenceTransformersAsyncEmbedder",
    "TfidfAsyncEmbedder",
    "AsyncEmbeddingFactory",
    "AsyncEmbeddingCache",
    "get_async_embedder",
]
