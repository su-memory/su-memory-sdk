"""
su-memory SDK Embedding模块
支持多种embedding后端：MiniMax-M2、OpenAI、本地模型
"""
import os
import time
import math
from typing import List, Dict
from dataclasses import dataclass
import json

# 可选的异步支持
try:
    import aiohttp
    ASYNCIO_AVAILABLE = True
except ImportError:
    ASYNCIO_AVAILABLE = False


@dataclass
class EmbeddingResult:
    """Embedding结果"""
    embedding: List[float]
    model: str
    tokens: int
    latency_ms: float


class EmbeddingBackend:
    """Embedding后端基类"""

    def encode(self, text: str) -> List[float]:
        raise NotImplementedError

    async def aencode(self, text: str) -> EmbeddingResult:
        raise NotImplementedError


class MiniMaxEmbedding(EmbeddingBackend):
    """
    MiniMax-M2 Embedding后端

    使用MiniMax的embo-01模型生成文本向量
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://api.minimax.chat/v1",
        model: str = "embo-01",
        dims: int = 1024
    ):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.dims = dims

    def encode(self, text: str) -> List[float]:
        """同步编码"""
        # 简化的hash-based fallback（当无API Key时使用）
        if not self.api_key:
            return self._hash_embedding(text)

        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "text": text
        }

        start = time.time()

        try:
            resp = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=10
            )
            resp.raise_for_status()

            data = resp.json()
            latency = (time.time() - start) * 1000

            return EmbeddingResult(
                embedding=data.get("embedding", []),
                model=self.model,
                tokens=data.get("tokens", 0),
                latency_ms=latency
            )
        except Exception as e:
            print(f"MiniMax embedding failed: {e}, using fallback")
            return self._hash_embedding(text)

    async def aencode(self, text: str) -> EmbeddingResult:
        """异步编码"""
        if not self.api_key or not ASYNCIO_AVAILABLE:
            return EmbeddingResult(
                embedding=self._hash_embedding(text),
                model="hash_fallback",
                tokens=0,
                latency_ms=0
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "text": text
        }

        start = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    data = await resp.json()
                    latency = (time.time() - start) * 1000

                    return EmbeddingResult(
                        embedding=data.get("embedding", []),
                        model=self.model,
                        tokens=data.get("tokens", 0),
                        latency_ms=latency
                    )
        except Exception as e:
            print(f"MiniMax embedding failed: {e}")
            return EmbeddingResult(
                embedding=self._hash_embedding(text),
                model="hash_fallback",
                tokens=0,
                latency_ms=(time.time() - start) * 1000
            )

    def _hash_embedding(self, text: str) -> List[float]:
        """Hash-based embedding fallback"""

        # 简单的hash到固定维度向量
        vec = [0.0] * self.dims

        # 使用多个hash函数生成不同维度的值
        for i, char in enumerate(text):
            char_ord = ord(char)
            hash_idx = char_ord % self.dims
            vec[hash_idx] += 1.0

        # 归一化
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec


class OpenAIEmbedding(EmbeddingBackend):
    """
    OpenAI Embedding后端
    支持 text-embedding-3-small 等模型
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        dims: int = 1536
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.dims = dims

    def encode(self, text: str) -> List[float]:
        """同步编码"""
        if not self.api_key:
            return self._simple_embedding(text)

        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "input": text
        }

        start = time.time()

        try:
            resp = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=10
            )
            resp.raise_for_status()

            data = resp.json()
            latency = (time.time() - start) * 1000

            embedding = data["data"][0]["embedding"]

            # 截取到指定维度
            if len(embedding) > self.dims:
                embedding = embedding[:self.dims]

            return EmbeddingResult(
                embedding=embedding,
                model=self.model,
                tokens=data.get("usage", {}).get("total_tokens", 0),
                latency_ms=latency
            )
        except Exception as e:
            print(f"OpenAI embedding failed: {e}")
            return self._simple_embedding(text)

    async def aencode(self, text: str) -> EmbeddingResult:
        """异步编码"""
        return self.encode(text)  # 简化实现

    def _simple_embedding(self, text: str) -> List[float]:
        """简单embedding fallback"""
        vec = [0.0] * self.dims
        for i, char in enumerate(text):
            vec[i % self.dims] += ord(char)

        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec


class LocalEmbedding(EmbeddingBackend):
    """
    本地Embedding后端
    使用sentence-transformers模型
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dims = 384

    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dims = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

    def encode(self, text: str) -> List[float]:
        """同步编码"""
        self._load_model()

        start = time.time()
        embedding = self._model.encode(text, convert_to_numpy=True)
        latency = (time.time() - start) * 1000

        return EmbeddingResult(
            embedding=embedding.tolist(),
            model=self.model_name,
            tokens=len(text) // 4,  # 估算
            latency_ms=latency
        )

    async def aencode(self, text: str) -> EmbeddingResult:
        """异步编码"""
        # sentence-transformers不支持真正的异步
        return self.encode(text)


class OllamaEmbedding(EmbeddingBackend):
    """
    Ollama本地Embedding后端
    使用Ollama运行的开源模型（如bge-m3）
    """

    def __init__(
        self,
        model: str = "bge-m3",
        base_url: str = "http://localhost:11434",
        dims: int = 1024
    ):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.dims = dims
        self._api_endpoint = f"{self.base_url}/api/embed"

    def encode(self, text: str) -> List[float]:
        """同步编码"""
        import urllib.request
        import urllib.error

        start = time.time()

        payload = {
            "model": self.model,
            "input": text
        }

        try:
            req = urllib.request.Request(
                self._api_endpoint,
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                embeddings = data.get("embeddings", [])

                if embeddings and len(embeddings) > 0:
                    embedding = embeddings[0]
                    # 更新维度信息
                    self.dims = len(embedding)
                    (time.time() - start) * 1000

                    # 返回 List[float] 而不是 EmbeddingResult
                    return embedding
                else:
                    raise ValueError("No embeddings returned")

        except urllib.error.URLError as e:
            print(f"Ollama request failed: {e}")
            return self._fallback_embedding(text)
        except Exception as e:
            print(f"Ollama encoding failed: {e}")
            return self._fallback_embedding(text)

    def encode_batch(self, texts: list) -> list:
        """Batch encode multiple texts in a single API call.
        
        Ollama API natively supports batch input for embeddings,
        making this 50-100x faster than sequential encode() calls.
        """
        import urllib.request, urllib.error
        
        if not texts:
            return []
        
        payload = {
            "model": self.model,
            "input": texts  # Ollama accepts list for batch
        }
        
        try:
            req = urllib.request.Request(
                self._api_endpoint,
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                embeddings = data.get("embeddings", [])
                
                if embeddings and len(embeddings) > 0:
                    self.dims = len(embeddings[0])
                    return embeddings
                else:
                    return [self._fallback_embedding(t) for t in texts]
                    
        except Exception as e:
            # Fallback to sequential
            return [self.encode(t) for t in texts]

    def _fallback_embedding(self, text: str) -> List[float]:
        """Fallback hash embedding"""

        vec = [0.0] * self.dims
        for i, char in enumerate(text):
            char_ord = ord(char)
            hash_idx = char_ord % self.dims
            vec[hash_idx] += 1.0

        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]

        # 返回 List[float] 而不是 EmbeddingResult
        return vec

    async def aencode(self, text: str) -> EmbeddingResult:
        """异步编码（使用线程池）"""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.encode, text)
            return future.result()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算余弦相似度"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def rrf_fusion(
    results_list: List[List[tuple]],
    k: int = 60,
    use_score_weight: bool = True,
    method_weights: List[float] = None
) -> List[tuple]:
    """
    增强版 Reciprocal Rank Fusion (RRF)
    用于融合多个检索结果，支持分数权重和方法权重

    Args:
        results_list: 多个检索结果列表，每个元素是 (id, score) 元组
        k: RRF常数，通常60
        use_score_weight: 是否使用原始分数作为权重
        method_weights: 各检索方法的自定义权重
    """
    scores = {}

    n_methods = len(results_list)
    default_weights = [1.0] * n_methods if not method_weights else method_weights

    for method_idx, results in enumerate(results_list):
        method_weight = default_weights[method_idx]

        for rank, (doc_id, original_score) in enumerate(results):
            # RRF基础得分
            rrf_score = 1 / (k + rank + 1)

            # 分数权重
            if use_score_weight and original_score > 0:
                # 使用对数缩放避免极端分数主导
                score_weight = 1 + math.log1p(original_score)
            else:
                score_weight = 1.0

            # 综合得分
            combined_score = rrf_score * score_weight * method_weight
            scores[doc_id] = scores.get(doc_id, 0) + combined_score

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores


def weighted_combination_fusion(
    results_list: List[List[tuple]],
    weights: List[float] = None
) -> List[tuple]:
    """
    加权组合融合
    直接将多个检索结果按权重组合

    Args:
        results_list: 多个检索结果列表
        weights: 各方法的权重（默认为等权重）
    """
    if not weights:
        weights = [1.0] * len(results_list)

    # 归一化权重
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    scores = {}
    max_scores = {}

    # 第一遍：收集分数
    for method_idx, results in enumerate(results_list):
        for doc_id, score in results:
            if doc_id not in max_scores:
                max_scores[doc_id] = 0
            max_scores[doc_id] = max(max_scores[doc_id], score)

    # 第二遍：计算加权得分
    for method_idx, results in enumerate(results_list):
        for doc_id, score in results:
            # 归一化分数
            max_s = max_scores.get(doc_id, 1.0)
            normalized = score / max_s if max_s > 0 else 0

            scores[doc_id] = scores.get(doc_id, 0) + normalized * weights[method_idx]

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores


class EmbeddingManager:
    """
    Embedding管理器
    统一管理多个embedding后端，支持自动检测
    """

    # 支持的后端列表
    SUPPORTED_BACKENDS = ["minimax", "openai", "local", "ollama", "chroma"]

    def __init__(self, backend: str = "auto", **kwargs):
        """
        初始化 Embedding 管理器

        Args:
            backend: 后端类型，"auto" 表示自动检测可用后端
            **kwargs: 后端配置参数
        """
        self.backend_name = backend
        self._backend = None
        self._backend_info = None

        if backend == "auto":
            self._auto_detect(**kwargs)
        else:
            self._init_backend(backend, **kwargs)

    def _auto_detect(self, **kwargs):
        """自动检测可用的后端"""
        # 按优先级尝试各后端
        preferred_order = ["ollama", "openai", "minimax", "local"]

        # 检查环境变量指定的优先后端
        env_preferred = os.environ.get("SU_MEMORY_EMBEDDING_PREFERRED", "")
        if env_preferred:
            preferred_order = [env_preferred] + [b for b in preferred_order if b != env_preferred]

        errors = []

        for backend in preferred_order:
            try:
                if self._test_backend(backend):
                    self._init_backend(backend, **kwargs)
                    print(f"  ✅ 自动选择 Embedding 后端: {backend}")
                    return
            except Exception as e:
                errors.append(f"{backend}: {str(e)}")

        # 全部失败，使用 hash fallback
        print("  ⚠️  未检测到可用嵌入服务，使用 Hash Fallback")
        print("     这将使用简单的文本哈希作为向量表示，功能受限")
        print("\n  推荐安装以下服务之一:")
        print("    1. Ollama (推荐): pip install httpx && ollama serve && ollama pull nomic-embed-text")
        print("    2. OpenAI: pip install openai && export OPENAI_API_KEY=sk-xxx")
        print("    3. MiniMax: export MINIMAX_API_KEY=xxx && export MINIMAX_GROUP_ID=xxx")
        print("    4. 本地模型: pip install sentence-transformers")

        self._backend = HashFallbackEmbedding()
        self.backend_name = "hash_fallback"
        self._backend_info = {"backend": "hash_fallback", "dims": 256}

    def _test_backend(self, backend: str) -> bool:
        """测试后端是否可用"""
        try:
            test_text = "test"

            if backend == "ollama":
                import urllib.request
                req = urllib.request.Request(
                    "http://localhost:11434/api/tags",
                    method="GET"
                )
                with urllib.request.urlopen(req, timeout=5):
                    return True

            elif backend == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    return False
                import requests
                resp = requests.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": "text-embedding-3-small", "input": test_text},
                    timeout=10
                )
                return resp.status_code == 200

            elif backend == "minimax":
                api_key = os.environ.get("MINIMAX_API_KEY")
                group_id = os.environ.get("MINIMAX_GROUP_ID")
                if not api_key or not group_id:
                    return False
                import requests
                resp = requests.post(
                    "https://api.minimax.chat/v1/embeddings",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": "embo-01", "text": test_text},
                    timeout=10
                )
                return resp.status_code == 200

            elif backend == "local":
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                model.encode(test_text)
                return True

            return False

        except Exception:
            return False

    def _init_backend(self, backend: str, **kwargs):
        """初始化指定后端"""
        if backend == "minimax":
            self._backend = MiniMaxEmbedding(**kwargs)
            self._backend_info = {"backend": "minimax", "dims": 1024}
        elif backend == "openai":
            self._backend = OpenAIEmbedding(**kwargs)
            self._backend_info = {"backend": "openai", "dims": 1536}
        elif backend == "local":
            self._backend = LocalEmbedding(**kwargs)
            self._backend_info = {"backend": "local", "dims": 384}
        elif backend == "ollama":
            self._backend = OllamaEmbedding(**kwargs)
            self._backend_info = {"backend": "ollama", "dims": 1024}
        elif backend == "chroma":
            self._backend = ChromaBackendEmbedding(**kwargs)
            self._backend_info = {"backend": "chroma", "dims": 768}
        else:
            raise ValueError(f"Unknown backend: {backend}. Supported: {self.SUPPORTED_BACKENDS}")

        self.backend_name = backend

    def encode(self, text: str) -> List[float]:
        """编码文本"""
        if self._backend is None:
            return HashFallbackEmbedding().encode(text)

        result = self._backend.encode(text)
        return result.embedding if isinstance(result, EmbeddingResult) else result

    async def aencode(self, text: str) -> EmbeddingResult:
        """异步编码"""
        if self._backend is None:
            return HashFallbackEmbedding().aencode(text)
        return await self._backend.aencode(text)

    @property
    def dims(self) -> int:
        """获取向量维度"""
        if hasattr(self._backend, 'dims'):
            return self._backend.dims
        if self._backend_info:
            return self._backend_info.get("dims", 1024)
        return 1024

    def get_info(self) -> Dict:
        """获取后端信息"""
        return {
            "backend": self.backend_name,
            "dims": self.dims,
            "info": self._backend_info or {}
        }


class HashFallbackEmbedding(EmbeddingBackend):
    """Hash Fallback Embedding - 当没有可用后端时使用"""

    def __init__(self, dims: int = 256):
        self.dims = dims

    def encode(self, text: str) -> List[float]:
        """基于文本特征的简单嵌入"""

        vec = [0.0] * self.dims

        # 使用字符频率作为特征
        for i, char in enumerate(text):
            idx = (ord(char) * 31 + i) % self.dims
            vec[idx] += 1.0

        # 归一化
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec

    async def aencode(self, text: str) -> EmbeddingResult:
        """异步编码"""
        return EmbeddingResult(
            embedding=self.encode(text),
            model="hash_fallback",
            tokens=0,
            latency_ms=0
        )


class ChromaBackendEmbedding(EmbeddingBackend):
    """ChromaDB Backend Embedding"""

    def __init__(self, collection_name: str = "su_memory", persist_directory: str = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._base_embedder = OllamaEmbedding()
        self._chroma_client = None
        self._collection = None
        self.dims = 768

    def encode(self, text: str) -> List[float]:
        """编码文本"""
        return self._base_embedder.encode(text)

    async def aencode(self, text: str) -> EmbeddingResult:
        """异步编码"""
        result = self._base_embedder.encode(text)
        return EmbeddingResult(
            embedding=result,
            model="ollama-chroma",
            tokens=0,
            latency_ms=0
        )


# 保持向后兼容的导出
__all__ = [
    "EmbeddingManager",
    "EmbeddingResult",
    "MiniMaxEmbedding",
    "OpenAIEmbedding",
    "LocalEmbedding",
    "OllamaEmbedding",
    "cosine_similarity",
    "rrf_fusion",
    "weighted_combination_fusion",
]
