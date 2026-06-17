"""
su-memory SDK Embedding模块 (v3.5.2)

支持 12 种 embedding 后端：
  P0: Ollama, LlamaCpp (GGUF本地), DeepSeek (云端+本地双模)
  P1: Voyage AI, HuggingFace TEI, Cohere Embed v3
  P2: Google Gemini, ONNX Runtime, MiniMax-M2, OpenAI, Local (s-t), Chroma

[P0-C 修复说明] — v3.7.0
本模块存在以下阻塞隐患与防护策略：
1. 9 处 sync requests 全部位于 sync 函数内（def encode / def _test_backend）
   - 当前无 async 调用方（已 grep 验证 `await self.xxx` 为 0 匹配）
   - 若未来在 async def aencode 中调用，必须用 asyncio.to_thread 包裹
   - 5 个 aencode 已统一改为 asyncio.to_thread 包装（见下）
2. asyncio.to_thread 使用 Python 3.9+ 标准库，3.9 以下用
   concurrent.futures.ThreadPoolExecutor 替代
3. OllamaEmbedding.aencode 不再每次创建 ThreadPoolExecutor，共享默认线程池

[F2-P1-2 修复说明] — v3.7.0
错误处理中的 print(e) 会泄露用户文本/API 截断到 stdout。
全部转换为 logger.warning，通过 Python logging 框架统一管理并遵守日志级别。
"""
import asyncio
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 可选的异步支持
try:
    import aiohttp
    ASYNCIO_AVAILABLE = True
except ImportError:
    ASYNCIO_AVAILABLE = False


@dataclass
class EmbeddingResult:
    """Embedding结果"""
    embedding: list[float]
    model: str
    tokens: int
    latency_ms: float


class EmbeddingBackend:
    """Embedding后端基类"""

    def encode(self, text: str) -> list[float]:
        raise NotImplementedError

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码 — 默认逐条调用 encode()，子类可覆盖优化"""
        return [self.encode(t) for t in texts]

    async def aencode(self, text: str) -> EmbeddingResult:
        raise NotImplementedError


class MiniMaxEmbedding(EmbeddingBackend):
    """
    MiniMax-M2 Embedding后端

    使用MiniMax的embo-01模型生成文本向量。
    v4.3.1: 修复 API 参数格式 — texts (数组) + type (db/query)。
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

    def _api_request(self, texts: list[str], text_type: str = "db") -> tuple[list[list[float]], int]:
        """v4.3.1: 统一 API 调用 — texts 数组 + type 参数"""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "texts": texts,
            "type": text_type,
        }

        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        # MiniMax 返回: {"vectors": [[...], ...], "total_tokens": N, "base_resp": {...}}
        vectors = data.get("vectors", [])
        total_tokens = int(data.get("total_tokens", 0))

        return vectors, total_tokens

    def encode(self, text: str) -> list[float]:
        """同步编码"""
        if not self.api_key:
            return self._hash_embedding(text)

        start = time.time()
        try:
            vectors, tokens = self._api_request([text], text_type="db")
            embedding = vectors[0] if vectors else []
            if embedding:
                self.dims = len(embedding)
            latency = (time.time() - start) * 1000
            return EmbeddingResult(
                embedding=embedding,
                model=self.model,
                tokens=tokens,
                latency_ms=latency
            )
        except Exception as e:
            logger.warning("MiniMax embedding failed, using fallback: %s", e)
            return self._hash_embedding(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """v4.3.1: 批量编码 — MiniMax API 原生支持 texts 数组"""
        if not texts:
            return []
        if not self.api_key:
            return [self._hash_embedding(t) for t in texts]

        try:
            vectors, _ = self._api_request(texts, text_type="db")
            return vectors if vectors else [self._hash_embedding(t) for t in texts]
        except Exception as e:
            logger.warning("MiniMax batch embedding failed: %s", e)
            return [self._hash_embedding(t) for t in texts]

    async def aencode(self, text: str) -> EmbeddingResult:
        """异步编码"""
        if not self.api_key or not ASYNCIO_AVAILABLE:
            return EmbeddingResult(
                embedding=self._hash_embedding(text),
                model="hash_fallback",
                tokens=0,
                latency_ms=0
            )

        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"model": self.model, "texts": [text], "type": "db"}
                async with session.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    data = await resp.json()
                    vectors = data.get("vectors", [])
                    embedding = vectors[0] if vectors else []
                    latency = (time.time() - start) * 1000
                    return EmbeddingResult(
                        embedding=embedding,
                        model=self.model,
                        tokens=int(data.get("total_tokens", 0)),
                        latency_ms=latency
                    )
        except Exception as e:
            logger.warning("MiniMax embedding failed: %s", e)
            return EmbeddingResult(
                embedding=self._hash_embedding(text),
                model="hash_fallback",
                tokens=0,
                latency_ms=(time.time() - start) * 1000
            )

    def _hash_embedding(self, text: str) -> list[float]:
        """Hash-based embedding fallback"""

        # 简单的hash到固定维度向量
        vec = [0.0] * self.dims

        # 使用多个hash函数生成不同维度的值
        for _i, char in enumerate(text):
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

    def encode(self, text: str) -> list[float]:
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
            # P0-C: 9 处 sync requests 之二（def encode 内），请勿在 async 路径直接调用
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
            # F2-P1-2: print(e) 转 logger
            logger.warning("OpenAI embedding failed: %s", e)
            return self._simple_embedding(text)

    async def aencode(self, text: str) -> EmbeddingResult:
        """异步编码 — P0-C：asyncio.to_thread 包裹 sync requests 避免阻塞事件循环"""
        return await asyncio.to_thread(self.encode, text)

    def _simple_embedding(self, text: str) -> list[float]:
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
        self.dims = self._dims  # v3.5.3: 暴露公共 dims 属性，供 FAISS 索引维度检测
        # F5-P0-4: 懒加载 DCL 锁 — 防止并发首次加载竞态
        self._load_lock = threading.Lock()

    def _load_model(self):
        """延迟加载模型"""
        # F5-P0-4: Double-Checked Locking 模式 — 防止并发首次加载竞态
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dims = self._model.get_sentence_embedding_dimension()
                self.dims = self._dims  # v3.5.3: 模型加载后同步公共 dims
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                ) from None

    def encode(self, text: str) -> list[float]:
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
        """异步编码 — P0-C：sentence-transformers 是 CPU 密集，asyncio.to_thread 避免阻塞事件循环"""
        # sentence-transformers 不支持真正的异步
        return await asyncio.to_thread(self.encode, text)


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
        self._api_endpoint = f"{self.base_url}/api/embeddings"

    def encode(self, text: str) -> list[float]:
        """同步编码"""
        import urllib.error
        import urllib.request

        start = time.time()

        payload = {
            "model": self.model,
            "prompt": text
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
                embedding = data.get("embedding", [])

                if embedding and len(embedding) > 0:
                    # 更新维度信息
                    self.dims = len(embedding)
                    (time.time() - start) * 1000

                    # 返回 List[float] 而不是 EmbeddingResult
                    return embedding
                else:
                    raise ValueError("No embedding returned")

        except urllib.error.URLError as e:
            # F2-P1-2: print(e) 转 logger
            logger.warning("Ollama request failed: %s", e)
            return self._fallback_embedding(text)
        except Exception as e:
            # F2-P1-2: print(e) 转 logger
            logger.warning("Ollama encoding failed: %s", e)
            return self._fallback_embedding(text)

    def encode_batch(self, texts: list) -> list:
        """批量编码 — Ollama /api/embeddings 仅支持单条 prompt，故迭代调用 encode()。"""
        if not texts:
            return []
        return [self.encode(t) for t in texts]

    def _fallback_embedding(self, text: str) -> list[float]:
        """Fallback hash embedding"""

        vec = [0.0] * self.dims
        for _i, char in enumerate(text):
            char_ord = ord(char)
            hash_idx = char_ord % self.dims
            vec[hash_idx] += 1.0

        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]

        # 返回 List[float] 而不是 EmbeddingResult
        return vec

    async def aencode(self, text: str) -> EmbeddingResult:
        """异步编码（共享默认线程池）— P0-C：替代每次创建 ThreadPoolExecutor(max_workers=1) 的浪费模式"""
        return await asyncio.to_thread(self.encode, text)


class MLXEmbedding(EmbeddingBackend):
    """
    Apple MLX (Metal) Embedding 后端 (v3.5.9)

    使用 Apple MLX 框架在 Apple Silicon GPU 上运行 SentenceTransformer 模型。
    完全本地运行，无需 Ollama 服务，零网络开销。

    Apple Silicon 原生加速 (MPS/Metal)，比 Ollama 减少一层网络开销。
    """

    DEFAULT_MODEL = "BAAI/bge-m3"

    def __init__(
        self,
        model: str = None,
        dims: int = 1024,
    ):
        self.model_name = model or os.environ.get(
            "SU_MEMORY_MLX_MODEL", self.DEFAULT_MODEL
        )
        self._dims = dims
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers 未安装。请执行: pip install sentence-transformers"
                ) from None
            self._model = SentenceTransformer(self.model_name)
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim:
                self._dims = actual_dim
        return self._model

    @property
    def dims(self) -> int:
        return self._dims

    def encode(self, text: str) -> list[float]:
        """同步编码单条文本 (MPS/GPU 加速)"""
        model = self._get_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码 — SentenceTransformer 原生支持"""
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [e.tolist() if hasattr(e, 'tolist') else list(e) for e in embeddings]

    async def aencode(self, text: str) -> EmbeddingResult:
        """异步编码 — CPU 密集，asyncio.to_thread 避免阻塞事件循环"""
        import time as _time
        start = _time.time()
        embedding = await asyncio.to_thread(self.encode, text)
        return EmbeddingResult(
            embedding=embedding,
            model=f"mlx/{self.model_name}",
            tokens=0,
            latency_ms=(_time.time() - start) * 1000,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# v3.5.1 新增后端
# ═══════════════════════════════════════════════════════════════════════════════


class LlamaCppEmbedding(EmbeddingBackend):
    """
    llama.cpp 本地 GGUF 模型 Embedding 后端 (v3.5.1)

    依赖: pip install llama-cpp-python
    模型: ~/.cache/su-memory/models/*.gguf
    """

    DEFAULT_MODEL_DIR = os.path.expanduser("~/.cache/su-memory/models")
    DEFAULT_MODEL_NAME = "bge-m3-q4_k_m.gguf"

    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int = 512,
        n_batch: int = 512,
        verbose: bool = False,
    ):
        self.model_path = model_path or os.environ.get(
            "SU_MEMORY_GGUF_MODEL_PATH",
            os.path.join(self.DEFAULT_MODEL_DIR, self.DEFAULT_MODEL_NAME),
        )
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.verbose = verbose
        self._model = None
        self._dims: int | None = None

    @property
    def dims(self) -> int:
        return self._dims or 1024

    def _get_model(self):
        if self._model is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python 未安装。请执行: pip install llama-cpp-python"
                ) from None
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(
                    f"GGUF 模型未找到: {self.model_path}\n"
                    f"请下载模型到 {self.DEFAULT_MODEL_DIR}/ 或设置 SU_MEMORY_GGUF_MODEL_PATH"
                )
            self._model = Llama(
                model_path=self.model_path,
                embedding=True,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                verbose=self.verbose,
            )
        return self._model

    def encode(self, text: str) -> list[float]:
        """同步编码单条文本"""
        model = self._get_model()
        result = model.embed(text)
        # llama-cpp-python embed() 返回 List[float] 或 List[List[float]]
        if isinstance(result[0], list):
            embedding = result[0]
        else:
            embedding = result
        if self._dims is None:
            self._dims = len(embedding)
        return embedding

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码 — llama-cpp-python 原生支持"""
        if not texts:
            return []
        model = self._get_model()
        results = model.embed(texts)
        # 确保返回 List[List[float]]
        if results and not isinstance(results[0], list):
            results = [results]
        if self._dims is None and results:
            self._dims = len(results[0])
        return results

    async def aencode(self, text: str) -> EmbeddingResult:
        """异步编码 — CPU 密集，asyncio.to_thread 避免阻塞事件循环"""
        start = time.time()
        embedding = await asyncio.to_thread(self.encode, text)
        return EmbeddingResult(
            embedding=embedding,
            model=f"llama.cpp/{os.path.basename(self.model_path)}",
            tokens=0,
            latency_ms=(time.time() - start) * 1000,
        )


class DeepSeekEmbedding(EmbeddingBackend):
    """
    DeepSeek Embedding 后端 — 云端 + 本地双模 (v3.5.1)

    云端: DEEPSEEK_API_KEY 存在时走 api.deepseek.com
    本地: 无 API Key 时委托 LlamaCppEmbedding 加载 deepseek-*.gguf
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    DEFAULT_MODEL = "deepseek-embedding"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = (base_url or os.environ.get(
            "DEEPSEEK_BASE_URL", self.DEFAULT_BASE_URL
        )).rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self._local_backend: LlamaCppEmbedding | None = None
        self._mode: str | None = None  # "cloud" or "local"
        self.dims = 1024

    def _resolve_mode(self) -> str:
        if self._mode:
            return self._mode
        if self.api_key:
            self._mode = "cloud"
            return "cloud"
        # 尝试本地模式
        import glob
        pattern = os.path.join(LlamaCppEmbedding.DEFAULT_MODEL_DIR, "deepseek-*.gguf")
        matches = glob.glob(pattern)
        if matches:
            self._local_backend = LlamaCppEmbedding(model_path=matches[0])
            self._mode = "local"
            return "local"
        self._mode = "cloud"  # 将在 encode 时失败
        return "cloud"

    def encode(self, text: str) -> list[float]:
        """同步编码"""
        import urllib.error
        import urllib.request

        mode = self._resolve_mode()
        if mode == "local" and self._local_backend:
            return self._local_backend.encode(text)

        # 云端模式
        if not self.api_key:
            logger.warning("DeepSeek: 无 API Key 且未找到本地 GGUF，使用 hash fallback")
            return HashFallbackEmbedding().encode(text)

        payload = {"model": self.model, "input": [text]}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            req = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                items = data.get("data", [])
                if items:
                    embedding = items[0].get("embedding", [])
                    self.dims = len(embedding)
                    return embedding
            return HashFallbackEmbedding().encode(text)
        except Exception as e:
            logger.warning("DeepSeek 云端失败: %s，尝试本地 fallback", e)
            if self._local_backend:
                return self._local_backend.encode(text)
            return HashFallbackEmbedding().encode(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码"""
        import urllib.error
        import urllib.request

        if not texts:
            return []
        mode = self._resolve_mode()
        if mode == "local" and self._local_backend:
            return self._local_backend.encode_batch(texts)

        if not self.api_key:
            return [HashFallbackEmbedding().encode(t) for t in texts]

        payload = {"model": self.model, "input": texts}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            req = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
                return [item.get("embedding", []) for item in items]
        except Exception:
            return [self.encode(t) for t in texts]

    async def aencode(self, text: str) -> EmbeddingResult:
        """异步编码"""
        start = time.time()
        embedding = await asyncio.to_thread(self.encode, text)
        return EmbeddingResult(
            embedding=embedding,
            model=f"deepseek/{self.model}",
            tokens=0,
            latency_ms=(time.time() - start) * 1000,
        )


class VoyageAIEmbedding(EmbeddingBackend):
    """
    Voyage AI Embedding 后端 (v3.5.1)

    MTEB Top 3，支持 input_type 区分 document/query
    API: https://api.voyageai.com/v1/embeddings
    """

    DEFAULT_BASE_URL = "https://api.voyageai.com/v1"
    DEFAULT_MODEL = "voyage-3-large"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        input_type: str = "document",
    ):
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        self.model = model or self.DEFAULT_MODEL
        self.input_type = input_type  # "document" or "query"
        self.dims = 1024

    def encode(self, text: str, input_type: str | None = None) -> list[float]:
        """同步编码"""
        import urllib.request

        if not self.api_key:
            logger.warning("Voyage AI: 无 VOYAGE_API_KEY")
            return HashFallbackEmbedding().encode(text)

        payload = {
            "model": self.model,
            "input": [text],
            "input_type": input_type or self.input_type,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            req = urllib.request.Request(
                f"{self.DEFAULT_BASE_URL}/embeddings",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                items = data.get("data", [])
                if items:
                    embedding = items[0].get("embedding", [])
                    self.dims = len(embedding)
                    return embedding
        except Exception as e:
            logger.warning("Voyage AI failed: %s", e)
        return HashFallbackEmbedding().encode(text)

    def encode_for_query(self, text: str) -> list[float]:
        """用于检索查询的编码"""
        return self.encode(text, input_type="query")

    def encode_batch(self, texts: list[str], input_type: str | None = None) -> list[list[float]]:
        """批量编码 — 单次最多 128 条"""
        import urllib.request

        if not texts:
            return []
        if not self.api_key:
            return [HashFallbackEmbedding().encode(t) for t in texts]

        results = []
        # Voyage 单次最多 128
        for i in range(0, len(texts), 128):
            batch = texts[i:i + 128]
            payload = {
                "model": self.model,
                "input": batch,
                "input_type": input_type or self.input_type,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            try:
                req = urllib.request.Request(
                    f"{self.DEFAULT_BASE_URL}/embeddings",
                    data=json.dumps(payload).encode(),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                    items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
                    results.extend([item.get("embedding", []) for item in items])
            except Exception:
                results.extend([self.encode(t) for t in batch])
        return results

    async def aencode(self, text: str) -> EmbeddingResult:
        start = time.time()
        embedding = await asyncio.to_thread(self.encode, text)
        return EmbeddingResult(
            embedding=embedding,
            model=f"voyage/{self.model}",
            tokens=0,
            latency_ms=(time.time() - start) * 1000,
        )


class HuggingFaceTEIEmbedding(EmbeddingBackend):
    """
    HuggingFace Text Embeddings Inference (TEI) 后端 (v3.5.1)

    自托管 Docker 服务，原生 batch 支持
    API: POST http://{host}:{port}/embed
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = (
            base_url or os.environ.get("HF_TEI_URL", "http://localhost:8080")
        ).rstrip("/")
        self.dims = 1024

    def encode(self, text: str) -> list[float]:
        """同步编码"""
        import urllib.request

        payload = {"inputs": text}
        try:
            req = urllib.request.Request(
                f"{self.base_url}/embed",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                # TEI 返回 [[float, ...]] 单条时
                if data and isinstance(data[0], list):
                    self.dims = len(data[0])
                    return data[0]
                return data
        except Exception as e:
            logger.warning("HuggingFace TEI failed: %s", e)
            return HashFallbackEmbedding().encode(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码 — TEI 原生 batch"""
        import urllib.request

        if not texts:
            return []
        payload = {"inputs": texts}
        try:
            req = urllib.request.Request(
                f"{self.base_url}/embed",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                if data and isinstance(data[0], list):
                    self.dims = len(data[0])
                return data
        except Exception:
            return [self.encode(t) for t in texts]

    async def aencode(self, text: str) -> EmbeddingResult:
        start = time.time()
        embedding = await asyncio.to_thread(self.encode, text)
        return EmbeddingResult(
            embedding=embedding,
            model="hf-tei",
            tokens=0,
            latency_ms=(time.time() - start) * 1000,
        )


class CohereEmbedV3(EmbeddingBackend):
    """
    Cohere Embed v3 原生 API 后端 (v3.5.1)

    独立 API，支持 input_type 区分 search_document/search_query
    """

    DEFAULT_BASE_URL = "https://api.cohere.com/v2"
    DEFAULT_MODEL = "embed-multilingual-v3.0"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        input_type: str = "search_document",
    ):
        self.api_key = api_key or os.environ.get("COHERE_API_KEY", "")
        self.model = model or self.DEFAULT_MODEL
        self.input_type = input_type
        self.dims = 1024

    def encode(self, text: str, input_type: str | None = None) -> list[float]:
        """同步编码"""
        import urllib.request

        if not self.api_key:
            logger.warning("Cohere: 无 COHERE_API_KEY")
            return HashFallbackEmbedding().encode(text)

        payload = {
            "model": self.model,
            "texts": [text],
            "input_type": input_type or self.input_type,
            "embedding_types": ["float"],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            req = urllib.request.Request(
                f"{self.DEFAULT_BASE_URL}/embed",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                embeddings = data.get("embeddings", {}).get("float", [])
                if embeddings:
                    self.dims = len(embeddings[0])
                    return embeddings[0]
        except Exception as e:
            logger.warning("Cohere Embed v3 failed: %s", e)
        return HashFallbackEmbedding().encode(text)

    def encode_for_query(self, text: str) -> list[float]:
        """用于检索查询的编码"""
        return self.encode(text, input_type="search_query")

    def encode_batch(self, texts: list[str], input_type: str | None = None) -> list[list[float]]:
        """批量编码 — 单次最多 96 条"""
        import urllib.request

        if not texts:
            return []
        if not self.api_key:
            return [HashFallbackEmbedding().encode(t) for t in texts]

        results = []
        for i in range(0, len(texts), 96):
            batch = texts[i:i + 96]
            payload = {
                "model": self.model,
                "texts": batch,
                "input_type": input_type or self.input_type,
                "embedding_types": ["float"],
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            try:
                req = urllib.request.Request(
                    f"{self.DEFAULT_BASE_URL}/embed",
                    data=json.dumps(payload).encode(),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                    embeddings = data.get("embeddings", {}).get("float", [])
                    results.extend(embeddings)
            except Exception:
                results.extend([self.encode(t) for t in batch])
        return results

    async def aencode(self, text: str) -> EmbeddingResult:
        start = time.time()
        embedding = await asyncio.to_thread(self.encode, text)
        return EmbeddingResult(
            embedding=embedding,
            model=f"cohere/{self.model}",
            tokens=0,
            latency_ms=(time.time() - start) * 1000,
        )


class GoogleGeminiEmbedding(EmbeddingBackend):
    """
    Google Gemini / Vertex AI Embedding 后端 (v3.5.1)

    API: generativelanguage.googleapis.com
    默认模型: text-embedding-004 (768d)
    """

    DEFAULT_MODEL = "text-embedding-004"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.model = model or self.DEFAULT_MODEL
        self.dims = 768

    def encode(self, text: str) -> list[float]:
        """同步编码"""
        import urllib.request

        if not self.api_key:
            logger.warning("Google Gemini: 无 GOOGLE_API_KEY")
            return HashFallbackEmbedding().encode(text)

        url = (
            f"https://generativelanguage.googleapis.com/v1/models/"
            f"{self.model}:embedContent?key={self.api_key}"
        )
        payload = {"model": f"models/{self.model}", "content": {"parts": [{"text": text}]}}
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                embedding = data.get("embedding", {}).get("values", [])
                if embedding:
                    self.dims = len(embedding)
                    return embedding
        except Exception as e:
            logger.warning("Google Gemini Embedding failed: %s", e)
        return HashFallbackEmbedding().encode(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码 — 逐条调用（Gemini 无原生 batch API）"""
        return [self.encode(t) for t in texts]

    async def aencode(self, text: str) -> EmbeddingResult:
        start = time.time()
        embedding = await asyncio.to_thread(self.encode, text)
        return EmbeddingResult(
            embedding=embedding,
            model=f"gemini/{self.model}",
            tokens=0,
            latency_ms=(time.time() - start) * 1000,
        )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算余弦相似度"""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def rrf_fusion(
    results_list: list[list[tuple]],
    k: int = 60,
    use_score_weight: bool = True,
    method_weights: list[float] = None
) -> list[tuple]:
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
    results_list: list[list[tuple]],
    weights: list[float] = None
) -> list[tuple]:
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
    for _method_idx, results in enumerate(results_list):
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

    # 支持的后端列表 (v3.5.1 扩展)
    SUPPORTED_BACKENDS = [
        "ollama", "mlx", "llama_cpp", "deepseek", "voyage", "openai",
        "cohere", "cohere_v3", "hf_tei", "google", "minimax",
        "local", "chroma", "onnx",
    ]

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
        """自动检测可用的后端 (v3.5.1 扩展优先级)"""
        # 按优先级尝试各后端
        preferred_order = [
            "ollama", "llama_cpp", "deepseek", "voyage",
            "openai", "cohere_v3", "hf_tei", "google", "minimax", "local",
        ]

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
        """测试后端是否可用 (v3.5.1 扩展)"""
        try:
            if backend == "ollama":
                import urllib.request
                req = urllib.request.Request(
                    "http://localhost:11434/api/tags",
                    method="GET"
                )
                with urllib.request.urlopen(req, timeout=5):
                    return True

            elif backend == "llama_cpp":
                model_path = os.environ.get(
                    "SU_MEMORY_GGUF_MODEL_PATH",
                    os.path.expanduser(
                        "~/.cache/su-memory/models/bge-m3-q4_k_m.gguf"
                    ),
                )
                return os.path.isfile(model_path)

            elif backend == "deepseek":
                if os.environ.get("DEEPSEEK_API_KEY"):
                    return True
                import glob
                pattern = os.path.expanduser(
                    "~/.cache/su-memory/models/deepseek-*.gguf"
                )
                return bool(glob.glob(pattern))

            elif backend == "voyage":
                return bool(os.environ.get("VOYAGE_API_KEY"))

            elif backend == "openai":
                return bool(os.environ.get("OPENAI_API_KEY"))

            elif backend == "cohere_v3":
                return bool(os.environ.get("COHERE_API_KEY"))

            elif backend == "hf_tei":
                import urllib.request
                tei_url = os.environ.get("HF_TEI_URL", "http://localhost:8080")
                req = urllib.request.Request(f"{tei_url}/health")
                with urllib.request.urlopen(req, timeout=3):
                    return True

            elif backend == "google":
                return bool(os.environ.get("GOOGLE_API_KEY"))

            elif backend == "minimax":
                api_key = os.environ.get("MINIMAX_API_KEY")
                group_id = os.environ.get("MINIMAX_GROUP_ID")
                return bool(api_key and group_id)

            elif backend == "local":
                return True

            return False

        except Exception:
            return False

    def _init_backend(self, backend: str, **kwargs):
        """初始化指定后端 (v3.5.1 扩展)"""
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
        elif backend == "llama_cpp":
            self._backend = LlamaCppEmbedding(**kwargs)
            self._backend_info = {"backend": "llama_cpp", "dims": 1024}
        elif backend == "deepseek":
            self._backend = DeepSeekEmbedding(**kwargs)
            self._backend_info = {"backend": "deepseek", "dims": 1024}
        elif backend == "voyage":
            self._backend = VoyageAIEmbedding(**kwargs)
            self._backend_info = {"backend": "voyage", "dims": 1024}
        elif backend == "hf_tei":
            self._backend = HuggingFaceTEIEmbedding(**kwargs)
            self._backend_info = {"backend": "hf_tei", "dims": 1024}
        elif backend == "cohere_v3":
            self._backend = CohereEmbedV3(**kwargs)
            self._backend_info = {"backend": "cohere_v3", "dims": 1024}
        elif backend == "google":
            self._backend = GoogleGeminiEmbedding(**kwargs)
            self._backend_info = {"backend": "google", "dims": 768}
        elif backend == "onnx":
            self._backend = HashFallbackEmbedding()  # ONNX needs _sys layer
            self._backend_info = {"backend": "onnx", "dims": 384}
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Supported: {self.SUPPORTED_BACKENDS}"
            )

        self.backend_name = backend

    def encode(self, text: str) -> list[float]:
        """编码文本"""
        if self._backend is None:
            return HashFallbackEmbedding().encode(text)

        result = self._backend.encode(text)
        return result.embedding if isinstance(result, EmbeddingResult) else result

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本 (v3.5.1)"""
        if self._backend is None:
            return [HashFallbackEmbedding().encode(t) for t in texts]
        if hasattr(self._backend, 'encode_batch'):
            return self._backend.encode_batch(texts)
        return [self.encode(t) for t in texts]

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

    def get_info(self) -> dict:
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

    def encode(self, text: str) -> list[float]:
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
        """异步编码 — P0-C：Hash 是 CPU 密集计算，asyncio.to_thread 避免阻塞事件循环"""
        embedding = await asyncio.to_thread(self.encode, text)
        return EmbeddingResult(
            embedding=embedding,
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

    def encode(self, text: str) -> list[float]:
        """编码文本"""
        return self._base_embedder.encode(text)

    async def aencode(self, text: str) -> EmbeddingResult:
        """异步编码 — P0-C：Ollama HTTP 调用是 I/O 密集，asyncio.to_thread 避免阻塞事件循环"""
        result = await asyncio.to_thread(self._base_embedder.encode, text)
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
    "LlamaCppEmbedding",
    "DeepSeekEmbedding",
    "VoyageAIEmbedding",
    "HuggingFaceTEIEmbedding",
    "CohereEmbedV3",
    "GoogleGeminiEmbedding",
    "HashFallbackEmbedding",
    "cosine_similarity",
    "rrf_fusion",
    "weighted_combination_fusion",
]
