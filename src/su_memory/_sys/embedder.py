"""
Embedding 后端抽象层

统一接口，支持 ollama / oMLX / OpenAI / Cohere / Jina / ONNX 本地模型
"""
import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod


class Embedder(ABC):
    """Embedding 引擎抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        pass

    @abstractmethod
    def embed_one(self, text: str, **kwargs) -> list[float]:
        pass

    @property
    def dims(self) -> int:
        return 384


class OllamaEmbedder(Embedder):
    """Ollama 本地 embedding 引擎"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "bge-m3",
        dims: int = 1024,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dims = dims
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    @property
    def dims(self) -> int:
        return self._dims

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        results = []
        for text in texts:
            results.append(self.embed_one(text))
        return results

    def embed_one(self, text: str, **kwargs) -> list[float]:
        body = json.dumps({"model": self.model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
            return data.get("embedding", [])
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama embed_one failed: {e}") from e

    def ping(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status == 200
        except Exception:
            return False


class OpenAICompatEmbedder(Embedder):
    """
    OpenAI v1/embeddings 兼容接口（适用于 oMLX / Cohere / Jina 等）
    """

    def __init__(
        self,
        base_url: str,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dims: int = 1536,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/v1").rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._dims = dims
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"{self.base_url}/{self.model}"

    @property
    def dims(self) -> int:
        return self._dims

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        single = len(texts) == 1
        body = {"model": self.model, "input": texts[0] if single else texts}
        req = urllib.request.Request(
            f"{self.base_url}/v1/embeddings",
            data=json.dumps(body).encode(),
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
            items = data.get("data", [])
            if single and len(items) == 1:
                return [items[0].get("embedding", [])]
            return [item.get("embedding", []) for item in sorted(items, key=lambda x: x.get("index", 0))]
        except urllib.error.URLError as e:
            raise RuntimeError(f"OpenAICompat embed failed: {e}") from e

    def embed_one(self, text: str, **kwargs) -> list[float]:
        return self.embed([text])[0]

    def ping(self) -> bool:
        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/models",
                headers=self._headers(),
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status == 200
        except Exception:
            return False


class ONNXEmbedder(Embedder):
    """本地 ONNX 模型后端（无网络依赖）— v3.5.1 补全实现"""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str | None = None,
        dims: int = 384,
        max_length: int = 512,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or os.path.join(
            os.path.dirname(model_path), "tokenizer.json"
        )
        self._dims = dims
        self._max_length = max_length
        self._session = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime 未安装。请执行: pip install onnxruntime"
            ) from None
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError(
                "tokenizers 未安装。请执行: pip install tokenizers"
            ) from None
        self._session = ort.InferenceSession(self.model_path)
        if os.path.exists(self.tokenizer_path):
            self._tokenizer = Tokenizer.from_file(self.tokenizer_path)
            self._tokenizer.enable_truncation(max_length=self._max_length)
            self._tokenizer.enable_padding(length=self._max_length)
        else:
            raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}")

    @property
    def name(self) -> str:
        return f"onnx:{os.path.basename(self.model_path)}"

    @property
    def dims(self) -> int:
        return self._dims

    def _mean_pool_and_normalize(self, token_embeddings, attention_mask):
        """Mean pooling + L2 normalize"""
        import numpy as np
        mask_expanded = attention_mask[:, :, None].astype(float)
        summed = (token_embeddings * mask_expanded).sum(axis=1)
        counts = mask_expanded.sum(axis=1).clip(min=1e-9)
        pooled = summed / counts
        norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return (pooled / norms).tolist()

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        import numpy as np
        self._ensure_loaded()
        encoded = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        # 部分模型需要 token_type_ids
        input_names = [inp.name for inp in self._session.get_inputs()]
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        outputs = self._session.run(None, feeds)
        token_embs = outputs[0]  # (batch, seq_len, hidden)
        result = self._mean_pool_and_normalize(token_embs, attention_mask)
        if result:
            self._dims = len(result[0])
        return result

    def embed_one(self, text: str, **kwargs) -> list[float]:
        return self.embed([text])[0]


class LlamaCppEmbedder(Embedder):
    """llama.cpp GGUF 本地推理引擎 (v3.5.1)"""

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
        self._dims_val: int | None = None

    @property
    def name(self) -> str:
        return f"llama.cpp/{os.path.basename(self.model_path)}"

    @property
    def dims(self) -> int:
        return self._dims_val or 1024

    def _get_model(self):
        if self._model is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python 未安装。请执行: pip install llama-cpp-python"
                ) from None
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(f"GGUF 模型未找到: {self.model_path}")
            self._model = Llama(
                model_path=self.model_path,
                embedding=True,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                verbose=self.verbose,
            )
        return self._model

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        model = self._get_model()
        results = model.embed(texts)
        if results and not isinstance(results[0], list):
            results = [results]
        if results and self._dims_val is None:
            self._dims_val = len(results[0])
        return results

    def embed_one(self, text: str, **kwargs) -> list[float]:
        return self.embed([text])[0]

    def ping(self) -> bool:
        try:
            self._get_model()
            return True
        except Exception:
            return False


class DeepSeekEmbedder(Embedder):
    """DeepSeek 云端 + 本地双模引擎 (v3.5.1)"""

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
        self._local: LlamaCppEmbedder | None = None
        self._mode: str | None = None
        self._dims_val = 1024

    @property
    def name(self) -> str:
        return f"deepseek/{self.model}"

    @property
    def dims(self) -> int:
        return self._dims_val

    def _resolve_mode(self) -> str:
        if self._mode:
            return self._mode
        if self.api_key:
            self._mode = "cloud"
            return "cloud"
        import glob
        pattern = os.path.join(LlamaCppEmbedder.DEFAULT_MODEL_DIR, "deepseek-*.gguf")
        matches = glob.glob(pattern)
        if matches:
            self._local = LlamaCppEmbedder(model_path=matches[0])
            self._mode = "local"
            return "local"
        self._mode = "cloud"
        return "cloud"

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        mode = self._resolve_mode()
        if mode == "local" and self._local:
            return self._local.embed(texts)
        if not self.api_key:
            raise RuntimeError("DeepSeek: 无 API Key 且未找到本地 GGUF")
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            embeddings = [item.get("embedding", []) for item in items]
            if embeddings:
                self._dims_val = len(embeddings[0])
            return embeddings
        except urllib.error.URLError as e:
            if self._local:
                return self._local.embed(texts)
            raise RuntimeError(f"DeepSeek embed failed: {e}") from e

    def embed_one(self, text: str, **kwargs) -> list[float]:
        return self.embed([text])[0]

    def ping(self) -> bool:
        if self._resolve_mode() == "local":
            return self._local.ping() if self._local else False
        return bool(self.api_key)


class VoyageAIEmbedder(Embedder):
    """Voyage AI 云端 Embedding (v3.5.1)"""

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
        self.input_type = input_type
        self._dims_val = 1024

    @property
    def name(self) -> str:
        return f"voyage/{self.model}"

    @property
    def dims(self) -> int:
        return self._dims_val

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("Voyage AI: 无 VOYAGE_API_KEY")
        input_type = kwargs.get("input_type", self.input_type)
        body = json.dumps({
            "model": self.model,
            "input": texts,
            "input_type": input_type,
        }).encode()
        req = urllib.request.Request(
            f"{self.DEFAULT_BASE_URL}/embeddings",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        embeddings = [item.get("embedding", []) for item in items]
        if embeddings:
            self._dims_val = len(embeddings[0])
        return embeddings

    def embed_one(self, text: str, **kwargs) -> list[float]:
        return self.embed([text], **kwargs)[0]

    def ping(self) -> bool:
        return bool(self.api_key)


class HuggingFaceTEIEmbedder(Embedder):
    """HuggingFace TEI 自托管服务 (v3.5.1)"""

    def __init__(self, base_url: str | None = None):
        self.base_url = (
            base_url or os.environ.get("HF_TEI_URL", "http://localhost:8080")
        ).rstrip("/")
        self._dims_val = 1024

    @property
    def name(self) -> str:
        return f"hf-tei:{self.base_url}"

    @property
    def dims(self) -> int:
        return self._dims_val

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        body = json.dumps({"inputs": texts}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/embed",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        if data and isinstance(data[0], list):
            self._dims_val = len(data[0])
        return data

    def embed_one(self, text: str, **kwargs) -> list[float]:
        return self.embed([text])[0]

    def ping(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status == 200
        except Exception:
            return False


class CohereV3Embedder(Embedder):
    """Cohere Embed v3 原生 API (v3.5.1)"""

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
        self._dims_val = 1024

    @property
    def name(self) -> str:
        return f"cohere/{self.model}"

    @property
    def dims(self) -> int:
        return self._dims_val

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("Cohere: 无 COHERE_API_KEY")
        input_type = kwargs.get("input_type", self.input_type)
        body = json.dumps({
            "model": self.model,
            "texts": texts,
            "input_type": input_type,
            "embedding_types": ["float"],
        }).encode()
        req = urllib.request.Request(
            f"{self.DEFAULT_BASE_URL}/embed",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        embeddings = data.get("embeddings", {}).get("float", [])
        if embeddings:
            self._dims_val = len(embeddings[0])
        return embeddings

    def embed_one(self, text: str, **kwargs) -> list[float]:
        return self.embed([text], **kwargs)[0]

    def ping(self) -> bool:
        return bool(self.api_key)


class GoogleGeminiEmbedder(Embedder):
    """Google Gemini / Vertex AI Embedding (v3.5.1)"""

    DEFAULT_MODEL = "text-embedding-004"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.model = model or self.DEFAULT_MODEL
        self._dims_val = 768

    @property
    def name(self) -> str:
        return f"gemini/{self.model}"

    @property
    def dims(self) -> int:
        return self._dims_val

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("Google Gemini: 无 GOOGLE_API_KEY")
        results = []
        for text in texts:
            url = (
                f"https://generativelanguage.googleapis.com/v1/models/"
                f"{self.model}:embedContent?key={self.api_key}"
            )
            body = json.dumps({
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text}]},
            }).encode()
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            embedding = data.get("embedding", {}).get("values", [])
            if embedding and self._dims_val != len(embedding):
                self._dims_val = len(embedding)
            results.append(embedding)
        return results

    def embed_one(self, text: str, **kwargs) -> list[float]:
        return self.embed([text])[0]

    def ping(self) -> bool:
        return bool(self.api_key)


# ── 工厂 ──────────────────────────────────────────────────

def create_embedder(
    backend: str = "ollama",
    **kwargs,
) -> Embedder:
    """
    根据 backend 名称创建 Embedder 实例 (v3.5.1 扩展)

    backend 名称规范：
        "ollama"            → OllamaEmbedder (localhost:11434)
        "ollama://host:port" → OllamaEmbedder (自定义地址)
        "llama_cpp"         → LlamaCppEmbedder (本地 GGUF)
        "deepseek"          → DeepSeekEmbedder (云端+本地)
        "voyage"            → VoyageAIEmbedder
        "hf_tei"            → HuggingFaceTEIEmbedder
        "cohere_v3"         → CohereV3Embedder (原生 API)
        "google"            → GoogleGeminiEmbedder
        "openai"            → OpenAICompatEmbedder (api.openai.com)
        "cohere"            → OpenAICompatEmbedder (api.cohere.ai)
        "jina"              → OpenAICompatEmbedder (api.jina.ai)
        "http://..."        → OpenAICompatEmbedder (自定义地址)
        "onnx"              → ONNXEmbedder (本地文件)
    """
    b = backend.lower().strip()

    if b == "ollama" or b.startswith("ollama://"):
        url = kwargs.pop("base_url", "http://localhost:11434")
        if b.startswith("ollama://"):
            url = b.replace("ollama://", "http://")
        return OllamaEmbedder(base_url=url, **kwargs)

    elif b == "llama_cpp":
        return LlamaCppEmbedder(**kwargs)

    elif b == "deepseek":
        return DeepSeekEmbedder(**kwargs)

    elif b == "voyage":
        return VoyageAIEmbedder(**kwargs)

    elif b == "hf_tei":
        return HuggingFaceTEIEmbedder(**kwargs)

    elif b == "cohere_v3":
        return CohereV3Embedder(**kwargs)

    elif b == "google":
        return GoogleGeminiEmbedder(**kwargs)

    elif b in ("openai", "cohere", "jina"):
        bases = {
            "openai": "https://api.openai.com/v1",
            "cohere": "https://api.cohere.ai/v1",
            "jina": "https://api.jina.ai/v1",
        }
        return OpenAICompatEmbedder(base_url=bases[b], **kwargs)

    elif b.startswith("http://") or b.startswith("https://"):
        return OpenAICompatEmbedder(base_url=b, **kwargs)

    elif b == "onnx":
        return ONNXEmbedder(**kwargs)

    else:
        raise ValueError(f"Unknown embedder backend: {backend}")
