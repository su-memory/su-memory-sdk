"""
Embedding 后端抽象层

统一接口，支持 ollama / oMLX / OpenAI / Cohere / Jina / ONNX 本地模型
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import os, json, urllib.request, urllib.error


class Embedder(ABC):
    """Embedding 引擎抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_one(self, text: str, **kwargs) -> List[float]:
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

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        results = []
        for text in texts:
            results.append(self.embed_one(text))
        return results

    def embed_one(self, text: str, **kwargs) -> List[float]:
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
            raise RuntimeError(f"Ollama embed_one failed: {e}")

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
        api_key: Optional[str] = None,
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

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
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
            raise RuntimeError(f"OpenAICompat embed failed: {e}")

    def embed_one(self, text: str, **kwargs) -> List[float]:
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
    """本地 ONNX 模型后端（无网络依赖）"""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        dims: int = 384,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._dims = dims

    @property
    def name(self) -> str:
        return f"onnx:{os.path.basename(self.model_path)}"

    @property
    def dims(self) -> int:
        return self._dims

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        raise NotImplementedError(
            "ONNX embed: use embed_one() or switch to Ollama/OpenAI backend"
        )

    def embed_one(self, text: str, **kwargs) -> List[float]:
        raise NotImplementedError(
            "ONNX backend requires tokenizer — use OllamaEmbedder or OpenAICompatEmbedder"
        )


# ── 工厂 ──────────────────────────────────────────────────

def create_embedder(
    backend: str = "ollama",
    **kwargs,
) -> Embedder:
    """
    根据 backend 名称创建 Embedder 实例

    backend 名称规范：
        "ollama"          → OllamaEmbedder (localhost:11434)
        "ollama://host:port" → OllamaEmbedder (自定义地址)
        "openai"           → OpenAICompatEmbedder (api.openai.com)
        "cohere"           → OpenAICompatEmbedder (api.cohere.ai)
        "jina"             → OpenAICompatEmbedder (api.jina.ai)
        "http://..."       → OpenAICompatEmbedder (自定义地址)
        "onnx"             → ONNXEmbedder (本地文件)
    """
    b = backend.lower().strip()

    if b == "ollama" or b.startswith("ollama://"):
        url = kwargs.pop("base_url", "http://localhost:11434")
        if b.startswith("ollama://"):
            url = b.replace("ollama://", "http://")
        return OllamaEmbedder(base_url=url, **kwargs)

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
