"""
su-memory SDK 向量嵌入服务抽象层

支持多种嵌入服务：
- OpenAI (text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large)
- MiniMax (embo-01)
- Ollama (本地模型)
- ChromaDB (内置向量数据库)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os


@dataclass
class EmbeddingResult:
    """嵌入结果"""
    embedding: List[float]
    model: str
    dimensions: int
    tokens_used: Optional[int] = None


class EmbeddingProvider(ABC):
    """嵌入服务抽象基类"""

    @abstractmethod
    def embed(self, texts: List[str], model: Optional[str] = None) -> List[EmbeddingResult]:
        """
        批量生成嵌入向量

        Args:
            texts: 文本列表
            model: 模型名称（可选）

        Returns:
            嵌入结果列表
        """
        pass

    @abstractmethod
    def embed_single(self, text: str, model: Optional[str] = None) -> EmbeddingResult:
        """
        单条文本嵌入

        Args:
            text: 文本
            model: 模型名称（可选）

        Returns:
            嵌入结果
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查服务是否可用

        Returns:
            是否可用
        """
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """获取默认模型名称"""
        pass

    def get_config_help(self) -> str:
        """获取配置帮助信息"""
        return ""


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI 嵌入服务"""

    DEFAULT_MODELS = {
        "small": "text-embedding-3-small",    # 1536维, $0.02/1M tokens
        "large": "text-embedding-3-large",    # 3072维, $0.13/1M tokens
        "legacy": "text-embedding-ada-002"     # 1536维, 兼容旧版
    }

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化 OpenAI 嵌入服务

        Args:
            api_key: API密钥，默认从环境变量 OPENAI_API_KEY 获取
            base_url: API地址，默认使用官方地址
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
        self._client = None

    def _get_client(self):
        """获取 OpenAI 客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError(
                    "请安装 OpenAI 包: pip install openai\n"
                    "或者设置环境变量使用代理服务"
                )
        return self._client

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[EmbeddingResult]:
        """批量生成嵌入"""
        if not texts:
            return []

        model = model or self.get_default_model()
        client = self._get_client()

        response = client.embeddings.create(
            model=model,
            input=texts
        )

        results = []
        for i, data in enumerate(response.data):
            results.append(EmbeddingResult(
                embedding=data.embedding,
                model=model,
                dimensions=len(data.embedding),
                tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else None
            ))

        return results

    def embed_single(self, text: str, model: Optional[str] = None) -> EmbeddingResult:
        """单条文本嵌入"""
        return self.embed([text], model)[0]

    def is_available(self) -> bool:
        """检查服务是否可用"""
        if not self.api_key:
            return False
        try:
            self._get_client().embeddings.create(
                model=self.get_default_model(),
                input=["test"]
            )
            return True
        except Exception:
            return False

    def get_default_model(self) -> str:
        """获取默认模型"""
        return self.DEFAULT_MODELS["small"]

    def get_config_help(self) -> str:
        return """OpenAI 嵌入服务配置:
1. 设置 API Key: export OPENAI_API_KEY=sk-xxx
2. (可选) 设置代理: export OPENAI_API_BASE_URL=https://api.openai.com/v1"""


class MiniMaxEmbedder(EmbeddingProvider):
    """MiniMax 嵌入服务"""

    DEFAULT_MODEL = "embo-01"
    DEFAULT_DIMENSIONS = 1536

    def __init__(self, api_key: Optional[str] = None, group_id: Optional[str] = None):
        """
        初始化 MiniMax 嵌入服务

        Args:
            api_key: API密钥，默认从环境变量 MINIMAX_API_KEY 获取
            group_id: 组ID，默认从环境变量 MINIMAX_GROUP_ID 获取
        """
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        self.group_id = group_id or os.environ.get("MINIMAX_GROUP_ID")
        self.base_url = "https://api.minimax.chat/v1"
        self._client = None

    def _get_client(self):
        """获取 MiniMax 客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError("请安装 OpenAI 包: pip install openai")
        return self._client

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[EmbeddingResult]:
        """批量生成嵌入"""
        if not texts:
            return []

        model = model or self.get_default_model()
        client = self._get_client()

        response = client.embeddings.create(
            model=model,
            input=texts,
            extra_body={"dimension": self.DEFAULT_DIMENSIONS}
        )

        results = []
        for data in response.data:
            results.append(EmbeddingResult(
                embedding=data.embedding,
                model=model,
                dimensions=len(data.embedding)
            ))

        return results

    def embed_single(self, text: str, model: Optional[str] = None) -> EmbeddingResult:
        """单条文本嵌入"""
        return self.embed([text], model)[0]

    def is_available(self) -> bool:
        """检查服务是否可用"""
        if not self.api_key or not self.group_id:
            return False
        try:
            self._get_client().embeddings.create(
                model=self.get_default_model(),
                input=["test"]
            )
            return True
        except Exception:
            return False

    def get_default_model(self) -> str:
        """获取默认模型"""
        return self.DEFAULT_MODEL

    def get_config_help(self) -> str:
        return """MiniMax 嵌入服务配置:
1. 设置 API Key: export MINIMAX_API_KEY=xxx
2. 设置 Group ID: export MINIMAX_GROUP_ID=xxx"""


class OllamaEmbedder(EmbeddingProvider):
    """Ollama 本地嵌入服务"""

    DEFAULT_MODEL = "nomic-embed-text"

    def __init__(self, base_url: Optional[str] = None):
        """
        初始化 Ollama 嵌入服务

        Args:
            base_url: Ollama 服务地址，默认从环境变量 OLLAMA_BASE_URL 获取
        """
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self._client = None

    def _get_client(self):
        """获取 Ollama 客户端"""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(base_url=self.base_url, timeout=60.0)
            except ImportError:
                raise ImportError("请安装 httpx: pip install httpx")
        return self._client

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[EmbeddingResult]:
        """批量生成嵌入"""
        if not texts:
            return []

        model = model or self.get_default_model()
        client = self._get_client()

        results = []
        for text in texts:
            response = client.post("/api/embeddings", json={
                "model": model,
                "prompt": text
            })
            response.raise_for_status()
            data = response.json()

            results.append(EmbeddingResult(
                embedding=data["embedding"],
                model=model,
                dimensions=len(data["embedding"])
            ))

        return results

    def embed_single(self, text: str, model: Optional[str] = None) -> EmbeddingResult:
        """单条文本嵌入"""
        return self.embed([text], model)[0]

    def is_available(self) -> bool:
        """检查服务是否可用"""
        try:
            client = self._get_client()
            response = client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def get_default_model(self) -> str:
        """获取默认模型"""
        return self.DEFAULT_MODEL

    def get_config_help(self) -> str:
        return """Ollama 嵌入服务配置:
1. 启动 Ollama: ollama serve
2. 拉取模型: ollama pull nomic-embed-text
3. (可选) 设置地址: export OLLAMA_BASE_URL=http://localhost:11434"""


class ChromaEmbedder(EmbeddingProvider):
    """ChromaDB 向量数据库嵌入服务"""

    def __init__(self, collection_name: str = "su_memory", persist_directory: Optional[str] = None):
        """
        初始化 ChromaDB 嵌入服务

        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
        self._base_embedder = None

    def _get_base_embedder(self) -> OllamaEmbedder:
        """获取基础嵌入器"""
        if self._base_embedder is None:
            self._base_embedder = OllamaEmbedder()
        return self._base_embedder

    def _get_client(self):
        """获取 ChromaDB 客户端"""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
            except ImportError:
                raise ImportError(
                    "请安装 ChromaDB: pip install chromadb\n"
                    "注意: ChromaDB 需要 Ollama 作为后端"
                )
        return self._client

    def _get_collection(self):
        """获取集合"""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "su-memory SDK embeddings"}
            )
        return self._collection

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[EmbeddingResult]:
        """批量生成嵌入（存储到 ChromaDB）"""
        if not texts:
            return []

        collection = self._get_collection()
        base_embedder = self._get_base_embedder()

        # 生成嵌入
        embeddings = base_embedder.embed(texts, model)

        # 添加到 ChromaDB
        ids = [f"embed_{i}_{hash(text) % 1000000}" for i, text in enumerate(texts)]
        collection.add(
            ids=ids,
            embeddings=[e.embedding for e in embeddings],
            documents=texts,
            metadatas=[{"model": e.model, "dimensions": e.dimensions} for e in embeddings]
        )

        return embeddings

    def embed_single(self, text: str, model: Optional[str] = None) -> EmbeddingResult:
        """单条文本嵌入"""
        return self.embed([text], model)[0]

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """查询相似文本"""
        collection = self._get_collection()
        base_embedder = self._get_base_embedder()

        query_embedding = base_embedder.embed_single(query_text)

        results = collection.query(
            query_embeddings=[query_embedding.embedding],
            n_results=top_k
        )

        return [{
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i] if "distances" in results else None,
            "metadata": results["metadatas"][0][i] if "metadatas" in results else None
        } for i in range(len(results["ids"][0]))]

    def is_available(self) -> bool:
        """检查服务是否可用"""
        try:
            base_embedder = self._get_base_embedder()
            return base_embedder.is_available()
        except Exception:
            return False

    def get_default_model(self) -> str:
        """获取默认模型"""
        return OllamaEmbedder.DEFAULT_MODEL

    def get_config_help(self) -> str:
        return """ChromaDB 嵌入服务配置:
1. 确保 Ollama 已启动: ollama serve
2. 拉取嵌入模型: ollama pull nomic-embed-text
3. 安装 ChromaDB: pip install chromadb
4. (可选) 设置持久化目录"""


class EmbeddingFactory:
    """嵌入服务工厂"""

    _providers: Dict[str, type] = {
        "openai": OpenAIEmbedder,
        "minimax": MiniMaxEmbedder,
        "ollama": OllamaEmbedder,
        "chroma": ChromaEmbedder,
    }

    @classmethod
    def create(cls, provider: str = "auto", **kwargs) -> EmbeddingProvider:
        """
        创建嵌入服务

        Args:
            provider: 提供商名称，"auto" 表示自动检测
            **kwargs: 提供商配置参数

        Returns:
            嵌入服务实例
        """
        if provider == "auto":
            return cls.auto_detect(**kwargs)

        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"未知的提供商: {provider}，可用: {available}")

        return cls._providers[provider](**kwargs)

    @classmethod
    def auto_detect(cls, preferred: List[str] = None) -> EmbeddingProvider:
        """
        自动检测可用嵌入服务

        按优先级尝试各服务，返回第一个可用的

        Args:
            preferred: 优先尝试的服务列表

        Returns:
            可用的嵌入服务实例
        """
        default_order = ["ollama", "openai", "minimax", "chroma"]
        order = preferred or default_order

        errors = []
        for name in order:
            try:
                provider_class = cls._providers.get(name)
                if not provider_class:
                    continue

                provider = provider_class()
                if provider.is_available():
                    print(f"  ✅ 自动选择嵌入服务: {name}")
                    return provider
            except Exception as e:
                errors.append(f"{name}: {str(e)}")

        # 全部不可用
        error_msg = "\n".join(errors)
        raise RuntimeError(
            f"没有可用的嵌入服务，请配置以下服务之一:\n"
            f"1. Ollama (推荐): pip install httpx && ollama serve && ollama pull nomic-embed-text\n"
            f"2. OpenAI: pip install openai && export OPENAI_API_KEY=sk-xxx\n"
            f"3. MiniMax: export MINIMAX_API_KEY=xxx && export MINIMAX_GROUP_ID=xxx\n"
            f"\n详细错误:\n{error_msg}"
        )

    @classmethod
    def list_providers(cls) -> List[str]:
        """列出所有支持的提供商"""
        return list(cls._providers.keys())


# 便捷函数
def get_embedder(provider: str = "auto", **kwargs) -> EmbeddingProvider:
    """获取嵌入服务"""
    return EmbeddingFactory.create(provider, **kwargs)
