"""
su-memory SDK 客户端
对外赋能的统一入口
"""
from typing import Optional, List, Dict, Any

from su_memory.sdk.config import SDKConfig
from su_memory.sdk.exceptions import SDKError


class SuMemoryClient:
    """
    su-memory SDK 核心客户端

    一行代码让AI应用拥有记忆能力。

    Example:
        >>> from su_memory.sdk import SuMemoryClient
        >>> client = SuMemoryClient()
        >>> mid = client.add("今天学习了Python")
        >>> results = client.query("学习内容")
        >>> print(results[0]["content"])
    """

    def __init__(
        self,
        mode: str = "local",
        config: Optional[SDKConfig] = None,
        **kwargs
    ):
        """
        初始化SDK客户端

        Args:
            mode: 运行模式 (local/cloud/edge/embedded)
            config: SDK配置对象
            **kwargs: 其他配置参数
        """
        self.config = config or SDKConfig(mode=mode, **kwargs)
        self.mode = mode

        # 根据模式初始化不同的后端
        if self.mode == "cloud":
            self._init_cloud_client()
        else:
            self._init_local_client()

    def _init_local_client(self):
        """初始化本地客户端"""
        from su_memory.client import SuMemory as LocalSuMemory

        self._client = LocalSuMemory(
            mode="local",
            persist_dir=self.config.persist_dir
        )

    def _init_cloud_client(self):
        """初始化云端客户端"""
        import httpx

        self._http_client = httpx.Client(
            base_url=self.config.api_url,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=self.config.timeout
        )

    def add(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        encoding: str = "auto"
    ) -> str:
        """
        添加记忆

        Args:
            content: 记忆内容
            metadata: 元数据
            encoding: 编码方式 (auto/semantic/bagua)

        Returns:
            memory_id: 记忆唯一标识
        """
        if self.mode == "cloud":
            return self._add_cloud(content, metadata)
        else:
            return self._client.add(content, metadata)

    def _add_cloud(self, content: str, metadata: dict) -> str:
        """云端添加记忆"""
        response = self._http_client.post(
            "/api/v1/memory/add",
            json={"content": content, "metadata": metadata}
        )
        response.raise_for_status()
        return response.json()["memory_id"]

    def query(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "semantic"
    ) -> List[Dict[str, Any]]:
        """
        查询记忆

        Args:
            query: 查询内容
            top_k: 返回数量
            mode: 查询模式 (semantic/causal/hybrid)

        Returns:
            results: 检索结果列表
        """
        if self.mode == "cloud":
            return self._query_cloud(query, top_k, mode)
        else:
            return self._query_local(query, top_k, mode)

    def _query_local(self, query: str, top_k: int, mode: str) -> List[Dict[str, Any]]:
        """本地查询"""
        results = self._client.query(query, top_k=top_k)
        return [
            {
                "memory_id": r.memory_id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata
            }
            for r in results
        ]

    def _query_cloud(self, query: str, top_k: int, mode: str) -> List[Dict[str, Any]]:
        """云端查询"""
        response = self._http_client.post(
            "/api/v1/memory/query",
            json={"query": query, "top_k": top_k, "mode": mode}
        )
        response.raise_for_status()
        return response.json()["results"]

    def predict(
        self,
        situation: str,
        action: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        预测行动结果

        Args:
            situation: 当前情境
            action: 拟采取行动
            options: 预测选项

        Returns:
            prediction: 预测结果
        """
        if self.mode == "cloud":
            return self._predict_cloud(situation, action, options)
        else:
            return self._predict_local(situation, action)

    def _predict_local(self, situation: str, action: str) -> Dict[str, Any]:
        """本地预测"""
        return {
            "outcome": "预测功能需使用云端模式",
            "confidence": 0.0,
            "mode": "local"
        }

    def _predict_cloud(self, situation: str, action: str, options: dict) -> Dict[str, Any]:
        """云端预测"""
        response = self._http_client.post(
            "/api/v1/predict",
            json={"situation": situation, "action": action}
        )
        response.raise_for_status()
        return response.json()["prediction"]

    def link(
        self,
        cause_id: str,
        effect_id: str,
        relation: str = "causes",
        strength: float = 1.0
    ):
        """建立因果链接"""
        return self._client.link(cause_id, effect_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        return self._client.get_stats()

    def __len__(self) -> int:
        """记忆数量"""
        return len(self._client)

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        pass
