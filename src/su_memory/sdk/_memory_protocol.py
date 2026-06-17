"""
MemoryProtocol — 统一记忆客户端接口

所有 SDK 客户端 (SuMemoryClient / SuMemoryLite / SuMemoryLitePro)
均实现此协议，确保公共 API 契约一致。

v3.5.5: 提取公共接口，统一三个 SDK 类的对外契约。P0-1 修复：forget()/get_all_memories() 添加 @abstractmethod 强制约束。
"""

from abc import ABC, abstractmethod
from typing import Any


class MemoryProtocol(ABC):
    """
    记忆客户端统一接口协议。

    定义了所有记忆客户端必须实现的公共方法。
    每个具体实现（Client/Lite/LitePro）提供不同的后端策略。

    Required methods (all implementations):
        - add(): 添加单条记忆
        - add_batch(): 批量添加记忆
        - query(): 语义搜索记忆
        - count(): 记忆总数
        - forget(): 删除单条记忆 (v3.5.5)
        - get_all_memories(): 获取全部记忆列表 (v3.5.5)
        - integration_health(): 集成健康检查

    Optional methods:
        - health_check(): 深度健康检查（Client/LitePro 实现，Lite 不实现）
    """

    @abstractmethod
    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        **kwargs
    ) -> str:
        """
        添加单条记忆。

        Args:
            content: 记忆内容文本
            metadata: 可选元数据
            **kwargs: 实现特定参数

        Returns:
            memory_id: 记忆唯一标识符
        """
        ...

    def add_batch(
        self,
        items: list[dict[str, Any]],
        **kwargs
    ) -> list[str]:
        """
        批量添加记忆。
        默认实现：逐条调用 add()。子类可重写为更高效的批量实现。

        Args:
            items: 记忆列表，每项含 content 和可选 metadata
            **kwargs: 实现特定参数

        Returns:
            memory_ids: 记忆ID列表
        """
        return [self.add(item.get("content", ""), item.get("metadata"), **kwargs) for item in items]

    @abstractmethod
    def forget(
        self,
        memory_id: str,
        **kwargs
    ) -> bool:
        """
        删除单条记忆 (v3.5.5)。

        Args:
            memory_id: 要删除的记忆ID
            **kwargs: 实现特定参数

        Returns:
            是否删除成功
        """
        ...

    @abstractmethod
    def get_all_memories(
        self,
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        获取全部记忆列表 (v3.5.5)。

        Args:
            **kwargs: 实现特定参数

        Returns:
            记忆列表，每项含 id, content, metadata 等
        """
        ...

    @abstractmethod
    def query(
        self,
        text: str,
        top_k: int = 10,
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        语义搜索记忆。

        Args:
            text: 查询文本
            top_k: 返回结果数量
            **kwargs: 实现特定参数

        Returns:
            搜索结果列表，每项含 content, score, memory_id 等
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """
        获取记忆总数。

        Returns:
            当前存储的记忆数量
        """
        ...

    def integration_health(self) -> dict[str, Any]:
        """
        集成健康检查 — 默认实现，子类可重写。

        Returns:
            健康状况字典，至少包含 status 和 detail
        """
        return {
            "status": "healthy",
            "detail": f"{self.__class__.__name__} running normally",
            "count": self.count(),
        }

    def health_check(self) -> dict[str, Any]:
        """
        深度健康检查 — 可选方法。

        默认返回简单状态。Lite 实现可以仅依赖此默认实现，
        Client/LitePro 应重写为完整检查。

        Returns:
            健康状况字典
        """
        return {
            "status": "ok",
            "detail": "health_check not implemented in this client",
            "count": self.count(),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(count={self.count()})"
