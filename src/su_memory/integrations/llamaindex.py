"""
su-memory SDK × LlamaIndex 集成

提供与 LlamaIndex 的深度集成，支持：
- BaseRetriever 接口实现
- BaseQueryEngine 支持
- StorageContext 集成
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# LlamaIndex 相关导入（可选）
LLAMAINDEX_AVAILABLE = False
BaseRetriever = object
BaseQueryEngine = object
LLMDocument = None
TextNode = None
NodeRelationship = None
RelatedNodeInfo = None
CallbackManager = None

try:
    from llama_index.core import Document as LLMDocument
    from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.base.response import ResponseMode
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    pass


@dataclass
class SuMemoryIndexConfig:
    """索引配置"""
    index_name: str = "su_memory_index"
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_top_k: int = 5


class SuMemoryLlamaIndexRetriever(BaseRetriever if LLAMAINDEX_AVAILABLE else object):
    """
    LlamaIndex Retriever 实现

    将 su-memory SDK 作为 LlamaIndex 的检索器使用

    Example:
        >>> from llama_index.core import VectorStoreIndex
        >>> from llama_index.llms.openai import OpenAI
        >>> from su_memory.integrations.llamaindex import SuMemoryLlamaIndexRetriever
        >>>
        >>> # 创建检索器
        >>> retriever = SuMemoryLlamaIndexRetriever(memory_client, top_k=5)
        >>>
        >>> # 创建索引（需要先索引数据）
        >>> index = VectorStoreIndex.from_vector_store(retriever)
        >>>
        >>> # 查询
        >>> query_engine = index.as_query_engine()
        >>> response = query_engine.query("今天发生了什么?")
    """

    def __init__(
        self,
        memory_client,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        callback_manager: Optional[CallbackManager] = None
    ):
        """
        初始化检索器

        Args:
            memory_client: SuMemoryLite 或 SuMemoryLitePro 实例
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            callback_manager: 回调管理器
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "请安装 LlamaIndex: pip install llama-index-core llama-index-llms-openai\n"
                "或查看文档: https://docs.llamaindex.ai/en/stable/getting_started/installation.html"
            )

        super().__init__(callback_manager=callback_manager)

        self._client = memory_client
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    def _retrieve(self, query_bundle) -> List["TextNode"]:
        """
        检索相关节点

        Args:
            query_bundle: QueryBundle 对象

        Returns:
            匹配的 TextNode 列表
        """
        query = query_bundle.query_str

        # 查询 su-memory
        results = self._client.query(query, top_k=self._top_k)

        nodes = []
        for r in results:
            score = r.get("score", 0)

            # 过滤低于阈值的结果
            if score < self._similarity_threshold:
                continue

            # 创建 LlamaIndex Node
            node = TextNode(
                text=r.get("content", ""),
                id_=r.get("memory_id", r.get("id", "")),
                score=score,
                metadata={
                    **{k: v for k, v in r.get("metadata", {}).items()},
                    "source": "su_memory"
                }
            )

            # 设置关系（如果有因果链接）
            if r.get("hops", 0) > 0:
                path = r.get("path", [])
                if len(path) > 1:
                    # 添加前一个节点作为父节点
                    node.relationships = {
                        NodeRelationship.PARENT: RelatedNodeInfo(
                            node_id=path[-2],
                            metadata={"type": "causal_link"}
                        )
                    }

            nodes.append(node)

        return nodes

    def retrieve(self, str_or_query_bundle) -> List["TextNode"]:
        """兼容旧版 API"""
        return self._retrieve(str_or_query_bundle)


class SuMemoryLlamaIndexQueryEngine(BaseQueryEngine if LLAMAINDEX_AVAILABLE else object):
    """
    LlamaIndex QueryEngine 实现

    提供完整的查询引擎功能

    Example:
        >>> from llama_index.core import VectorStoreIndex
        >>> from su_memory.integrations.llamaindex import SuMemoryLlamaIndexQueryEngine
        >>>
        >>> # 创建查询引擎
        >>> query_engine = SuMemoryLlamaIndexQueryEngine(memory_client)
        >>>
        >>> # 查询
        >>> response = query_engine.query("今天发生了什么?")
        >>> print(response)
    """

    def __init__(
        self,
        memory_client,
        top_k: int = 5,
        enable_multihop: bool = True,
        callback_manager: Optional[CallbackManager] = None
    ):
        """
        初始化查询引擎

        Args:
            memory_client: SuMemoryLite 或 SuMemoryLitePro 实例
            top_k: 返回结果数量
            enable_multihop: 是否启用多跳推理
            callback_manager: 回调管理器
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError("请安装 LlamaIndex")

        super().__init__(callback_manager=callback_manager)

        self._client = memory_client
        self._top_k = top_k
        self._enable_multihop = enable_multihop

    def _query(self, query_bundle) -> "Response":  # noqa: F821
        """执行查询"""
        from llama_index.core.base.response import Response

        query = query_bundle.query_str
        all_nodes = []

        # 1. 基础相似度搜索
        basic_results = self._client.query(query, top_k=self._top_k)
        all_nodes.extend(basic_results)

        # 2. 多跳推理（如果启用且可用）
        if self._enable_multihop:
            try:
                multihop_results = self._client.query_multihop(query, max_hops=3)
                # 合并结果，避免重复
                existing_ids = {r.get("id", r.get("memory_id", "")) for r in basic_results}
                for r in multihop_results:
                    rid = r.get("id", r.get("memory_id", ""))
                    if rid not in existing_ids:
                        all_nodes.append(r)
                        existing_ids.add(rid)
            except AttributeError:
                pass  # SuMemoryLite 不支持多跳

        # 构建响应
        source_nodes = []
        for r in all_nodes[:self._top_k]:
            node = TextNode(
                text=r.get("content", ""),
                id_=r.get("memory_id", r.get("id", "")),
                score=r.get("score", 0),
                metadata={"source": "su_memory"}
            )
            source_nodes.append(node)

        # 生成上下文文本
        context_text = "\n\n".join([
            f"[记忆 {i+1}] {r.get('content', '')}"
            for i, r in enumerate(all_nodes[:self._top_k])
        ])

        response = Response(
            response=context_text,
            source_nodes=source_nodes,
            metadata={"retriever": "su_memory"}
        )

        return response

    def query(self, query_str: str):
        """执行查询（兼容 API）"""
        from llama_index.core.query_engine import QueryBundle
        return self._query(QueryBundle(query_str=query_str))


class SuMemoryLlamaIndexReader:
    """
    LlamaIndex DocumentReader 实现

    将 su-memory 记忆读取为 LlamaIndex Document

    Example:
        >>> from su_memory.integrations.llamaindex import SuMemoryLlamaIndexReader
        >>> from llama_index.core import VectorStoreIndex
        >>>
        >>> reader = SuMemoryLlamaIndexReader(memory_client)
        >>> documents = reader.load_data()
        >>>
        >>> # 创建索引
        >>> index = VectorStoreIndex.from_documents(documents)
    """

    def __init__(
        self,
        memory_client,
        session_id: Optional[str] = None
    ):
        """
        初始化读取器

        Args:
            memory_client: SuMemoryLite 或 SuMemoryLitePro 实例
            session_id: 可选，限定加载特定会话的记忆
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError("请安装 LlamaIndex")

        self._client = memory_client
        self._session_id = session_id

    def load_data(self, show_progress: bool = False) -> List[LLMDocument]:
        """
        加载数据

        Args:
            show_progress: 是否显示进度

        Returns:
            Document 列表
        """
        documents = []

        try:
            memories = self._client.get_all_memories()
        except AttributeError:
            return documents

        for mem in memories:
            # 如果指定了会话ID，只加载该会话的记忆
            if self._session_id:
                mem_session = mem.get("metadata", {}).get("session_id")
                if mem_session != self._session_id:
                    continue

            doc = LLMDocument(
                text=mem.get("content", ""),
                metadata={
                    "memory_id": mem.get("id", mem.get("memory_id", "")),
                    "timestamp": mem.get("timestamp", 0),
                    "session_id": mem.get("metadata", {}).get("session_id", ""),
                    "source": "su_memory"
                }
            )
            documents.append(doc)

        return documents

    def lazy_load_data(self, show_progress: bool = False):
        """懒加载（生成器）"""
        try:
            memories = self._client.get_all_memories()
        except AttributeError:
            return

        for mem in memories:
            if self._session_id:
                mem_session = mem.get("metadata", {}).get("session_id")
                if mem_session != self._session_id:
                    continue

            yield LLMDocument(
                text=mem.get("content", ""),
                metadata={
                    "memory_id": mem.get("id", mem.get("memory_id", "")),
                    "timestamp": mem.get("timestamp", 0),
                    "source": "su_memory"
                }
            )


class SuMemoryIndex:
    """
    LlamaIndex VectorStoreIndex 实现

    将 su-memory 作为 LlamaIndex 的向量索引使用。

    Example:
        >>> from llama_index.core import VectorStoreIndex
        >>> from su_memory.integrations.llamaindex import SuMemoryIndex
        >>>
        >>> # 创建索引
        >>> index = SuMemoryIndex(memory_client)
        >>>
        >>> # 添加节点
        >>> index.insert_nodes([node1, node2])
        >>>
        >>> # 查询
        >>> retriever = index.as_retriever()
        >>> nodes = retriever.retrieve("查询文本")
    """

    def __init__(
        self,
        memory_client,
        index_name: str = "su_memory_index",
        callback_manager: Optional[CallbackManager] = None
    ):
        """
        初始化索引

        Args:
            memory_client: SuMemoryLite 或 SuMemoryLitePro 实例
            index_name: 索引名称
            callback_manager: 回调管理器
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError("请安装 LlamaIndex")


        self._client = memory_client
        self._index_name = index_name
        self._callback_manager = callback_manager
        self._nodes = []

    def insert_nodes(self, nodes: List["TextNode"]) -> None:
        """插入节点到索引

        Args:
            nodes: TextNode 列表
        """
        for node in nodes:
            # 将节点添加到记忆
            self._client.add(
                content=node.text,
                metadata={
                    "node_id": node.id_,
                    "index_name": self._index_name,
                    "score": node.score or 0,
                    **node.metadata
                }
            )
            self._nodes.append(node)

    def insert(self, text: str, metadata: Optional[Dict] = None) -> str:
        """插入文本

        Args:
            text: 文本内容
            metadata: 元数据

        Returns:
            记忆ID
        """
        return self._client.add(
            content=text,
            metadata={**(metadata or {}), "index_name": self._index_name}
        )

    def as_retriever(self, similarity_top_k: int = 5) -> "SuMemoryLlamaIndexRetriever":
        """转换为检索器

        Args:
            similarity_top_k: 返回数量

        Returns:
            SuMemoryLlamaIndexRetriever 实例
        """
        return SuMemoryLlamaIndexRetriever(
            memory_client=self._client,
            top_k=similarity_top_k,
            callback_manager=self._callback_manager
        )

    def as_query_engine(self, **kwargs) -> "SuMemoryLlamaIndexQueryEngine":
        """转换为查询引擎

        Args:
            **kwargs: 其他参数

        Returns:
            SuMemoryLlamaIndexQueryEngine 实例
        """
        return SuMemoryLlamaIndexQueryEngine(
            memory_client=self._client,
            callback_manager=self._callback_manager,
            **kwargs
        )

    def get_nodes(self) -> List["TextNode"]:
        """获取所有节点

        Returns:
            TextNode 列表
        """
        return self._nodes.copy()

    def delete(self, node_id: str) -> None:
        """删除节点

        Args:
            node_id: 节点ID
        """
        # 从记忆中删除
        self._client.delete(node_id)
        # 从本地节点列表移除
        self._nodes = [n for n in self._nodes if n.id_ != node_id]


def create_vector_index(memory_client, **kwargs) -> Optional[Any]:
    """
    创建向量索引的便捷函数

    Args:
        memory_client: SuMemoryLite 或 SuMemoryLitePro 实例
        **kwargs: 其他参数传递给 VectorStoreIndex

    Returns:
        VectorStoreIndex 或 None
    """
    if not LLAMAINDEX_AVAILABLE:
        return None

    from llama_index.core import VectorStoreIndex

    reader = SuMemoryLlamaIndexReader(memory_client)
    documents = reader.load_data()

    if not documents:
        return None

    return VectorStoreIndex.from_documents(documents, **kwargs)


def create_query_engine(memory_client, **kwargs) -> Optional[Any]:
    """
    创建查询引擎的便捷函数

    Args:
        memory_client: SuMemoryLite 或 SuMemoryLitePro 实例
        **kwargs: 其他参数

    Returns:
        QueryEngine 或 None
    """
    if not LLAMAINDEX_AVAILABLE:
        return None

    index = create_vector_index(memory_client, **kwargs)
    if index is None:
        return None

    return index.as_query_engine()


# 导出
__all__ = [
    "SuMemoryLlamaIndexRetriever",
    "SuMemoryLlamaIndexQueryEngine",
    "SuMemoryLlamaIndexReader",
    "SuMemoryIndex",
    "SuMemoryIndexConfig",
    "create_vector_index",
    "create_query_engine",
    "LLAMAINDEX_AVAILABLE",
]
