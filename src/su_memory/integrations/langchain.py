"""
su-memory SDK × LangChain 集成适配器

提供与 LangChain 的深度集成，支持：
- BaseMemoryStore 接口实现
- Runnable 接口支持
- 检索增强生成 (RAG) 管道集成
"""

from typing import List, Dict, Any, Optional, Sequence
from dataclasses import dataclass

# LangChain 相关导入（可选）
LANGCHAIN_AVAILABLE = False
Document = None
BaseRetriever = None
Runnable = None
CallbackManagerForRetrieverRun = None

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.runnables import Runnable, RunnablePassthrough
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass


@dataclass
class SuMemoryRetrieverConfig:
    """检索器配置"""
    search_type: str = "similarity"  # similarity, mmr, similarity_score_threshold
    top_k: int = 5
    threshold: float = 0.5
    filter: Optional[Dict] = None
    fetch_k: int = 20
    lambda_mult: float = 0.5


class SuMemoryRetriever:
    """
    LangChain Retriever 实现

    将 su-memory SDK 作为 LangChain 的检索器使用

    Example:
        >>> from langchain_openai import OpenAI
        >>> from langchain.chains import RetrievalQA
        >>> from su_memory.integrations.langchain import SuMemoryRetriever
        >>>
        >>> # 创建检索器
        >>> retriever = SuMemoryRetriever(pro, top_k=5)
        >>>
        >>> # 创建 QA 链
        >>> qa_chain = RetrievalQA.from_chain_type(
        ...     llm=OpenAI(),
        ...     chain_type="stuff",
        ...     retriever=retriever
        ... )
        >>>
        >>> # 查询
        >>> result = qa_chain({"query": "今天发生了什么?"})
    """

    def __init__(
        self,
        memory_client,
        config: Optional[SuMemoryRetrieverConfig] = None
    ):
        """
        初始化检索器

        Args:
            memory_client: SuMemoryLite 或 SuMemoryLitePro 实例
            config: 检索配置
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "请安装 LangChain: pip install langchain-core langchain-openai\n"
                "或查看文档: https://python.langchain.com/docs/get_started/installation"
            )

        self._client = memory_client
        self._config = config or SuMemoryRetrieverConfig()

    def _search_similarity(self, query: str, top_k: int) -> List[Document]:
        """相似度搜索"""
        results = self._client.query(query, top_k=top_k)

        return [
            Document(
                page_content=r.get("content", ""),
                metadata={
                    "memory_id": r.get("memory_id", r.get("id", "")),
                    "score": r.get("score", 0),
                    "source": "su_memory",
                    **{k: v for k, v in r.get("metadata", {}).items()}
                }
            )
            for r in results
            if r.get("score", 0) >= self._config.threshold
        ]

    def _search_multihop(self, query: str, top_k: int) -> List[Document]:
        """多跳推理搜索"""
        try:
            results = self._client.query_multihop(query, max_hops=3)
        except AttributeError:
            # SuMemoryLite 不支持多跳
            return self._search_similarity(query, top_k)

        return [
            Document(
                page_content=r.get("content", ""),
                metadata={
                    "memory_id": r.get("memory_id", r.get("id", "")),
                    "score": r.get("score", 0),
                    "hops": r.get("hops", 1),
                    "path": r.get("path", []),
                    "causal_type": r.get("causal_type", ""),
                    "source": "su_memory_multihop",
                    **{k: v for k, v in r.get("metadata", {}).items()}
                }
            )
            for r in results
        ]

    def invoke(self, query: str) -> List[Document]:
        """LangChain Runnable 接口"""
        return self._search_similarity(query, self._config.top_k)

    def get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """BaseRetriever 接口实现"""
        if self._config.search_type == "similarity":
            return self._search_similarity(query, self._config.top_k)
        elif self._config.search_type == "mmr":
            # MMR (Maximal Marginal Relevance) 需要额外实现
            return self._search_similarity(query, self._config.top_k)
        else:
            return self._search_similarity(query, self._config.top_k)


class SuMemoryLoader:
    """
    LangChain DocumentLoader 实现

    将 su-memory 中的记忆加载为 LangChain Document

    Example:
        >>> from su_memory.integrations.langchain import SuMemoryLoader
        >>> from langchain.text_splitter import RecursiveCharacterTextSplitter
        >>>
        >>> loader = SuMemoryLoader(pro)
        >>> documents = loader.load()
        >>>
        >>> # 可选：分割长文档
        >>> splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        >>> chunks = splitter.split_documents(documents)
    """

    def __init__(self, memory_client, session_id: Optional[str] = None):
        """
        初始化加载器

        Args:
            memory_client: SuMemoryLite 或 SuMemoryLitePro 实例
            session_id: 可选，限定加载特定会话的记忆
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("请安装 LangChain: pip install langchain-core")

        self._client = memory_client
        self._session_id = session_id

    def load(self) -> List[Document]:
        """加载所有记忆为 Document"""
        documents = []

        try:
            # 尝试获取所有记忆
            memories = self._client.get_all_memories()
        except AttributeError:
            # 如果没有 get_all_memories 方法，返回空
            return documents

        for mem in memories:
            # 如果指定了会话ID，只加载该会话的记忆
            if self._session_id:
                mem_session = mem.get("metadata", {}).get("session_id")
                if mem_session != self._session_id:
                    continue

            documents.append(Document(
                page_content=mem.get("content", ""),
                metadata={
                    "memory_id": mem.get("id", mem.get("memory_id", "")),
                    "timestamp": mem.get("timestamp", 0),
                    "session_id": mem.get("metadata", {}).get("session_id", ""),
                    "source": "su_memory"
                }
            ))

        return documents

    def lazy_load(self):
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

            yield Document(
                page_content=mem.get("content", ""),
                metadata={
                    "memory_id": mem.get("id", mem.get("memory_id", "")),
                    "timestamp": mem.get("timestamp", 0),
                    "source": "su_memory"
                }
            )


class SuMemoryTool:
    """
    LangChain Tool 实现
    
    将 su-memory 作为 LangChain Agent 的工具使用。
    
    Example:
        >>> from langchain.agents import AgentExecutor, create_react_agent
        >>> from langchain_openai import OpenAI
        >>> from su_memory.integrations.langchain import SuMemoryTool
        >>>
        >>> # 创建工具
        >>> tool = SuMemoryTool(pro, name="memory_search", description="搜索记忆")
        >>>
        >>> # 创建 Agent
        >>> agent = create_react_agent(
        ...     llm=OpenAI(),
        ...     tools=[tool]
        ... )
    """
    
    def __init__(
        self,
        memory_client,
        name: str = "memory_search",
        description: str = "从记忆中搜索相关信息。输入应该是搜索关键词。"
    ):
        """
        初始化工具
        
        Args:
            memory_client: SuMemoryLite 或 SuMemoryLitePro 实例
            name: 工具名称
            description: 工具描述
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("请安装 LangChain: pip install langchain-core")
        
        from langchain_core.tools import BaseTool
        
        self._client = memory_client
        self._name = name
        self._description = description
    
    @property
    def name(self) -> str:
        """工具名称"""
        return self._name
    
    @property
    def description(self) -> str:
        """工具描述"""
        return self._description
    
    def invoke(self, tool_input: str) -> str:
        """执行工具
        
        Args:
            tool_input: 工具输入（搜索关键词）
        
        Returns:
            搜索结果文本
        """
        query = tool_input.strip()
        if not query:
            return "请提供搜索关键词"
        
        try:
            results = self._client.query(query, top_k=5)
            
            if not results:
                return f"没有找到与 '{query}' 相关的内容"
            
            output = f"找到 {len(results)} 条相关记忆：\n\n"
            for i, r in enumerate(results, 1):
                content = r.get("content", "")
                score = r.get("score", 0)
                output += f"{i}. [{score:.2f}] {content}\n\n"
            
            return output
        except Exception as e:
            return f"搜索失败: {str(e)}"
    
    def run(self, tool_input: str) -> str:
        """同步运行工具"""
        return self.invoke(tool_input)
    
    def async_run(self, tool_input: str) -> str:
        """异步运行工具"""
        return self.invoke(tool_input)


class SuMemoryMemory:
    """
    LangChain Memory 接口实现

    将 su-memory 作为 LangChain Agent / Chain 的记忆存储

    Example:
        >>> from langchain.agents import AgentExecutor, create_react_agent
        >>> from langchain_openai import OpenAI
        >>> from su_memory.integrations.langchain import SuMemoryMemory
        >>>
        >>> # 创建记忆
        >>> memory = SuMemoryMemory(session_id="agent_session")
        >>>
        >>> # 创建 Agent
        >>> agent = create_react_agent(
        ...     llm=OpenAI(),
        ...     tools=tools,
        ...     memory=memory
        ... )
        >>>
        >>> # 执行
        >>> agent_executor = AgentExecutor(
        ...     agent=agent,
        ...     tools=tools,
        ...     memory=memory,
        ...     verbose=True
        ... )
    """

    def __init__(
        self,
        memory_client,
        session_id: Optional[str] = None,
        return_messages: bool = False,
        input_key: str = "input",
        output_key: str = "output"
    ):
        """
        初始化记忆

        Args:
            memory_client: SuMemoryLite 或 SuMemoryLitePro 实例
            session_id: 会话 ID
            return_messages: 是否返回消息格式
            input_key: 输入键名
            output_key: 输出键名
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("请安装 LangChain: pip install langchain-core")

        self._client = memory_client
        self._session_id = session_id
        self._return_messages = return_messages
        self._input_key = input_key
        self._output_key = output_key

    def load_memory_variables(
        self,
        inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        加载记忆变量

        Args:
            inputs: 输入字典

        Returns:
            包含记忆的字典
        """
        if inputs is None:
            inputs = {}

        query = inputs.get(self._input_key, "")

        if query:
            # 检索相关记忆
            results = self._client.query(query, top_k=5)

            if self._return_messages:
                messages = [
                    {"type": "human", "content": r.get("content", "")}
                    for r in results
                ]
                return {"history": messages}
            else:
                context = "\n".join([
                    f"- {r.get('content', '')}"
                    for r in results
                ])
                return {"history": context}

        return {"history": ""}

    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> None:
        """
        保存上下文到记忆

        Args:
            inputs: 输入字典
            outputs: 输出字典
        """
        input_text = inputs.get(self._input_key, "")
        output_text = outputs.get(self._output_key, "")

        if input_text:
            self._client.add(
                content=f"用户: {input_text}",
                metadata={"session_id": self._session_id, "type": "user"}
            )

        if output_text:
            self._client.add(
                content=f"助手: {output_text}",
                metadata={"session_id": self._session_id, "type": "assistant"}
            )

    def clear(self) -> None:
        """清除记忆"""
        # 如果客户端支持按会话清除
        if self._session_id and hasattr(self._client, "clear_session"):
            self._client.clear_session(self._session_id)


def create_rag_chain(
    memory_client,
    llm,
    chain_type: str = "stuff",
    search_type: str = "similarity",
    top_k: int = 5
):
    """
    创建 RAG 链的便捷函数

    Args:
        memory_client: SuMemoryLite 或 SuMemoryLitePro 实例
        llm: LangChain LLM 实例
        chain_type: 链类型 ("stuff", "map_reduce", "refine")
        search_type: 搜索类型
        top_k: 检索数量

    Returns:
        RAG Chain
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("请安装 LangChain")

    from langchain.chains import RetrievalQA

    retriever = SuMemoryRetriever(
        memory_client,
        config=SuMemoryRetrieverConfig(
            search_type=search_type,
            top_k=top_k
        )
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True
    )


def create_conversational_chain(
    memory_client,
    llm,
    system_prompt: Optional[str] = None
):
    """
    创建对话链的便捷函数

    Args:
        memory_client: SuMemoryLite 或 SuMemoryLitePro 实例
        llm: LangChain LLM 实例
        system_prompt: 系统提示

    Returns:
        Conversational Chain
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("请安装 LangChain")

    from langchain.chains import ConversationChain
    from langchain.prompts import PromptTemplate

    memory = SuMemoryMemory(memory_client)

    if system_prompt:
        template = system_prompt + "\n\n{history}\n\n用户: {input}\n助手:"
    else:
        template = """你是一个有帮助的AI助手。以下是与用户之前的对话历史：

{history}

用户: {input}
助手:"""

    prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    return ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )


# 导出
__all__ = [
    "SuMemoryRetriever",
    "SuMemoryRetrieverConfig",
    "SuMemoryLoader",
    "SuMemoryTool",
    "SuMemoryMemory",
    "create_rag_chain",
    "create_conversational_chain",
    "LANGCHAIN_AVAILABLE",
]
