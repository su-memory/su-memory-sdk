"""
su-memory LangChain适配器
用于将su-memory作为LangChain的记忆组件

支持 LangChain 的 BaseChatMemory 接口，
可以与 LangChain Agent 和 Chain 无缝集成。
"""
from typing import Any, Dict, List, Union

# LangChain相关导入（可选）
LANGCHAIN_AVAILABLE = False
BaseMessage = None
HumanMessage = None
AIMessage = None
SystemMessage = None
BaseChatMemory = None

try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.memory import BaseChatMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # 定义兼容性类型（当langchain-core未安装时）
    pass


class SimpleChatHistory:
    """
    简单聊天历史实现（当langchain不可用时使用）
    """

    def __init__(self):
        self.messages: List[Dict[str, str]] = []

    def add_user_message(self, message: str) -> None:
        self.messages.append({"type": "human", "content": message})

    def add_ai_message(self, message: str) -> None:
        self.messages.append({"type": "ai", "content": message})

    def clear(self) -> None:
        self.messages.clear()

    @property
    def messages(self) -> List[Dict[str, str]]:
        return self._messages

    @messages.setter
    def messages(self, value: List[Dict[str, str]]) -> None:
        self._messages = value


class SuMemoryChatMemory:
    """
    LangChain BaseChatMemory接口适配器

    将su-memory作为LangChain的记忆组件使用。
    支持完整的对话历史管理和语义检索。

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain.agents import Agent
        >>> from su_memory.adapters import SuMemoryChatMemory
        >>> from su_memory.sdk import SuMemoryLite
        >>>
        >>> # 创建记忆客户端
        >>> client = SuMemoryLite()
        >>> memory = SuMemoryChatMemory(memory_key="chat_history", client=client)
        >>>
        >>> # 与LangChain集成
        >>> llm = ChatOpenAI(temperature=0)
        >>> agent = Agent(llm=llm, memory=memory, ...)
    """

    # 检查LangChain是否可用
    _langchain_available = LANGCHAIN_AVAILABLE

    def __init__(
        self,
        client=None,
        memory_key: str = "chat_history",
        return_messages: bool = False,
        chat_memory=None,
        **kwargs
    ):
        """
        初始化记忆组件

        Args:
            client: SuMemory客户端实例（SuMemoryClient或SuMemoryLite）
            memory_key: 记忆变量名
            return_messages: 是否返回消息对象而非字符串
            chat_memory: LangChain的ChatMessageHistory实例（可选）
        """
        self.client = client
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.input_key = kwargs.get("input_key")
        self.output_key = kwargs.get("output_key")

        # 使用提供的chat_memory或创建新的
        if chat_memory is not None:
            self.chat_memory = chat_memory
        elif LANGCHAIN_AVAILABLE:
            from langchain.memory import ChatMessageHistory
            self.chat_memory = ChatMessageHistory()
        else:
            self.chat_memory = SimpleChatHistory()

    @property
    def buffer(self) -> Union[str, List[Any]]:
        """
        获取对话缓冲区

        Returns:
            如果return_messages为True，返回消息列表；
            否则返回格式化的字符串。
        """
        if LANGCHAIN_AVAILABLE:
            messages = self.chat_memory.messages
        else:
            messages = self.chat_memory.messages

        if self.return_messages and LANGCHAIN_AVAILABLE:
            return messages

        # 返回格式化字符串
        return "\n".join([
            f"{self._get_message_type(m)}: {self._get_message_content(m)}"
            for m in messages
        ])

    def _get_message_type(self, message: Any) -> str:
        """
        获取消息类型名称
        """
        if not LANGCHAIN_AVAILABLE:
            msg_type = message.get("type", "unknown")
            # 转换为标准格式
            if msg_type == "human":
                return "Human"
            elif msg_type == "ai":
                return "AI"
            return msg_type.capitalize()

        if isinstance(message, HumanMessage):
            return "Human"
        elif isinstance(message, AIMessage):
            return "AI"
        elif isinstance(message, SystemMessage):
            return "System"
        return "Unknown"

    def _get_message_content(self, message: Any) -> str:
        """
        获取消息内容
        """
        if not LANGCHAIN_AVAILABLE:
            return message.get("content", "")
        return message.content

    @property
    def memory_variables(self) -> List[str]:
        """
        获取记忆变量列表
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        加载记忆变量
        """
        return {self.memory_key: self.buffer}

    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str]
    ) -> None:
        """
        保存对话上下文到记忆
        """
        # 获取输入文本
        input_key = self.input_key if self.input_key else "input"
        input_text = inputs.get(input_key, "")

        # 获取输出文本
        output_key = self.output_key if self.output_key else "output"
        output_text = outputs.get(output_key, "")

        if not input_text and not output_text:
            return

        # 格式化上下文
        context = f"Human: {input_text}\nAI: {output_text}"

        # 保存到su-memory
        if self.client:
            self.client.add(context, metadata={
                "type": "conversation",
                "input": input_text[:200] if input_text else None,
                "output": output_text[:200] if output_text else None
            })

        # 保存到ChatMessageHistory
        if LANGCHAIN_AVAILABLE:
            if input_text:
                self.chat_memory.add_user_message(input_text)
            if output_text:
                self.chat_memory.add_ai_message(output_text)
        else:
            if input_text:
                self.chat_memory.add_user_message(input_text)
            if output_text:
                self.chat_memory.add_ai_message(output_text)

    def clear(self) -> None:
        """
        清空所有记忆
        """
        # 清空su-memory
        if self.client and hasattr(self.client, 'clear'):
            self.client.clear()

        # 清空ChatMessageHistory
        self.chat_memory.clear()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索相关记忆
        """
        if self.client and hasattr(self.client, 'query'):
            return self.client.query(query, top_k=top_k)
        return []

    def search_metadata(
        self,
        metadata_filter: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        根据元数据过滤检索记忆
        """
        if not self.client or not hasattr(self.client, '_memories'):
            return []

        results = []
        for memory in self.client._memories:
            mem_metadata = memory.get("metadata", {})
            if all(mem_metadata.get(k) == v for k, v in metadata_filter.items()):
                results.append(memory)
                if len(results) >= top_k:
                    break

        return results


# 保持向后兼容
SuMemoryMemory = SuMemoryChatMemory
