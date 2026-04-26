"""
su-memory Integrations Package (集成适配器包)

提供与主流AI框架的集成适配器。

Available Adapters:
- SuMemoryRetriever: LangChain Retriever适配器
- SuMemoryMemory: LangChain Memory适配器
- SuMemoryTool: LangChain Tool适配器
- SuMemoryIndex: LlamaIndex索引连接器

Example:
    >>> from su_memory.integrations import SuMemoryRetriever
    >>> retriever = SuMemoryRetriever()
"""

from .langchain import SuMemoryRetriever, SuMemoryMemory, SuMemoryTool, SuMemoryLoader
from .llamaindex import SuMemoryIndex, SuMemoryLlamaIndexRetriever

__all__ = [
    # LangChain
    "SuMemoryRetriever",
    "SuMemoryMemory",
    "SuMemoryTool",
    "SuMemoryLoader",
    # LlamaIndex
    "SuMemoryIndex",
    "SuMemoryLlamaIndexRetriever",
]
