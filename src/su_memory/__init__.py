"""
su_memory — 易学AI记忆引擎

一行代码让AI应用拥有记忆能力。

Example:
    >>> from su_memory import SuMemory
    >>> client = SuMemory()
    >>> client.add("这个项目的ROI增长了25%")
    >>> results = client.query("投资汇报关")
"""

__version__ = "1.3.0"

from su_memory.client import SuMemory

from su_memory.sdk import SuMemoryLite, SuMemoryLitePro

# 导入增强检索器
try:
    from su_memory.sdk.enhanced_retriever import EnhancedRetriever
except ImportError:
    EnhancedRetriever = None

# 导入 VectorGraphRAG
try:
    from su_memory.sdk.vector_graph_rag import VectorGraphRAG, create_vector_graph_rag
except ImportError:
    VectorGraphRAG = None
    create_vector_graph_rag = None

from su_memory.core import (
    CausalChain,
    CausalInference,
    MetaCognition,
    SuCompressor,
    BeliefTracker,
    DynamicPriorityCalculator,
    SemanticEncoder,
    EncoderCore,
    EncodingInfo,
    IntentClassifier,
    IntentConfig,
    ProgressiveDisclosure,
    SessionBridge,
    SessionContext,
    RecencyFeedbackSystem,
    MultiHopRetriever,
    HopResult,
    WikiLinker,
    WikiResult,
    RecallTrigger,
    RecallResult,
    RecallResponse,
)
from su_memory.encoding import MemoryEncoding

__all__ = [
    # SDK客户端
    "SuMemoryLite",
    "SuMemoryLitePro",
    
    "SuMemory",
    
    # 增强检索器
    "EnhancedRetriever",
    
    # VectorGraphRAG 多跳推理
    "VectorGraphRAG",
    "create_vector_graph_rag",
    
    "CausalChain",
    "CausalInference",
    "MetaCognition",
    "SuCompressor",
    "BeliefTracker",
    "DynamicPriorityCalculator",
    "SemanticEncoder",
    "EncoderCore",
    "EncodingInfo",
    "MemoryEncoding",
    "IntentClassifier",
    "IntentConfig",
    "ProgressiveDisclosure",
    "SessionBridge",
    "SessionContext",
    "RecencyFeedbackSystem",
    "MultiHopRetriever",
    "HopResult",
    "WikiLinker",
    "WikiResult",
    "RecallTrigger",
    "RecallResult",
    "RecallResponse",
    # 数据迁移
    "MemoryMigrator",
    "MemoryRecord",
    "MigrationReport",
    "DataSourceType",
    "migrate_json",
    "migrate_csv",
    "migrate_sqlite",
    "migrate_obsidian",
]

# 数据迁移模块
from su_memory._sys.migrator import (
    MemoryMigrator,
    MemoryRecord,
    MigrationReport,
    DataSourceType,
    migrate_json,
    migrate_csv,
    migrate_sqlite,
    migrate_obsidian,
)

