"""
su_memory — Semantic Memory Engine

一行代码让AI应用拥有记忆能力。

Example:
    >>> from su_memory import SuMemory
    >>> client = SuMemory()
    >>> client.add("这个项目的ROI增长了25%")
    >>> results = client.query("投资汇报关")
"""

__version__ = "3.5.5"

# 环境检测：确保安装正确
import os
import shutil
import sys

# 检测标志：确保只提示一次
_ENV_CHECKDone = False

# 检查 pip 和 python 环境一致性
def _check_installation():
    """检查安装环境"""
    global _ENV_CHECKDone

    # 已检测过，跳过
    if _ENV_CHECKDone:
        return

    _ENV_CHECKDone = True

    python_path = os.path.dirname(os.path.dirname(sys.executable))
    pip_path = shutil.which("pip")

    if pip_path:
        pip_dir = os.path.dirname(os.path.dirname(pip_path))
        if python_path != pip_dir:
            import warnings
            warnings.warn(
                f"\n⚠️  su-memory 安装环境警告:\n"
                f"   Python: {sys.executable}\n"
                f"   pip:    {pip_path}\n"
                f"   pip 和 python 可能指向不同环境。\n"
                f"   建议使用: python -m pip install su-memory\n",
                UserWarning,
                stacklevel=2
            )

# 仅在首次导入时检测（可设置环境变量跳过）
if not os.environ.get("SU_MEMORY_SKIP_ENV_CHECK"):
    try:
        _check_installation()
    except Exception:
        pass  # 静默忽略检查错误，避免影响正常功能

from su_memory.client import SuMemory
from su_memory.sdk import SuMemoryLite, SuMemoryLitePro

# 导入增强检索器 — 核心模块，保持 eager
try:
    from su_memory.sdk.enhanced_retriever import EnhancedRetriever
except ImportError:
    EnhancedRetriever = None

# 导入 VectorGraphRAG — 核心模块，保持 eager
try:
    from su_memory.sdk.vector_graph_rag import VectorGraphRAG, create_vector_graph_rag
except ImportError:
    VectorGraphRAG = None
    create_vector_graph_rag = None

# 数据迁移 — 核心功能，保持 eager
from su_memory._sys.migrator import (
    DataSourceType,
    MemoryMigrator,
    MemoryRecord,
    MigrationReport,
    migrate_csv,
    migrate_json,
    migrate_obsidian,
    migrate_sqlite,
)
from su_memory.core import (
    BeliefTracker,
    CausalChain,
    CausalInference,
    DynamicPriorityCalculator,
    EncoderCore,
    EncodingInfo,
    HopResult,
    IntentClassifier,
    IntentConfig,
    MetaCognition,
    MultiHopRetriever,
    ProgressiveDisclosure,
    RecallResponse,
    RecallResult,
    RecallTrigger,
    RecencyFeedbackSystem,
    SemanticEncoder,
    SessionBridge,
    SessionContext,
    SuCompressor,
    WikiLinker,
    WikiResult,
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
    # 异步客户端 (v2.7.0)
    "AsyncSuMemory",
    "StreamChunk",
    # 异步存储抽象 (v2.7.0)
    "StorageBackend",
    "AsyncMemoryItem",
    "PgVectorBackend",
    "TieredStorage",
    "TierConfig",
    # 嵌入服务
    "EmbeddingProvider",
    "EmbeddingResult",
    "OpenAIEmbedder",
    "MiniMaxEmbedder",
    "OllamaEmbedder",
    "ChromaEmbedder",
    "EmbeddingFactory",
    "get_embedder",

    # 时空索引模块
    "SpacetimeIndexEngine",
    "SpacetimeNode",
    "SpacetimeConfig",
    "create_spacetime_engine",
    "create_energy_aware_node",
    "ENERGY_TO_SEASON",
    "ENERGY_TO_FOUR_PHASE",

    # 自适应引擎 (v1.6.0)
    "AdaptiveEngine",
    "ParameterSpace",
    "LearningMetrics",
    "ParameterType",
    "MetricType",
    "AdaptationStrategy",
    "create_adaptive_engine",
    "create_parameter_space",
    "create_metrics_collector",

    # 参数适配器 (v1.6.0 W19-W20)
    "RetrievalWeightAdapter",
    "EncodingDimensionAdapter",
    "CacheStrategyAdapter",
    "ParameterAdapterRegistry",
    "AdapterType",
    "CacheStrategy",
    "create_retrieval_adapter",
    "create_encoding_adapter",
    "create_cache_adapter",
    "create_adapter_registry",

    # 本地预测模型 (v1.6.0 W21-W22)
    "LocalModelManager",
    "PredictionCache",
    "SimpleLinearModel",
    "NaiveBayesClassifier",
    "TFIDFRanker",
    "ModelType",
    "PredictionStatus",
    "CacheEvictionPolicy",
    "create_linear_model",
    "create_naive_bayes",
    "create_tfidf_ranker",
    "create_prediction_cache",
    "create_model_manager",

    # 增量学习 (v1.6.0 W23-W24)
    "IncrementalLearningManager",
    "FeedbackLoop",
    "IncrementalUpdater",
    "MemoryForgetting",
    "FeedbackType",
    "UpdateStrategy",
    "ForgettingPolicy",
    "create_feedback_loop",
    "create_incremental_updater",
    "create_memory_forgetting",
    "create_learning_manager",

    # 插件系统 (v1.7.0 W25-W26)
    "PluginInterface",
    "PluginMetadata",
    "PluginState",
    "PluginType",
    "PluginRegistry",
    "PluginAlreadyExistsError",
    "PluginNotFoundError",
    "PluginDependencyError",
    "SandboxedExecutor",
    "ExecutionResult",
    "ResourceLimit",
    "get_registry",
    "register_plugin",
    "unregister_plugin",
    "get_plugin",
    "list_plugins",
    "execute_plugin",
    # 官方插件
    "TextEmbeddingPlugin",
    "RerankPlugin",
    "MonitorPlugin",
    "HashVectorizer",
    "RerankScorer",
    "PerformanceMetrics",
    "MonitorContext",

    # 本地数据管理 (v1.7.0 W29-W30)
    "SQLiteBackend",
    "MemoryItem",
    "AutoCompressor",
    "BackupManager",
    "DataExporter",

    # CLI工具 (v1.7.0 W31-W32)
    "cli",

    # LangChain集成 (v1.7.0 W31-W32)
    "SuMemoryRetriever",
    "SuMemoryRetrieverConfig",
    "SuMemoryLoader",
    "SuMemoryTool",
    "SuMemoryMemory",
    "create_rag_chain",
    "create_conversational_chain",
    "LANGCHAIN_AVAILABLE",

    # LlamaIndex集成 (v1.7.0 W31-W32)
    "SuMemoryLlamaIndexRetriever",
    "SuMemoryLlamaIndexQueryEngine",
    "SuMemoryLlamaIndexReader",
    "SuMemoryIndex",
    "SuMemoryIndexConfig",
    "create_vector_index",
    "create_query_engine",
    "LLAMAINDEX_AVAILABLE",

    # 统一异常体系 (v2.6.0)
    "ErrorCode",
    "SuMemoryError",
    "MemoryNotFoundError",
    "EncodingError",
    "StorageError",
    "ConfigurationError",
    "APIError",

    # ═══ 能量中心公开 API (v3.5.0) ═══
    # 天层 — 时空建模
    "TianGan",
    "DiZhi",
    "TemporalCore",
    "StemBranchCode",
    "create_stem_branch",
    "get_cycle_name",
    "TemporalSystem",
    "TemporalInfo",
    "DynamicPriority",
    # 地层 — 卦象空间
    "TrigramCore",
    "TaijiMapper",
    "PatternInference",
    # 人层 — 能量与因果
    "EnergyCore",
    "EnergyState",
    "EnergyBalanceResult",
    "EnergyFlow",
    "EnergyBus",
    "EnergyNode",
    "EnergyChannel",
    "EnergySignal",
    "EnergyLayer",
    "PropagationConfig",
    "create_energy_bus",
    "create_complete_energy_network",
    "CategoryCausalEngine",
    "EnergyMemoryNode",
    # 能量关系
    "analyze_balance",
    "calculate_link_weight",
    "analyze_relation",
    "get_affinity_score",
    "surface_entities",
    "find_reverse_causal_chain",
    "is_enhancing",
    "is_suppressing",
    "get_enhanced_energy",
    "get_cycle_sequence",
    "get_enhance_relation",
    "get_suppress_relation",
    "get_suppress_chain",
    "get_enhancing_energy",
    "get_suppressed_energy",
    "get_suppressing_energy",
    "RelationType",
    "EnergyRelation",
    "MemoryNodeEnergy",
    "RELATION_STRENGTH",
    "FOUR_SYMBOLS_TO_ENERGY",
    # 三才合一
    "UnifiedInfoUnit",
    "UnifiedInfoFactory",
    "create_unified_unit",
    # 检索融合
    "MultiViewRetriever",
    # 元认知
    "CognitiveGap",
    "KnowledgeAging",

    # 基础类型层 — 枚举系统 (v3.5.0)
    "YinYang",
    "ThreePowers",
    "FourSymbols",
    "Season",
    "TimeStem",
    "TimeBranch",
    "BranchRelation",
    "TrigramType",
    "TrigramRelation",
    "StrengthState",
    "EnergyPattern",

    # 基础类型层 — 语义/能量分类 (v3.5.0)
    "SemanticCategory",
    "MEMORY_TYPE_TO_CATEGORY",
    "EnergyNetwork",
    "ENERGY_ENHANCE_MAP",
    "ENERGY_SUPPRESS_MAP",
    "get_energy_state",
    "check_state_interaction",
    "energy_similarity",
    "energy_from_category",

    # 时空量化系统 (v3.5.0)
    "TimeCycle",
    "TimeCodeInfo",
    "create_time_code",
    "STEM_HE",
    "STEM_CHONG",
    "BRANCH_HE",
    "BRANCH_SANHE",
    "BRANCH_CHONG",
    "BRANCH_XING",
    "get_stem",
    "get_branch",
    "get_cycle",
]

# =============================================================================
# 懒加载模块 — 以下模块仅在首次访问时加载，减少启动时间
# =============================================================================

from su_memory._sys._lazy import LazyModule

_lazy = LazyModule(__name__)

_lazy.register("su_memory._sys._plugin_interface", [
    "PluginInterface", "PluginMetadata", "PluginState", "PluginType",
    "PluginEvent", "PluginEventHandler", "create_plugin_metadata", "validate_plugin",
])

_lazy.register("su_memory._sys._plugin_registry", [
    "PluginRegistry", "PluginAlreadyExistsError", "PluginNotFoundError",
    "PluginDependencyError", "PluginStateError",
    "get_registry", "register_plugin", "unregister_plugin",
    "get_plugin", "list_plugins",
])

_lazy.register("su_memory._sys._plugin_sandbox", [
    "SandboxedExecutor", "ExecutionResult", "ResourceLimit",
    "ExecutionContext", "SandboxEnvironment",
    "execute_plugin", "execute_with_retry", "get_default_executor",
])

_lazy.register("su_memory.plugins", [
    "TextEmbeddingPlugin", "RerankPlugin", "MonitorPlugin",
    "HashVectorizer", "RerankScorer", "ScoreResult",
    "PerformanceMetrics", "MonitorContext",
    "create_text_embedding_plugin", "create_rerank_plugin", "create_monitor_plugin",
])

_lazy.register("su_memory.embeddings.base", [
    "EmbeddingProvider", "EmbeddingResult",
    "OpenAIEmbedder", "MiniMaxEmbedder", "OllamaEmbedder", "ChromaEmbedder",
    "EmbeddingFactory", "get_embedder",
])

_lazy.register("su_memory._sys._spacetime_index", [
    "SpacetimeIndexEngine", "SpacetimeNode", "SpacetimeConfig",
    "create_spacetime_engine", "create_energy_aware_node",
    "ENERGY_TO_SEASON", "ENERGY_TO_FOUR_PHASE",
])

_lazy.register("su_memory._sys._adaptive_engine", [
    "AdaptiveEngine", "ParameterSpace", "LearningMetrics",
    "ParameterType", "MetricType", "AdaptationStrategy",
    "create_adaptive_engine", "create_parameter_space", "create_metrics_collector",
])

# ═══════════════════════════════════════════
# v3.5.0 能量中心公开 API — 懒加载注册
# ═══════════════════════════════════════════

# P0 — 已集成模块 (lite_pro 内部已使用)
_lazy.register("su_memory._sys._causal_engine", [
    "CategoryCausalEngine", "EnergyMemoryNode",
])
_lazy.register("su_memory._sys._unified_unit", [
    "UnifiedInfoUnit", "UnifiedInfoFactory",
    "create_unified_unit",
])
_lazy.register("su_memory._sys._energy_bus", [
    "EnergyBus", "EnergyNode", "EnergyChannel", "EnergySignal", "EnergyLayer",
    "PropagationConfig",
    "create_energy_bus", "create_complete_energy_network",
])
_lazy.register("su_memory._sys._energy_core", [
    "EnergyCore", "EnergyState", "EnergyBalanceResult", "EnergyFlow",
])

# P1 — 独立引擎
_lazy.register("su_memory._sys._temporal_core", [
    "TemporalCore", "StemBranchCode",
    "create_stem_branch", "get_cycle_name",
])
_lazy.register("su_memory._sys.chrono", [
    "TianGan", "DiZhi",
    "TemporalSystem", "TemporalInfo", "DynamicPriority",
])
_lazy.register("su_memory._sys._category_core", [
    "TrigramCore",
])
_lazy.register("su_memory._sys._dimension_map", [
    "TaijiMapper",
])
_lazy.register("su_memory._sys.fusion", [
    "MultiViewRetriever",
])
_lazy.register("su_memory._sys._pattern_inference", [
    "PatternInference",
])
_lazy.register("su_memory._sys._energy_relations", [
    "analyze_balance", "calculate_link_weight",
    "analyze_relation", "get_affinity_score",
    "surface_entities", "find_reverse_causal_chain",
    "is_enhancing", "is_suppressing",
    "get_enhanced_energy", "get_cycle_sequence",
    "get_enhance_relation", "get_suppress_relation",
    "get_suppress_chain",
    "get_enhancing_energy", "get_suppressed_energy",
    "get_suppressing_energy",
    "RelationType", "EnergyRelation",
    "MemoryNodeEnergy", "RELATION_STRENGTH",
    "FOUR_SYMBOLS_TO_ENERGY",
])

# v3.5.0 基础类型层 — Foundation Types
_lazy.register("su_memory._sys._enums", [
    "YinYang", "ThreePowers", "FourSymbols", "Season",
    "TimeStem", "TimeBranch", "BranchRelation",
    "TrigramType", "TrigramRelation",
    "StrengthState", "EnergyPattern",
])
_lazy.register("su_memory._sys._terms", [
    "SEMANTIC_CATEGORY", "SEMANTIC_CATEGORY_NAMES",
    "ENERGY_ENHANCE", "ENERGY_SUPPRESS",
    "STRENGTH_STATE", "MONTH_ENERGY_STATE", "STRENGTH_MULTIPLIER",
    "TIME_STEMS", "TIME_BRANCHES", "TIME_BRANCH_ENERGY",
    "STEM_HE_MAP", "STEM_CHONG_MAP",
    "BRANCH_HE_MAP", "BRANCH_CHONG_MAP",
    "BRANCH_SANHE_MAP", "BRANCH_HIDDEN_STEM_MAP",
    "TRIGRAM_ENERGY_MAP", "TRIGRAM_BODY_MAP",
])
_lazy.register("su_memory._sys._c1", [
    "SemanticCategory", "MEMORY_TYPE_TO_CATEGORY",
    "CATEGORY_ANCHORS", "KEYWORDS_TO_CATEGORY", "ENERGY_TO_CATEGORY",
])
_lazy.register("su_memory._sys._c2", [
    "EnergyState", "EnergyNetwork",
    "ENERGY_ENHANCE_MAP", "ENERGY_SUPPRESS_MAP", "STATE_STRENGTH_MAP",
    "get_energy_state", "check_state_interaction",
    "energy_similarity", "energy_from_category",
])

# v3.5.0 时空量化 + 元认知
_lazy.register("su_memory._sys._time_code", [
    "TimeCycle", "TimeCodeInfo", "create_time_code",
    "STEM_HE", "STEM_CHONG",
    "BRANCH_HE", "BRANCH_SANHE",
    "BRANCH_CHONG", "BRANCH_XING",
    "get_stem", "get_branch", "get_cycle",
])
_lazy.register("su_memory._sys.awareness", [
    "CognitiveGap", "KnowledgeAging",
])

_lazy.register("su_memory._sys._parameter_adapters", [
    "RetrievalWeightAdapter", "EncodingDimensionAdapter", "CacheStrategyAdapter",
    "ParameterAdapterRegistry", "AdapterType", "CacheStrategy",
    "create_retrieval_adapter", "create_encoding_adapter",
    "create_cache_adapter", "create_adapter_registry",
])

_lazy.register("su_memory._sys._local_models", [
    "LocalModelManager", "PredictionCache",
    "SimpleLinearModel", "NaiveBayesClassifier", "TFIDFRanker",
    "ModelType", "PredictionStatus", "CacheEvictionPolicy",
    "create_linear_model", "create_naive_bayes", "create_tfidf_ranker",
    "create_prediction_cache", "create_model_manager",
])

_lazy.register("su_memory._sys._incremental_learning", [
    "IncrementalLearningManager", "FeedbackLoop",
    "IncrementalUpdater", "MemoryForgetting",
    "FeedbackType", "UpdateStrategy", "ForgettingPolicy",
    "create_feedback_loop", "create_incremental_updater",
    "create_memory_forgetting", "create_learning_manager",
])

_lazy.register("su_memory.storage", [
    "SQLiteBackend", "MemoryItem", "AutoCompressor",
    "BackupManager", "DataExporter",
])

_lazy.register("su_memory.cli", ["cli"])

_lazy.register("su_memory.integrations.langchain", [
    "SuMemoryRetriever", "SuMemoryRetrieverConfig",
    "SuMemoryLoader", "SuMemoryTool", "SuMemoryMemory",
    "create_rag_chain", "create_conversational_chain",
    "LANGCHAIN_AVAILABLE",
])

_lazy.register("su_memory.integrations.llamaindex", [
    "SuMemoryLlamaIndexRetriever", "SuMemoryLlamaIndexQueryEngine",
    "SuMemoryLlamaIndexReader", "SuMemoryIndex", "SuMemoryIndexConfig",
    "create_vector_index", "create_query_engine",
    "LLAMAINDEX_AVAILABLE",
])

# 统一异常体系 — 通过 __getattr__ 延迟暴露
try:
    from su_memory.exceptions import (
        APIError as _APIError,
    )
    from su_memory.exceptions import (
        ConfigurationError as _ConfigurationError,
    )
    from su_memory.exceptions import (
        EncodingError as _EncodingError,
    )
    from su_memory.exceptions import (
        ErrorCode,
        MemoryNotFoundError,
        SuMemoryError,
    )
    from su_memory.exceptions import (
        StorageError as _StorageError,
    )
except ImportError:
    pass

# 安装懒加载
_lazy.install()

