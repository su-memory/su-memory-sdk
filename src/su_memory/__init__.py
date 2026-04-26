"""
su_memory — Semantic Memory Engine

一行代码让AI应用拥有记忆能力。

Example:
    >>> from su_memory import SuMemory
    >>> client = SuMemory()
    >>> client.add("这个项目的ROI增长了25%")
    >>> results = client.query("投资汇报关")
"""

__version__ = "1.7.0"

# 环境检测：确保安装正确
import os
import sys
import shutil
import site as _site

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
]

# 插件系统模块 (v1.7.0 W25-W26)
try:
    from su_memory._sys._plugin_interface import (
        PluginInterface,
        PluginMetadata,
        PluginState,
        PluginType,
        PluginEvent,
        PluginEventHandler,
        create_plugin_metadata,
        validate_plugin,
    )
except ImportError:
    PluginInterface = None
    PluginMetadata = None
    PluginState = None
    PluginType = None
    PluginEvent = None
    PluginEventHandler = None
    create_plugin_metadata = None
    validate_plugin = None

try:
    from su_memory._sys._plugin_registry import (
        PluginRegistry,
        PluginAlreadyExistsError,
        PluginNotFoundError,
        PluginDependencyError,
        PluginStateError,
        get_registry,
        register_plugin,
        unregister_plugin,
        get_plugin,
        list_plugins,
    )
except ImportError:
    PluginRegistry = None
    PluginAlreadyExistsError = None
    PluginNotFoundError = None
    PluginDependencyError = None
    PluginStateError = None
    get_registry = None
    register_plugin = None
    unregister_plugin = None
    get_plugin = None
    list_plugins = None

try:
    from su_memory._sys._plugin_sandbox import (
        SandboxedExecutor,
        ExecutionResult,
        ResourceLimit,
        ExecutionContext,
        SandboxEnvironment,
        execute_plugin,
        execute_with_retry,
        get_default_executor,
    )
except ImportError:
    SandboxedExecutor = None
    ExecutionResult = None
    ResourceLimit = None
    ExecutionContext = None
    SandboxEnvironment = None
    execute_plugin = None
    execute_with_retry = None
    get_default_executor = None

# 官方插件
try:
    from su_memory.plugins import (
        TextEmbeddingPlugin,
        RerankPlugin,
        MonitorPlugin,
        HashVectorizer,
        RerankScorer,
        ScoreResult,
        PerformanceMetrics,
        MonitorContext,
        create_text_embedding_plugin,
        create_rerank_plugin,
        create_monitor_plugin,
    )
except ImportError:
    TextEmbeddingPlugin = None
    RerankPlugin = None
    MonitorPlugin = None
    HashVectorizer = None
    RerankScorer = None
    ScoreResult = None
    PerformanceMetrics = None
    MonitorContext = None
    create_text_embedding_plugin = None
    create_rerank_plugin = None
    create_monitor_plugin = None

# 向量嵌入服务
try:
    from su_memory.embeddings.base import (
        EmbeddingProvider,
        EmbeddingResult,
        OpenAIEmbedder,
        MiniMaxEmbedder,
        OllamaEmbedder,
        ChromaEmbedder,
        EmbeddingFactory,
        get_embedder,
    )
except ImportError:
    # 静默忽略，如果 embedder 可用会导入成功
    pass

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

# 时空索引模块
try:
    from su_memory._sys._spacetime_index import (
        SpacetimeIndexEngine,
        SpacetimeNode,
        SpacetimeConfig,
        create_spacetime_engine,
        create_energy_aware_node,
        ENERGY_TO_SEASON,
        ENERGY_TO_FOUR_PHASE,
    )
except ImportError:
    SpacetimeIndexEngine = None
    SpacetimeNode = None
    SpacetimeConfig = None
    create_spacetime_engine = None
    create_energy_aware_node = None
    ENERGY_TO_SEASON = {}
    ENERGY_TO_FOUR_PHASE = {}

# 自适应引擎模块 (v1.6.0)
try:
    from su_memory._sys._adaptive_engine import (
        AdaptiveEngine,
        ParameterSpace,
        LearningMetrics,
        ParameterType,
        MetricType,
        AdaptationStrategy,
        create_adaptive_engine,
        create_parameter_space,
        create_metrics_collector,
    )
except ImportError:
    AdaptiveEngine = None
    ParameterSpace = None
    LearningMetrics = None
    ParameterType = None
    MetricType = None
    AdaptationStrategy = None
    create_adaptive_engine = None
    create_parameter_space = None
    create_metrics_collector = None

# 参数适配器模块 (v1.6.0 W19-W20)
try:
    from su_memory._sys._parameter_adapters import (
        RetrievalWeightAdapter,
        EncodingDimensionAdapter,
        CacheStrategyAdapter,
        ParameterAdapterRegistry,
        AdapterType,
        CacheStrategy,
        create_retrieval_adapter,
        create_encoding_adapter,
        create_cache_adapter,
        create_adapter_registry,
    )
except ImportError:
    RetrievalWeightAdapter = None
    EncodingDimensionAdapter = None
    CacheStrategyAdapter = None
    ParameterAdapterRegistry = None
    AdapterType = None
    CacheStrategy = None
    create_retrieval_adapter = None
    create_encoding_adapter = None
    create_cache_adapter = None
    create_adapter_registry = None

# 本地预测模型模块 (v1.6.0 W21-W22)
try:
    from su_memory._sys._local_models import (
        LocalModelManager,
        PredictionCache,
        SimpleLinearModel,
        NaiveBayesClassifier,
        TFIDFRanker,
        ModelType,
        PredictionStatus,
        CacheEvictionPolicy,
        create_linear_model,
        create_naive_bayes,
        create_tfidf_ranker,
        create_prediction_cache,
        create_model_manager,
    )
except ImportError:
    LocalModelManager = None
    PredictionCache = None
    SimpleLinearModel = None
    NaiveBayesClassifier = None
    TFIDFRanker = None
    ModelType = None
    PredictionStatus = None
    CacheEvictionPolicy = None
    create_linear_model = None
    create_naive_bayes = None
    create_tfidf_ranker = None
    create_prediction_cache = None
    create_model_manager = None

# 增量学习模块 (v1.6.0 W23-W24)
try:
    from su_memory._sys._incremental_learning import (
        IncrementalLearningManager,
        FeedbackLoop,
        IncrementalUpdater,
        MemoryForgetting,
        FeedbackType,
        UpdateStrategy,
        ForgettingPolicy,
        create_feedback_loop,
        create_incremental_updater,
        create_memory_forgetting,
        create_learning_manager,
    )
except ImportError:
    IncrementalLearningManager = None
    FeedbackLoop = None
    IncrementalUpdater = None
    MemoryForgetting = None
    FeedbackType = None
    UpdateStrategy = None
    ForgettingPolicy = None
    create_feedback_loop = None
    create_incremental_updater = None
    create_memory_forgetting = None
    create_learning_manager = None

# 本地数据管理模块 (v1.7.0 W29-W30)
try:
    from su_memory.storage import (
        SQLiteBackend,
        MemoryItem,
        AutoCompressor,
        BackupManager,
        DataExporter,
    )
except ImportError:
    SQLiteBackend = None
    MemoryItem = None
    AutoCompressor = None
    BackupManager = None
    DataExporter = None

# CLI工具 (v1.7.0 W31-W32)
try:
    from su_memory.cli import cli
except ImportError:
    cli = None

# LangChain集成 (v1.7.0 W31-W32)
try:
    from su_memory.integrations.langchain import (
        SuMemoryRetriever,
        SuMemoryRetrieverConfig,
        SuMemoryLoader,
        SuMemoryTool,
        SuMemoryMemory,
        create_rag_chain,
        create_conversational_chain,
        LANGCHAIN_AVAILABLE,
    )
except ImportError:
    SuMemoryRetriever = None
    SuMemoryRetrieverConfig = None
    SuMemoryLoader = None
    SuMemoryTool = None
    SuMemoryMemory = None
    create_rag_chain = None
    create_conversational_chain = None
    LANGCHAIN_AVAILABLE = False

# LlamaIndex集成 (v1.7.0 W31-W32)
try:
    from su_memory.integrations.llamaindex import (
        SuMemoryLlamaIndexRetriever,
        SuMemoryLlamaIndexQueryEngine,
        SuMemoryLlamaIndexReader,
        SuMemoryIndex,
        SuMemoryIndexConfig,
        create_vector_index,
        create_query_engine,
        LLAMAINDEX_AVAILABLE,
    )
except ImportError:
    SuMemoryLlamaIndexRetriever = None
    SuMemoryLlamaIndexQueryEngine = None
    SuMemoryLlamaIndexReader = None
    SuMemoryIndex = None
    SuMemoryIndexConfig = None
    create_vector_index = None
    create_query_engine = None
    LLAMAINDEX_AVAILABLE = False

