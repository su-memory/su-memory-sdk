"""
su-memory SDK - 对外赋能核心模块

此模块提供统一的SDK接口，支持:
- Python SDK (SuMemoryClient)
- 轻量级SDK (SuMemoryLite)
- 增强版SDK (SuMemoryLitePro) - 全面超越Hindsight v4.7/5
- 预测模块 (PredictionModule)
- 可解释性模块 (ExplainabilityModule)
- 贝叶斯增强器 (BayesianAugmenter) — 串联验证
- LangChain适配器
"""

from su_memory.sdk._memory_protocol import MemoryProtocol
from su_memory.sdk.client import SuMemoryClient
from su_memory.sdk.config import SDKConfig
from su_memory.sdk.exceptions import (
    EncodingError,
    MemoryNotFoundError,
    SDKError,
    StorageError,
)
from su_memory.sdk.lite import SuMemoryLite
from su_memory.sdk.lite_pro import SuMemoryLitePro

# 贝叶斯增强器（可选）
try:
    from su_memory.sdk.bayesian_augmenter import (
        AccuracyRecord,
        BayesianAugmenter,
        ComparisonDelta,
        EnhancedOutput,
    )
except ImportError:
    BayesianAugmenter = None
    EnhancedOutput = None
    ComparisonDelta = None
    AccuracyRecord = None

# v3.4.0: 频谱因果引擎 (scipy + numpy 依赖)
try:
    from su_memory.sdk._spectral_causal import (
        BayesianCausal,
        FourierCausal,
        GaussianDAG,
        GaussianDistribution,
    )
except ImportError:
    GaussianDAG = None
    FourierCausal = None
    BayesianCausal = None
    GaussianDistribution = None

# v3.5.0: Reflection QA 合成引擎 (M5)
try:
    from su_memory.sdk._reflection_synthesizer import (
        ReflectionSynthesizer,
        SynthesizedQAPair,
    )
except ImportError:
    ReflectionSynthesizer = None
    SynthesizedQAPair = None

# v3.5.0: SIGReg 嵌入正则 + Entity Surfacing (M6)
try:
    from su_memory._sys._energy_relations import (
        find_reverse_causal_chain,
        surface_entities,
    )
    from su_memory.sdk._sigreg import SIGReg, apply_sigreg_to_index
except ImportError:
    SIGReg = None
    apply_sigreg_to_index = None
    surface_entities = None
    find_reverse_causal_chain = None

# v3.6.0: MCI World Model — 参数化记忆 + 能量一致性损失 (M7-M9)
try:
    from su_memory.sdk._energy_loss import (
        EnergyConsistencyLoss,
        TopologicalEnergyMatrix,
    )
except ImportError:
    EnergyConsistencyLoss = None
    TopologicalEnergyMatrix = None

try:
    from su_memory.sdk._parametric_memory import (
        ParametricMemory,
        ParametricMemoryConfig,
    )
except ImportError:
    ParametricMemory = None
    ParametricMemoryConfig = None

try:
    from su_memory.sdk._world_model import (
        CausalWorldModelState,
        MCIWorldModel,
    )
except ImportError:
    MCIWorldModel = None
    CausalWorldModelState = None

# v3.7.0: Pearl do-calculus 干预引擎
try:
    from su_memory.sdk._do_calculus import (
        CausalGraph,
        DoCalculus,
        InterventionResult,
    )
except ImportError:
    DoCalculus = None
    CausalGraph = None
    InterventionResult = None

# v3.8.0: Pearl counterfactual 反事实推理引擎 (L3)
try:
    from su_memory.sdk._counterfactual import (
        CounterfactualEngine,
        CounterfactualResult,
        StructuralEquationModel,
    )
except ImportError:
    CounterfactualEngine = None
    CounterfactualResult = None
    StructuralEquationModel = None

# v3.5.5: Document Ingestion Pipeline (P1-1)
try:
    from su_memory.sdk.document_pipeline import (
        ChunkStrategy,
        DocumentIngestionPipeline,
        FixedSizeChunker,
        FormatDetector,
        IngestResult,
        MarkdownHeaderChunker,
        SentenceChunker,
        get_chunker,
    )
except ImportError:
    DocumentIngestionPipeline = None
    FormatDetector = None
    ChunkStrategy = None
    FixedSizeChunker = None
    SentenceChunker = None
    MarkdownHeaderChunker = None
    IngestResult = None
    get_chunker = None

# v3.5.5: User Profile Engine (P1-2)
try:
    from su_memory.sdk.profile_engine import (
        InteractionPattern,
        UserProfile,
        UserProfileEngine,
    )
except ImportError:
    UserProfileEngine = None
    UserProfile = None
    InteractionPattern = None

__version__ = "4.4.1"

__all__ = [
    # 核心协议
    "MemoryProtocol",
    # 核心客户端
    "SuMemoryClient",
    "SuMemoryLite",
    "SuMemoryLitePro",
    "SDKConfig",
    # 贝叶斯增强
    "BayesianAugmenter",
    "EnhancedOutput",
    "ComparisonDelta",
    "AccuracyRecord",
    # v3.4.0 频谱因果
    "GaussianDAG",
    "FourierCausal",
    "BayesianCausal",
    "GaussianDistribution",
    # v3.5.0 Reflection QA + SIGReg + Entity Surfacing
    "ReflectionSynthesizer",
    "SynthesizedQAPair",
    "SIGReg",
    "apply_sigreg_to_index",
    "surface_entities",
    "find_reverse_causal_chain",
    # v3.6.0 MCI World Model — 参数化记忆 + 因果干预
    "MCIWorldModel",
    "CausalWorldModelState",
    "ParametricMemory",
    "ParametricMemoryConfig",
    "EnergyConsistencyLoss",
    "TopologicalEnergyMatrix",
    # v3.7.0 Pearl do-calculus 干预引擎
    "DoCalculus",
    "CausalGraph",
    "InterventionResult",
    # v3.8.0 Pearl counterfactual 反事实推理引擎 (L3)
    "CounterfactualEngine",
    "CounterfactualResult",
    "StructuralEquationModel",
    # v3.5.5 P1-1: Document Ingestion Pipeline
    "DocumentIngestionPipeline",
    "FormatDetector",
    "ChunkStrategy",
    "FixedSizeChunker",
    "SentenceChunker",
    "MarkdownHeaderChunker",
    "IngestResult",
    "get_chunker",
    # v3.5.5 P1-2: User Profile Engine
    "UserProfileEngine",
    "UserProfile",
    "InteractionPattern",
    # v3.0.0 存储后端
    "StorageBackend",
    "StorageConfig",
    "StorageMemory",
    "BackendType",
    "BackendHealth",
    "create_backend",
    "SqliteStorageBackend",
    "PgStorageBackend",
    "RedisStorageBackend",
    # 异常
    "SDKError",
    "MemoryNotFoundError",
    "EncodingError",
    "StorageError",
]

# v3.0.0: 存储后端 (lazy import, 避免强制依赖)
try:
    from su_memory._sys._sqlite_storage import SqliteStorageBackend
    from su_memory._sys._storage_backend import (
        BackendHealth,
        BackendType,
        StorageBackend,
        StorageConfig,
        StorageMemory,
        create_backend,
    )
except ImportError:
    StorageBackend = None
    StorageConfig = None
    StorageMemory = None
    BackendType = None
    BackendHealth = None
    create_backend = None
    SqliteStorageBackend = None

# PgStorageBackend / RedisStorageBackend (可选依赖: asyncpg / redis)
try:
    from su_memory._sys._pg_storage import PgStorageBackend
except ImportError:
    PgStorageBackend = None

try:
    from su_memory._sys._redis_storage import RedisStorageBackend
except ImportError:
    RedisStorageBackend = None
