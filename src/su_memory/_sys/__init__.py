"""su-memory 内部系统模块 — internal core unified API

v3.5.0 internal core architecture:

【基础类型层 (Foundation Types)】— 被 8+ 核心模块依赖:
- _enums:           完整枚举类型体系 (YinYang, ThreePowers, FourSymbols, TimeStem,
                     TimeBranch, TrigramType, EnergyType, StrengthState, EnergyPattern...)
- _terms:           术语数据字典 (affinity mapping, embedded elements, triple affinity, seasonal strength...)
- _c1:              语义分类映射 (SemanticCategory, MEMORY_TYPE_TO_CATEGORY)
- _c2:              能量网络类型 (EnergyType, EnergyNetwork, EnergyState)

Temporal Layer (Sky) — spatiotemporal modeling:
- _temporal_core:   cyclic encoding core (StemBranchCode, TemporalCore)
- chrono:           时间系统 (TemporalSystem — date to stem-branch、时间衰减、相似度)
- _time_code:       独立时空量化系统 (TimeCycle, TimeCodeInfo)

Spatial Layer (Earth) — semantic topology:
- _category_core:   Trigram Core (TrigramCore — binary mapping, trinary structure, quaternary attribution)
- _dimension_map:   dimensional mapping (TaijiMapper — innate/acquired 3D mapping)
- _pattern_inference: pattern inference (PatternInference — 64-state triple inference)

Energy Layer (Human) — energy and causality:
- _energy_core:     energy core (EnergyCore — strength states, pattern analysis)
- _energy_bus:      energy bus (EnergyBus, EnergyNode — 3-layer propagation network)
- _energy_relations: energy relations (affinity analysis, compatibility, quaternary mapping、MEMO Step 4)
- _causal_engine:   因果引擎 (CategoryCausalEngine — 能量加权因果推理)

三才合一 (Integration):
- _unified_unit:    统一信息单元 (UnifiedInfoUnit, UnifiedInfoFactory)

元认知 (MetaCognition):
- awareness:        认知空洞发现 (CognitiveGap, KnowledgeAging)

检索融合 (Retrieval):
- fusion:           五维融合检索 (MultiViewRetriever)

贝叶斯推理子系统:
- bayesian:           贝叶斯推理核心引擎
- bayesian_network:   贝叶斯网络/概率图模型
- evidence:           证据收集与似然计算
- bayesian_reasoning: 统一集成入口
- states:             信念演化追踪 (+贝叶斯版)

v3.0.0 存储后端:
- _storage_backend:   存储后端抽象层 (StorageBackend ABC)
- _sqlite_storage:    SQLite 后端 (默认)
- _pg_storage:        PostgreSQL + pgvector 后端
- _redis_storage:     Redis 后端
"""

# ═══════════════════════════════════════════
# 基础类型层 — Foundation Enum System
# ═══════════════════════════════════════════
# ═══════════════════════════════════════════
# 基础类型层 — 语义/能量分类
# ═══════════════════════════════════════════
from ._c1 import (
    CATEGORY_ANCHORS,
    ENERGY_TO_CATEGORY,
    KEYWORDS_TO_CATEGORY,
    MEMORY_TYPE_TO_CATEGORY,
    SemanticCategory,
)
from ._c2 import (
    ENERGY_ENHANCE_MAP,
    ENERGY_SUPPRESS_MAP,
    STATE_STRENGTH_MAP,
    EnergyNetwork,
    check_state_interaction,
    energy_from_category,
    energy_similarity,
)
from ._c2 import (
    EnergyState as C2EnergyState,
)
from ._c2 import (
    EnergyType as C2EnergyType,
)
from ._c2 import (
    get_energy_state as get_seasonal_energy_state,
)

# ═══════════════════════════════════════════
# 地层 — semantic topology
# ═══════════════════════════════════════════
from ._category_core import TrigramCore
from ._causal_engine import (
    CategoryCausalEngine,
    EnergyMemoryNode,
)
from ._dimension_map import TaijiMapper
from ._energy_bus import (
    EnergyBus,
    EnergyChannel,
    EnergyLayer,
    EnergyNode,
    EnergySignal,
    PropagationConfig,
    create_complete_energy_network,
    create_energy_bus,
)

# ═══════════════════════════════════════════
# 人层 — 能量与因果
# ═══════════════════════════════════════════
from ._energy_core import (
    EnergyBalanceResult,
    EnergyCore,
    EnergyFlow,
)
from ._energy_core import (
    EnergyState as EnergyStateInfo,
)
from ._energy_relations import (
    FOUR_SYMBOLS_TO_ENERGY,
    RELATION_STRENGTH,
    EnergyRelation,
    MemoryNodeEnergy,
    RelationType,
    analyze_balance,
    analyze_relation,
    calculate_link_weight,
    find_reverse_causal_chain,
    get_affinity_score,
    get_cycle_sequence,
    get_enhance_relation,
    get_enhanced_energy,
    get_enhancing_energy,
    get_suppress_chain,
    get_suppress_relation,
    get_suppressed_energy,
    get_suppressing_energy,
    is_enhancing,
    is_suppressing,
    surface_entities,
)
from ._energy_relations import (
    EnergyType as EnergyRelationsType,
)
from ._enums import (
    BranchRelation,
    EnergyPattern,
    # Four Symbols
    FourSymbols,
    Season,
    StrengthState,
    ThreePowers,
    TimeBranch,
    # Temporal
    TimeStem,
    TrigramRelation,
    # Spatial
    TrigramType,
    # Duality
    YinYang,
)
from ._enums import (
    EnergyRelation as EnergyEnumRelation,
)
from ._enums import (
    # Energy
    EnergyType as EnergyEnumType,
)
from ._pattern_inference import PatternInference
from ._pg_storage import PgStorageBackend
from ._redis_storage import RedisStorageBackend
from ._sqlite_storage import SqliteStorageBackend

# ═══════════════════════════════════════════
# v3.0.0 存储后端
# ═══════════════════════════════════════════
from ._storage_backend import (
    BackendHealth,
    BackendType,
    StorageBackend,
    StorageConfig,
    StorageMemory,
    create_backend,
)

# ═══════════════════════════════════════════
# 天层 — 时空建模
# ═══════════════════════════════════════════
from ._temporal_core import (
    StemBranchCode,
    TemporalCore,
    create_stem_branch,
    get_cycle_name,
)

# ═══════════════════════════════════════════
# 基础类型层 — 数据字典
# ═══════════════════════════════════════════
from ._terms import (
    BRANCH_CHONG_MAP,
    BRANCH_HE_MAP,
    BRANCH_HIDDEN_STEM_MAP,
    BRANCH_SANHE_MAP,
    MONTH_ENERGY_STATE,
    SEMANTIC_CATEGORY,
    SEMANTIC_CATEGORY_NAMES,
    STEM_CHONG_MAP,
    STEM_HE_MAP,
    STRENGTH_MULTIPLIER,
    STRENGTH_STATE,
    TIME_BRANCH_ENERGY,
    TIME_BRANCHES,
    TIME_STEMS,
    TRIGRAM_BODY_MAP,
    TRIGRAM_ENERGY_MAP,
)
from ._terms import (
    ENERGY_ENHANCE as TERMS_ENERGY_ENHANCE,
)
from ._terms import (
    ENERGY_SUPPRESS as TERMS_ENERGY_SUPPRESS,
)
from ._time_code import (
    BRANCH_CHONG,
    BRANCH_HE,
    BRANCH_SANHE,
    BRANCH_XING,
    STEM_CHONG,
    STEM_HE,
    TimeCodeInfo,
    TimeCycle,
    create_time_code,
    get_branch,
    get_cycle,
    get_stem,
)

# ═══════════════════════════════════════════
# 三才合一
# ═══════════════════════════════════════════
from ._unified_unit import (
    UnifiedInfoFactory,
    UnifiedInfoUnit,
    create_unified_unit,
)

# ═══════════════════════════════════════════
# 元认知
# ═══════════════════════════════════════════
from .awareness import (
    CognitiveGap,
    KnowledgeAging,
)

# ═══════════════════════════════════════════
# 贝叶斯推理子系统
# ═══════════════════════════════════════════
from .bayesian import (
    BayesianBelief,
    BayesianEngine,
    BetaDistribution,
    LikelihoodFunctions,
)
from .bayesian_network import (
    BayesianNetwork,
    BeliefPropagator,
    NetworkNode,
    ProbabilisticEdge,
)
from .bayesian_reasoning import (
    BayesianAdvisor,
    BayesianPredictor,
    BayesianReasoningSystem,
)
from .chrono import (
    DiZhi,
    DynamicPriority,
    TemporalInfo,
    TemporalSystem,
    TianGan,
)
from .evidence import (
    EvidenceCollector,
    EvidenceRecord,
    SourceProfile,
)

# ═══════════════════════════════════════════
# 检索融合
# ═══════════════════════════════════════════
from .fusion import MultiViewRetriever
from .states import (
    BayesianBeliefState,
    BayesianBeliefTracker,
    BeliefStage,
    BeliefState,
    BeliefTracker,
)

__all__ = [
    # 基础类型层 — 枚举系统
    "YinYang",
    "ThreePowers",
    "FourSymbols",
    "Season",
    "TimeStem",
    "TimeBranch",
    "BranchRelation",
    "TrigramType",
    "TrigramRelation",
    "EnergyEnumType",
    "EnergyEnumRelation",
    "StrengthState",
    "EnergyPattern",
    # 基础类型层 — 数据字典
    "SEMANTIC_CATEGORY",
    "SEMANTIC_CATEGORY_NAMES",
    "TERMS_ENERGY_ENHANCE",
    "TERMS_ENERGY_SUPPRESS",
    "STRENGTH_STATE",
    "MONTH_ENERGY_STATE",
    "STRENGTH_MULTIPLIER",
    "TIME_STEMS",
    "TIME_BRANCHES",
    "TIME_BRANCH_ENERGY",
    "STEM_HE_MAP",
    "STEM_CHONG_MAP",
    "BRANCH_HE_MAP",
    "BRANCH_CHONG_MAP",
    "BRANCH_SANHE_MAP",
    "BRANCH_HIDDEN_STEM_MAP",
    "TRIGRAM_ENERGY_MAP",
    "TRIGRAM_BODY_MAP",
    # 基础类型层 — 语义/能量分类
    "SemanticCategory",
    "MEMORY_TYPE_TO_CATEGORY",
    "CATEGORY_ANCHORS",
    "KEYWORDS_TO_CATEGORY",
    "ENERGY_TO_CATEGORY",
    "C2EnergyType",
    "C2EnergyState",
    "EnergyNetwork",
    "ENERGY_ENHANCE_MAP",
    "ENERGY_SUPPRESS_MAP",
    "STATE_STRENGTH_MAP",
    "get_seasonal_energy_state",
    "check_state_interaction",
    "energy_similarity",
    "energy_from_category",
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
    # 地层 — semantic topology
    "TrigramCore",
    "TaijiMapper",
    "PatternInference",
    # 人层 — 能量与因果
    "EnergyCore",
    "EnergyStateInfo",
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
    # 人层 — 能量关系函数
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
    "EnergyRelationsType",
    "MemoryNodeEnergy",
    "RELATION_STRENGTH",
    "FOUR_SYMBOLS_TO_ENERGY",
    # 三才合一
    "UnifiedInfoUnit",
    "UnifiedInfoFactory",
    "create_unified_unit",
    # 元认知
    "CognitiveGap",
    "KnowledgeAging",
    # 检索融合
    "MultiViewRetriever",
    # 贝叶斯引擎
    "BayesianEngine",
    "BetaDistribution",
    "BayesianBelief",
    "LikelihoodFunctions",
    # 贝叶斯网络
    "BayesianNetwork",
    "BeliefPropagator",
    "ProbabilisticEdge",
    "NetworkNode",
    # 证据收集
    "EvidenceCollector",
    "EvidenceRecord",
    "SourceProfile",
    # 统一推理
    "BayesianReasoningSystem",
    "BayesianPredictor",
    "BayesianAdvisor",
    # 信念追踪
    "BeliefTracker",
    "BayesianBeliefTracker",
    "BayesianBeliefState",
    "BeliefState",
    "BeliefStage",
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
    # v3.5.5 P1-3: Memory Lifecycle Manager
    "MemoryLifecycleManager",
    "LifecycleReport",
    "LifecycleAction",
]

# v3.5.5 P1-3: Memory Lifecycle Manager (lazy import)
try:
    from su_memory._sys._lifecycle_manager import (
        LifecycleAction,
        LifecycleReport,
        MemoryLifecycleManager,
    )
except ImportError:
    MemoryLifecycleManager = None
    LifecycleReport = None
    LifecycleAction = None
