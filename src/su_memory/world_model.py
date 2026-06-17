"""
世界模型统一入口 — World Model Unified API

su-memory v3.5.0 core three-layer unified architecture:

  【基础类型层 (Foundation)】
     YinYang          Duality pair
     ThreePowers       天地人三才
     FourSymbols       四象 (少阳/太阳/少阴/太阴)
     Season            五季 (春/夏/长夏/秋/冬)
     TimeStem          Ten Celestial Stems
     TimeBranch        Twelve Earthly Branches
     BranchRelation    Branch Relations (六合/三合/六冲/三刑/六害/破)
     TrigramType       Trigram Patterns类型
     TrigramRelation   semantic relations (inverse/mutual/reverse/partial/cross)
     StrengthState     旺衰五态 (旺/相/休/囚/死)
     EnergyPattern     能量格局 (制化/从旺/专旺/反局/配合)
     SemanticCategory  语义分类 (八象映射)
     EnergyNetwork     能量网络

  天层 (Temporal/Sky) — "何时"
     TemporalCore     cyclic temporal encoding core
     StemBranchCode    Time Code unit
     TemporalSystem    temporal system (date to stem-branch, time decay, similarity)
     TimeCycle         六十甲子循环
     TimeCodeInfo       时空标注

  地层 (Spatial/Earth) — "何地"
     TrigramCore       Trigram Patterns空间编码核心
     TaijiMapper       3D dimensional mapping
     PatternInference  64卦三层推断 (本→互→变)

  人层 (Energy/Human) — "何性"
     EnergyCore        energy core (strength states, pattern analysis)
     EnergyBus         跨层能量传播总线
     CategoryCausalEngine  能量加权因果推理引擎

  三才合一 (Integration)
     UnifiedInfoUnit   统一信息单元 (天+地+人)
     UnifiedInfoFactory 统一信息工厂

  元认知 (MetaCognition)
     CognitiveGap      认知空洞
     KnowledgeAging    知识老化

  检索融合 (Retrieval)
     MultiViewRetriever 五维融合检索器

  能量关系 (Relations)
     analyze_balance        Energy Types balance分析
     calculate_link_weight  能量加权链接
     analyze_relation       能量关系判定
     get_affinity_score     亲和度计算
     surface_entities       从果溯因 (MEMO Step 4)
     find_reverse_causal_chain  逆因果链搜索

Usage:
    >>> from su_memory.world_model import YinYang, FourSymbols
    >>> from su_memory.world_model import TemporalCore

    >>> tc = TemporalCore()
    >>> code = tc.encode(year=2024, month=6, day=15)
    >>> print(code.cycle_name)  # '甲午'

    >>> from su_memory.world_model import UnifiedInfoFactory
    >>> factory = UnifiedInfoFactory()
    >>> unit = factory.create_from_date(2024, 6, 15)
    >>> print(unit.temporal_stem, unit.energy_type)
"""

# ═══════════════════════════════════════════
# 基础类型层 — 枚举系统
# ═══════════════════════════════════════════
# ═══════════════════════════════════════════
# 基础类型层 — 数据字典
# ═══════════════════════════════════════════
# ═══════════════════════════════════════════
# 基础类型层 — 语义/能量分类
# ═══════════════════════════════════════════
# ═══════════════════════════════════════════
# Spatial Layer — semantic topology
# ═══════════════════════════════════════════
# ═══════════════════════════════════════════
# 人层 — 能量与因果
# ═══════════════════════════════════════════
# ═══════════════════════════════════════════
# 能量关系
# ═══════════════════════════════════════════
# ═══════════════════════════════════════════
# 三才合一
# ═══════════════════════════════════════════
# ═══════════════════════════════════════════
# 元认知
# ═══════════════════════════════════════════
# ═══════════════════════════════════════════
# 检索融合
# ═══════════════════════════════════════════
from su_memory._sys import (
    BRANCH_CHONG,
    BRANCH_CHONG_MAP,
    BRANCH_HE,
    BRANCH_HE_MAP,
    BRANCH_HIDDEN_STEM_MAP,
    BRANCH_SANHE,
    BRANCH_SANHE_MAP,
    BRANCH_XING,
    CATEGORY_ANCHORS,
    ENERGY_ENHANCE_MAP,
    ENERGY_SUPPRESS_MAP,
    ENERGY_TO_CATEGORY,
    FOUR_SYMBOLS_TO_ENERGY,
    KEYWORDS_TO_CATEGORY,
    MEMORY_TYPE_TO_CATEGORY,
    MONTH_ENERGY_STATE,
    RELATION_STRENGTH,
    SEMANTIC_CATEGORY,
    SEMANTIC_CATEGORY_NAMES,
    STATE_STRENGTH_MAP,
    STEM_CHONG,
    STEM_CHONG_MAP,
    STEM_HE,
    STEM_HE_MAP,
    STRENGTH_MULTIPLIER,
    TIME_BRANCH_ENERGY,
    TIME_BRANCHES,
    TIME_STEMS,
    TRIGRAM_BODY_MAP,
    TRIGRAM_ENERGY_MAP,
    BranchRelation,
    CategoryCausalEngine,
    CognitiveGap,
    DiZhi,
    DynamicPriority,
    EnergyBalanceResult,
    EnergyBus,
    EnergyChannel,
    EnergyCore,
    EnergyEnumRelation,
    EnergyEnumType,
    EnergyFlow,
    EnergyLayer,
    EnergyMemoryNode,
    EnergyNetwork,
    EnergyNode,
    EnergyPattern,
    EnergyRelation,
    EnergyRelationsType,
    EnergySignal,
    EnergyStateInfo,
    FourSymbols,
    KnowledgeAging,
    MemoryNodeEnergy,
    MultiViewRetriever,
    PatternInference,
    PropagationConfig,
    RelationType,
    Season,
    SemanticCategory,
    StemBranchCode,
    StrengthState,
    TaijiMapper,
    TemporalCore,
    TemporalInfo,
    TemporalSystem,
    ThreePowers,
    TianGan,
    TimeBranch,
    TimeCodeInfo,
    TimeCycle,
    TimeStem,
    TrigramCore,
    TrigramRelation,
    TrigramType,
    UnifiedInfoFactory,
    UnifiedInfoUnit,
    YinYang,
    analyze_balance,
    analyze_relation,
    calculate_link_weight,
    check_state_interaction,
    create_complete_energy_network,
    create_energy_bus,
    create_stem_branch,
    create_time_code,
    create_unified_unit,
    energy_from_category,
    energy_similarity,
    find_reverse_causal_chain,
    get_affinity_score,
    get_branch,
    get_cycle,
    get_cycle_name,
    get_cycle_sequence,
    get_enhance_relation,
    get_enhanced_energy,
    get_enhancing_energy,
    get_seasonal_energy_state,
    get_stem,
    get_suppress_chain,
    get_suppress_relation,
    get_suppressed_energy,
    get_suppressing_energy,
    is_enhancing,
    is_suppressing,
    surface_entities,
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
    "STEM_HE_MAP",
    "STEM_CHONG_MAP",
    "BRANCH_HE_MAP",
    "BRANCH_CHONG_MAP",
    "BRANCH_SANHE_MAP",
    "BRANCH_HIDDEN_STEM_MAP",
    "STRENGTH_MULTIPLIER",
    "MONTH_ENERGY_STATE",
    "TIME_STEMS",
    "TIME_BRANCHES",
    "TIME_BRANCH_ENERGY",
    "TRIGRAM_ENERGY_MAP",
    "TRIGRAM_BODY_MAP",
    # 基础类型层 — 语义/能量分类
    "SemanticCategory",
    "MEMORY_TYPE_TO_CATEGORY",
    "CATEGORY_ANCHORS",
    "KEYWORDS_TO_CATEGORY",
    "ENERGY_TO_CATEGORY",
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
    # Spatial Layer — semantic topology
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
]
