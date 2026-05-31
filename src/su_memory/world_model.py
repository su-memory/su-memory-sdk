"""
世界模型统一入口 — World Model Unified API

su-memory v3.5.0 能量中心三才合一架构:

  【基础类型层 (Foundation)】
     YinYang          阴阳二象
     ThreePowers       天地人三才
     FourSymbols       四象 (少阳/太阳/少阴/太阴)
     Season            五季 (春/夏/长夏/秋/冬)
     TimeStem          十天干
     TimeBranch        十二地支
     BranchRelation    地支关系 (六合/三合/六冲/三刑/六害/破)
     TrigramType       八卦类型
     TrigramRelation   卦象关系 (错/互/综/半/交)
     StrengthState     旺衰五态 (旺/相/休/囚/死)
     EnergyPattern     能量格局 (制化/从旺/专旺/反局/配合)
     SemanticCategory  语义分类 (八象映射)
     EnergyNetwork     能量网络

  天层 (Temporal/Sky) — "何时"
     TemporalCore     六十花甲时间编码核心
     StemBranchCode    天干地支编码单元
     TemporalSystem    时间系统 (日期→干支、时间衰减、相似度)
     TimeCycle         六十甲子循环
     TimeCodeInfo       时空标注

  地层 (Spatial/Earth) — "何地"
     TrigramCore       八卦空间编码核心
     TaijiMapper       太极3D维度映射
     PatternInference  64卦三层推断 (本→互→变)

  人层 (Energy/Human) — "何性"
     EnergyCore        五行能量核心 (旺相休囚死、格局分析)
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
     analyze_balance        五行均衡分析
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
from su_memory._sys import (
    YinYang,
    ThreePowers,
    FourSymbols,
    Season,
    TimeStem,
    TimeBranch,
    BranchRelation,
    TrigramType,
    TrigramRelation,
    EnergyEnumType,
    EnergyEnumRelation,
    StrengthState,
    EnergyPattern,
)

# ═══════════════════════════════════════════
# 基础类型层 — 数据字典
# ═══════════════════════════════════════════
from su_memory._sys import (
    SEMANTIC_CATEGORY,
    SEMANTIC_CATEGORY_NAMES,
    STEM_HE_MAP,
    STEM_CHONG_MAP,
    BRANCH_HE_MAP,
    BRANCH_CHONG_MAP,
    BRANCH_SANHE_MAP,
    BRANCH_HIDDEN_STEM_MAP,
    STRENGTH_MULTIPLIER,
    MONTH_ENERGY_STATE,
    TIME_STEMS,
    TIME_BRANCHES,
    TIME_BRANCH_ENERGY,
    TRIGRAM_ENERGY_MAP,
    TRIGRAM_BODY_MAP,
)

# ═══════════════════════════════════════════
# 基础类型层 — 语义/能量分类
# ═══════════════════════════════════════════
from su_memory._sys import (
    SemanticCategory,
    MEMORY_TYPE_TO_CATEGORY,
    CATEGORY_ANCHORS,
    KEYWORDS_TO_CATEGORY,
    ENERGY_TO_CATEGORY,
    EnergyNetwork,
    ENERGY_ENHANCE_MAP,
    ENERGY_SUPPRESS_MAP,
    STATE_STRENGTH_MAP,
    get_seasonal_energy_state,
    check_state_interaction,
    energy_similarity,
    energy_from_category,
)
from su_memory._sys import (
    TemporalCore,
    StemBranchCode,
    create_stem_branch,
    get_cycle_name,
    TianGan,
    DiZhi,
    TemporalSystem,
    TemporalInfo,
    DynamicPriority,
    TimeCycle,
    TimeCodeInfo,
    create_time_code,
    STEM_HE,
    STEM_CHONG,
    BRANCH_HE,
    BRANCH_SANHE,
    BRANCH_CHONG,
    BRANCH_XING,
    get_stem,
    get_branch,
    get_cycle,
)

# ═══════════════════════════════════════════
# 地层 — 卦象空间
# ═══════════════════════════════════════════
from su_memory._sys import (
    TrigramCore,
    TaijiMapper,
    PatternInference,
)

# ═══════════════════════════════════════════
# 人层 — 能量与因果
# ═══════════════════════════════════════════
from su_memory._sys import (
    EnergyCore,
    EnergyStateInfo,
    EnergyBalanceResult,
    EnergyFlow,
    EnergyBus,
    EnergyNode,
    EnergyChannel,
    EnergySignal,
    EnergyLayer,
    PropagationConfig,
    create_energy_bus,
    create_complete_energy_network,
    CategoryCausalEngine,
    EnergyMemoryNode,
)

# ═══════════════════════════════════════════
# 能量关系
# ═══════════════════════════════════════════
from su_memory._sys import (
    analyze_balance,
    calculate_link_weight,
    analyze_relation,
    get_affinity_score,
    surface_entities,
    find_reverse_causal_chain,
    is_enhancing,
    is_suppressing,
    get_enhanced_energy,
    get_cycle_sequence,
    get_enhance_relation,
    get_suppress_relation,
    get_suppress_chain,
    get_enhancing_energy,
    get_suppressed_energy,
    get_suppressing_energy,
    RelationType,
    EnergyRelation,
    EnergyRelationsType,
    MemoryNodeEnergy,
    RELATION_STRENGTH,
    FOUR_SYMBOLS_TO_ENERGY,
)

# ═══════════════════════════════════════════
# 三才合一
# ═══════════════════════════════════════════
from su_memory._sys import (
    UnifiedInfoUnit,
    UnifiedInfoFactory,
    create_unified_unit,
)

# ═══════════════════════════════════════════
# 元认知
# ═══════════════════════════════════════════
from su_memory._sys import (
    CognitiveGap,
    KnowledgeAging,
)

# ═══════════════════════════════════════════
# 检索融合
# ═══════════════════════════════════════════
from su_memory._sys import MultiViewRetriever

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
    # 地层 — 卦象空间
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
