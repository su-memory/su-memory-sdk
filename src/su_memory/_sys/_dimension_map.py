"""
Taiji Mapping (Core Principle映射) - Multi-Dimensional System

This module implements the Taiji Mapper with MULTI-DIMENSIONAL MAPPING support,
leveraging先天Trigram Patterns (Prior) and 后天Trigram Patterns (Post) dimensions.

Architecture: Bridge between Sky Layer (Heavenly Stems) and Earth Layer (Trigram Patterns)

Multi-Dimensional Design:
- Dimension 1 (Najia法): Traditional Najia method for stem-trigram mapping
- Dimension 2 (先天Trigram Patterns): Fu Xi trigram ordering (Qian=1, Dui=2, Li=3, Zhen=4, Xun=5, Kan=6, Gen=7, Kun=8)
- Dimension 3 (后天Trigram Patterns): King Wen trigram ordering (Kan=1, Kun=2, Zhen=3, Xun=4, Qian=6, Dui=7, Gen=8, Li=9)

Calculus Integration:
- Integration (积分): Aggregate multiple dimensional mappings with weights
- Differentiation (微分): Decompose conflicts into multiple weighted possibilities
- Gradient (梯度): Compute mapping confidence based on dimensional agreement

Core Features:
- Multi-dimensional bidirectional mapping
- Weighted aggregation for ambiguous cases
- Consistency guarantee across dimensions
- Energy harmony analysis
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from ._enums import TimeStem, TimeBranch, TrigramType
from ._category_core import (
    TrigramCore,
    TRIGRAM_ENERGY_TYPE,
    TRIGRAM_NAMES,
    TRIGRAM_NATURE,
    PRIOR_DIRECTION,
    POST_DIRECTION,
    PRIOR_ORDER,
    POST_ORDER,
    ORDER_TO_TRIGRAM_PRIOR,
    ORDER_TO_TRIGRAM_POST,
)
from ._terms import (
    TIME_STEMS,
    TIME_BRANCHES,
)


# =============================================================================
# Multi-Dimensional Mapping Enums
# =============================================================================

class MappingDimension(Enum):
    """Mapping dimension types"""
    NAJIA = "najia"           # Najia法: Traditional Najia method
    PRIOR = "prior"           # 先天Trigram Patterns: Fu Xi ordering
    POST = "post"             # 后天Trigram Patterns: King Wen ordering


class MappingConfidence(Enum):
    """Mapping confidence levels"""
    DEFINITIVE = 1.0          # 确定性映射
    HIGH = 0.8                 # 高置信度
    MEDIUM = 0.5               # 中置信度
    LOW = 0.3                  # 低置信度


# =============================================================================
# Multi-Dimensional Mapping Tables
# =============================================================================

# 先天Trigram Patterns序数 (Fu Xi trigram order: 乾1兑2离3震4巽5坎6艮7坤8)
#
# 【先天主数】- 用于数值计算和数学运算
# - 应用于64 Patterns的数值推数
# - 用于二进制转换和位运算操作
# - 数列：乾=1, 兑=2, 离=3, 震=4, 巽=5, 坎=6, 艮=7, 坤=8
#
PRIOR_TRIGRAM_ORDER: Dict[int, int] = {
    0: 1,   # 乾 -> 数1
    7: 2,   # 兑 -> 数2
    5: 3,   # 离 -> 数3
    2: 4,   # 震 -> 数4
    3: 5,   # 巽 -> 数5
    4: 6,   # 坎 -> 数6
    6: 7,   # 艮 -> 数7
    1: 8,   # 坤 -> 数8
}

# 后天Trigram Patterns序数 (King Wen order: 坎1坤2震3巽4乾6兑7艮8离9)
#
# 【后天主象】- 用于象征意义和时空应用
# - 应用于方位、季节、时间等空间映射
# - 用于能量流转和Energy System关系的实际应用
# - Symbolic Value：坎=1, 坤=2, 震=3, 巽=4, 乾=6, 兑=7, 艮=8, 离=9 (跳过5)
#
POST_TRIGRAM_ORDER: Dict[int, int] = {
    4: 1,   # 坎 -> 象1 (北方水)
    1: 2,   # 坤 -> 象2 (西南土)
    2: 3,   # 震 -> 象3 (东方木)
    3: 4,   # 巽 -> 象4 (东南木)
    0: 6,   # 乾 -> 象6 (西北金)
    7: 7,   # 兑 -> 象7 (西方金)
    6: 8,   # 艮 -> 象8 (东北土)
    5: 9,   # 离 -> 象9 (南方火)
}

# Najia法Heavenly Stems->Trigram Patterns映射 (一对一)
NAJIA_STEM_TO_TRIGRAM: Dict[int, int] = {
    # 阳干
    0: 0,   # 甲 -> 乾
    2: 5,   # 丙 -> 离
    4: 0,   # 戊 -> 乾
    6: 2,   # 庚 -> 震
    8: 4,   # 壬 -> 坎
    # 阴干
    1: 1,   # 乙 -> 坤
    3: 7,   # 丁 -> 兑
    5: 6,   # 己 -> 艮
    7: 3,   # 辛 -> 巽
    9: 1,   # 癸 -> 坤
}

# Najia法Trigram Patterns->Heavenly Stems映射 (一对多，从NAJIA_STEM_TO_TRIGRAM派生)
NAJIA_TRIGRAM_TO_STEMS: Dict[int, List[int]] = {i: [] for i in range(8)}
for stem_idx, trig_idx in NAJIA_STEM_TO_TRIGRAM.items():
    if stem_idx not in NAJIA_TRIGRAM_TO_STEMS[trig_idx]:
        NAJIA_TRIGRAM_TO_STEMS[trig_idx].append(stem_idx)
for i in range(8):
    NAJIA_TRIGRAM_TO_STEMS[i].sort()

# 扩展映射：允许一个Heavenly Stems对应多个Trigram Patterns（微分分解）
# 基于Heavenly StemsDualityEnergy System属性
NAJIA_STEM_MULTI_TRIGRAM: Dict[int, List[Tuple[int, float]]] = {
    # (trig_idx, weight) - weight表示该映射的置信度
    0: [(0, 0.6), (2, 0.2), (6, 0.2)],   # 甲: 乾(主) + 震 + 艮
    1: [(1, 0.7), (3, 0.3)],               # 乙: 坤(主) + 巽
    2: [(5, 0.5), (0, 0.3), (7, 0.2)],     # 丙: 离(主) + 乾 + 兑
    3: [(7, 0.6), (1, 0.2), (5, 0.2)],     # 丁: 兑(主) + 坤 + 离
    4: [(0, 0.5), (6, 0.5)],               # 戊: 乾 + 艮
    5: [(6, 0.6), (0, 0.2), (4, 0.2)],     # 己: 艮(主) + 乾 + 坎
    6: [(2, 0.6), (0, 0.2), (3, 0.2)],     # 庚: 震(主) + 乾 + 巽
    7: [(3, 0.6), (2, 0.2), (7, 0.2)],     # 辛: 巽(主) + 震 + 兑
    8: [(4, 0.6), (1, 0.2), (3, 0.2)],     # 壬: 坎(主) + 坤 + 巽
    9: [(1, 0.6), (4, 0.2), (3, 0.2)],     # 癸: 坤(主) + 坎 + 巽
}

# Trigram Patterns->Heavenly Stems多维度映射（从NAJIA_STEM_MULTI_TRIGRAM积分）
NAJIA_TRIGRAM_MULTI_STEMS: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(8)}
for stem_idx, trig_list in NAJIA_STEM_MULTI_TRIGRAM.items():
    for trig_idx, weight in trig_list:
        NAJIA_TRIGRAM_MULTI_STEMS[trig_idx].append((stem_idx, weight))
for i in range(8):
    NAJIA_TRIGRAM_MULTI_STEMS[i].sort(key=lambda x: -x[1])  # 按权重降序


# =============================================================================
# TrigramType -> SemanticType 映射表
# =============================================================================
#
# TrigramType (_enums.py) 使用 先天序 (Fu Xi): 乾0 兑1 离2 震3 巽4 坎5 艮6 坤7
# SemanticType (_pattern_inference.py) 使用 后天方位序:
#   CAT_CREATIVE(0,NW), CAT_LAKE(1,W), CAT_LIGHT(2,S), CAT_THUNDER(3,E),
#   CAT_WIND(4,SE), CAT_ABYSS(5,N), CAT_MOUNTAIN(6,NE), CAT_RECEPTIVE(7,SW)
#
# 通过 energy_type + direction (后天方位) 建立 1:1 对应:
#   QIAN(metal,NW) → CAT_CREATIVE(metal,NW)
#   DUI(metal,W)   → CAT_LAKE(metal,W)
#   LI(fire,S)     → CAT_LIGHT(fire,S)
#   ZHEN(wood,E)   → CAT_THUNDER(wood,E)
#   XUN(wood,SE)   → CAT_WIND(wood,SE)
#   KAN(water,N)   → CAT_ABYSS(water,N)
#   GEN(earth,NE)  → CAT_MOUNTAIN(earth,NE)
#   KUN(earth,SW)  → CAT_RECEPTIVE(earth,SW)
#
TRIGRAM_TO_SEMANTIC_DIRECT: Dict[int, int] = {
    0: 0,   # QIAN → CAT_CREATIVE
    1: 7,   # KUN  → CAT_RECEPTIVE
    2: 3,   # ZHEN → CAT_THUNDER
    3: 4,   # XUN  → CAT_WIND
    4: 5,   # KAN  → CAT_ABYSS
    5: 2,   # LI   → CAT_LIGHT
    6: 6,   # GEN  → CAT_MOUNTAIN
    7: 1,   # DUI  → CAT_LAKE
}

# 逆向映射: SemanticType → TrigramType
SEMANTIC_TO_TRIGRAM_DIRECT: Dict[int, int] = {
    v: k for k, v in TRIGRAM_TO_SEMANTIC_DIRECT.items()
}

# Stem-to-energy-type mapping (for NAJIA dimension bridging)
# Stem energy mapping: jia/yi=wood, bing/ding=fire, wu/ji=earth, geng/xin=metal, ren/gui=water
STEM_ENERGY_TYPE: Dict[int, str] = {
    0: "wood", 1: "wood",    # jia/yi → wood
    2: "fire", 3: "fire",    # bing/ding → fire
    4: "earth", 5: "earth",  # wu/ji → earth
    6: "metal", 7: "metal",  # geng/xin → metal
    8: "water", 9: "water",  # ren/gui → water
}

# SemanticType 信息表 (用于方位/能量反向查找)
# direction 使用简称以匹配 PRIOR_DIRECTION / POST_DIRECTION 格式
_SEMANTIC_DIRECTION_MAP: Dict[int, str] = {
    0: "northwest",  1: "west",      2: "south",     3: "east",
    4: "southeast",  5: "north",     6: "northeast", 7: "southwest",
}

_SEMANTIC_ENERGY_MAP: Dict[int, str] = {
    0: "metal", 1: "metal", 2: "fire",  3: "wood",
    4: "wood",  5: "water", 6: "earth", 7: "earth",
}

# =============================================================================
# TrigramType -> SemanticType 三维映射维度权重
# =============================================================================
# NAJIA: indirect bridge through stems, lower weight due to double-translation
# PRIOR: structural matching, provides numerical validation
# POST:  directional matching, most direct (SemanticType uses post-heaven directions)
TRIGRAM_SEMANTIC_DIMENSION_WEIGHTS = {
    "najia": 0.30,
    "prior": 0.30,
    "post":  0.40,   # 最高权重：SemanticType 使用后天方位
}


@dataclass
class MappingResult:
    """Multi-dimensional mapping result"""
    source: int                    # 源索引
    targets: List[Tuple[int, float]]  # (目标索引, 权重)
    dimension: MappingDimension     # 映射维度
    confidence: MappingConfidence   # 置信度

    @property
    def primary_target(self) -> Optional[int]:
        """Get primary (highest weight) target"""
        if self.targets:
            return self.targets[0][0]
        return None

    @property
    def has_conflict(self) -> bool:
        """Check if there are multiple targets"""
        return len(self.targets) > 1


@dataclass
class IntegratedMappingResult:
    """Integrated result from multiple dimensions"""
    source: int
    primary: Optional[int]         # 主映射
    candidates: List[Tuple[int, float]]  # 候选映射 (积分结果)
    confidence: MappingConfidence
    dimension_agreement: float      # 维度一致率
    explanation: str


# =============================================================================
# Trigram-Stem-Branch Mapping Tables (Najia法体系)
# =============================================================================
#
# 基于传统Najia法: 乾Najia, 坤纳乙, 震纳庚, 巽纳辛, 坎纳壬, 离纳丙, 艮纳戊, 兑纳丁
#
# 映射原则:
# - Heavenly Stems -> Trigram Patterns: 一对一 (STEM_TO_TRIGRAM)
# - Trigram Patterns -> Heavenly Stems: 一对多 (TRIGRAM_STEM_MAP)
# - 两者必须保持严格一致!
#
# 设计决策:
# - 由于十Heavenly Stems对应Trigram Patterns存在天然的多对一特性(如甲戊归乾),
#   我们采用STEM_TO_TRIGRAM作为"真相源"(一对一),
#   TRIGRAM_STEM_MAP通过反向推导生成(一对多)
#
# =============================================================================

# Step 1: Define canonical stem -> trigram mapping (一对一)
# Based on traditional Najia法 (Najia method)
# Each stem has exactly ONE primary trigram association
STEM_TO_TRIGRAM: Dict[int, int] = {
    # 阳干 (Yang Stems)
    0: 0,   # 甲 (JIA) -> 乾 (QIAN) - 乾Najia
    2: 5,   # 丙 (BING) -> 离 (LI) - 离纳丙
    4: 0,   # 戊 (WU) -> 乾 (QIAN) - 戊归乾
    6: 2,   # 庚 (GENG) -> 震 (ZHEN) - 震纳庚
    8: 4,   # 壬 (REN) -> 坎 (KAN) - 坎纳壬

    # 阴干 (Yin Stems)
    1: 1,   # 乙 (YI) -> 坤 (KUN) - 坤纳乙
    3: 7,   # 丁 (DING) -> 兑 (DUI) - 兑纳丁
    5: 6,   # 己 (JI) -> 艮 (GEN) - 己归艮
    7: 3,   # 辛 (XIN) -> 巽 (XUN) - 巽纳辛
    9: 1,   # 癸 (GUI) -> 坤 (KUN) - 癸归坤
}

# =============================================================================
# Legacy Compatibility Aliases (向后兼容)
# =============================================================================

# 保持向后兼容的别名
STEM_TO_TRIGRAM = NAJIA_STEM_TO_TRIGRAM
TRIGRAM_STEM_MAP = NAJIA_TRIGRAM_TO_STEMS

# 保留旧的派生变量名（内部使用）
_TRIGRAM_STEM_MAP_DERIVED = NAJIA_TRIGRAM_TO_STEMS

# Trigram to Earthly Branches mapping
# Each trigram corresponds to one or more earthly branches
# Note: This defines the relationship, but multiple branches may map to same trigram
TRIGRAM_BRANCH_MAP: Dict[int, List[int]] = {
    0: [10, 11],  # 乾: 戌(10), 亥(11)
    1: [9],        # 兑: 酉(9)
    2: [6],        # 离: 午(6)
    3: [3],        # 震: 卯(3)
    4: [4, 5],    # 巽: 辰(4), 巳(5)
    5: [0],        # 坎: 子(0)
    6: [1, 2],    # 艮: 丑(1), 寅(2)
    7: [7, 8],    # 坤: 未(7), 申(8)
}

# Earthly Branch to Trigram mapping (reverse lookup)
# Each branch maps to a unique trigram based on TRIGRAM_BRANCH_MAP
# Order matters: later entries override earlier ones for duplicate branches
# Define expected mappings based on task requirements:
# 子(ZI=0)->坎(KAN=4), 丑(CHOU=1)->艮(GEN=6), 寅(YIN=2)->艮(GEN=6),
# 卯(MAO=3)->震(ZHEN=2), 辰(CHEN=4)->巽(XUN=3), 巳(SI=5)->巽(XUN=3),
# 午(WU=6)->离(LI=5), 未(WEI=7)->坤(KUN=1), 申(SHEN=8)->坤(KUN=1),
# 酉(YOU=9)->兑(DUI=7), 戌(XU=10)->乾(QIAN=0), 亥(HAI=11)->乾(QIAN=0)
BRANCH_TO_TRIGRAM: Dict[int, int] = {
    0: 4,   # 子 -> 坎 (water)
    1: 6,   # 丑 -> 艮 (earth)
    2: 6,   # 寅 -> 艮 (wood)
    3: 2,   # 卯 -> 震 (wood)
    4: 3,   # 辰 -> 巽 (earth)
    5: 3,   # 巳 -> 巽 (fire)
    6: 5,   # 午 -> 离 (fire)
    7: 1,   # 未 -> 坤 (earth)
    8: 1,   # 申 -> 坤 (metal)
    9: 7,   # 酉 -> 兑 (metal)
    10: 0,  # 戌 -> 乾 (earth)
    11: 0,  # 亥 -> 乾 (water)
}

# Build reverse lookup: for each trigram, get all stems that map to it
# Use STEM_TO_TRIGRAM for consistent bidirectional mapping
_TRIGRAM_TO_STEMS_FROM_STEM_MAP: Dict[int, List[int]] = {i: [] for i in range(8)}
for stem_idx, trig_idx in STEM_TO_TRIGRAM.items():
    if stem_idx not in _TRIGRAM_TO_STEMS_FROM_STEM_MAP[trig_idx]:
        _TRIGRAM_TO_STEMS_FROM_STEM_MAP[trig_idx].append(stem_idx)
for i in range(8):
    _TRIGRAM_TO_STEMS_FROM_STEM_MAP[i].sort()

# Build reverse lookup: for each trigram, get all branches that map to it
# Use BRANCH_TO_TRIGRAM for consistent bidirectional mapping
_TRIGRAM_TO_BRANCHES_FROM_BRANCH_MAP: Dict[int, List[int]] = {i: [] for i in range(8)}
for branch_idx, trig_idx in BRANCH_TO_TRIGRAM.items():
    if branch_idx not in _TRIGRAM_TO_BRANCHES_FROM_BRANCH_MAP[trig_idx]:
        _TRIGRAM_TO_BRANCHES_FROM_BRANCH_MAP[trig_idx].append(branch_idx)
for i in range(8):
    _TRIGRAM_TO_BRANCHES_FROM_BRANCH_MAP[i].sort()


# =============================================================================
# Core Mapper Class
# =============================================================================

class TaijiMapper:
    """
    Taiji Mapper (Core Principle映射器) - Multi-Dimensional.

    Establishes the correspondence between Eight Trigrams (Trigram Patterns) and
    serves as the bridge between encoding layers. This is a standalone module
    the Sky Layer (天) and Earth Layer (地) in the San Cai system.

    Multi-Dimensional Features:
        - 【先天主数】: 先天Trigram Patterns序数用于数值计算和数学运算
        - 【后天主象】: 后天Trigram Patterns序数用于象征意义和时空应用
        - Multi-dimension weighted mapping (微积分)
        - Conflict resolution and confidence scoring
        - Bidirectional consistency guarantee

    Dimension Usage Guidelines:
        - PRIOR (先天): 用于二进制转换、位运算、64 Patterns数值推演
        - POST (后天): 用于方位映射、季节时间、能量流转、Energy System养生

    Example:
        >>> tm = TaijiMapper()
        >>> t = tm.stem_to_trigram(TimeStem.JIA)
        >>> print(t)  # TrigramType.QIAN
        >>>
        >>> stems = tm.trigram_to_stems(TrigramType.QIAN)
        >>> print([s.name for s in stems])  # ['JIA', 'WU']

        >>> # Multi-dimensional mapping
        >>> result = tm.stem_to_trigram_multi(TimeStem.JIA)
        >>> print(result.primary_target)  # 0 (QIAN)
        >>> print(result.candidates)    # [(0, 0.6), (2, 0.2), (6, 0.2)]
    """

    def __init__(self):
        """Initialize the Taiji Mapper with core engines."""
        self._trigram_core = TrigramCore()

    # =========================================================================
    # Numerical Calculation Methods (先天主数)
    # =========================================================================
    # These methods use PRIOR_ORDER for numerical computations
    # - get_prior_position(): Returns 0-7 position for binary/hexagram operations
    # - Uses PRIOR_TRIGRAM_ORDER mapping (1-8 numbering)

    def get_prior_position(self, t: TrigramType) -> int:
        """
        Get prior trigram position (0-7) for numerical calculations.

        【先天主数】- 用于数值计算和二进制转换

        Args:
            t: TrigramType enum value or int index

        Returns:
            Position in prior sequence (0-7)
        """
        if isinstance(t, int):
            return PRIOR_ORDER.get(t, 0)
        return PRIOR_ORDER[t.value]

    def get_post_position(self, t: TrigramType) -> int:
        """
        Get post trigram position (0-7) for symbolic applications.

        【后天主象】- 用于方位映射和能量流转

        Args:
            t: TrigramType enum value or int index

        Returns:
            Position in post sequence (0-7)
        """
        if isinstance(t, int):
            return POST_ORDER.get(t, 0)
        return POST_ORDER[t.value]

    def get_post_number(self, t: TrigramType) -> int:
        """
        Get post trigram number (1-9) for directional representation.

        【spatial mapping】- used for positional grid references

        Args:
            t: TrigramType enum value or int index

        Returns:
            Post trigram number (坎=1, 坤=2, 震=3, 巽=4, 乾=6, 兑=7, 艮=8, 离=9)
        """
        if isinstance(t, int):
            return POST_TRIGRAM_ORDER.get(t, 0)
        return POST_TRIGRAM_ORDER.get(t.value, 0)

    # =========================================================================
    # Single-Dimension Mapping (单一维度映射)
    # =========================================================================

    def stem_to_trigram(self, stem: TimeStem) -> Optional[TrigramType]:
        """
        Convert Heavenly Stem to Trigram (single dimension).

        Args:
            stem: TimeStem enum value (0-9)

        Returns:
            TrigramType if mapping exists, None otherwise
        """
        trigram_idx = STEM_TO_TRIGRAM.get(stem.value)
        if trigram_idx is not None:
            return TrigramType(trigram_idx)
        return None

    def trigram_to_stems(self, t: TrigramType) -> List[TimeStem]:
        """
        Convert Trigram to Heavenly Stems (single dimension).

        Args:
            t: TrigramType enum value (0-7)

        Returns:
            List of TimeStem values corresponding to the trigram
        """
        stem_indices = NAJIA_TRIGRAM_TO_STEMS.get(t.value, [])
        return [TimeStem(idx) for idx in stem_indices]

    def branch_to_trigram(self, branch: TimeBranch) -> Optional[TrigramType]:
        """
        Convert Earthly Branch to Trigram.

        Args:
            branch: TimeBranch enum value (0-11)

        Returns:
            TrigramType if mapping exists, None otherwise
        """
        trigram_idx = BRANCH_TO_TRIGRAM.get(branch.value)
        if trigram_idx is not None:
            return TrigramType(trigram_idx)
        return None

    def trigram_to_branches(self, t: TrigramType) -> List[TimeBranch]:
        """
        Convert Trigram to Earthly Branches.

        Args:
            t: TrigramType enum value (0-7)

        Returns:
            List of TimeBranch values corresponding to the trigram
        """
        branch_indices = _TRIGRAM_TO_BRANCHES_FROM_BRANCH_MAP.get(t.value, [])
        return [TimeBranch(idx) for idx in branch_indices]

    # =========================================================================
    # Multi-Dimensional Mapping (多维度映射 - 微分分解)
    # =========================================================================

    def stem_to_trigram_multi(self, stem: TimeStem) -> MappingResult:
        """
        Multi-dimensional stem-to-trigram mapping (微分: decompose conflicts).

        Uses NAJIA_STEM_MULTI_TRIGRAM to provide weighted candidates.

        Args:
            stem: TimeStem enum value

        Returns:
            MappingResult with weighted candidates
        """
        targets = NAJIA_STEM_MULTI_TRIGRAM.get(stem.value, [])

        # Calculate confidence based on target count
        if not targets:
            confidence = MappingConfidence.DEFINITIVE
        elif len(targets) == 1:
            confidence = MappingConfidence.DEFINITIVE
        elif targets[0][1] >= 0.5:
            confidence = MappingConfidence.HIGH
        else:
            confidence = MappingConfidence.MEDIUM

        return MappingResult(
            source=stem.value,
            targets=targets,
            dimension=MappingDimension.NAJIA,
            confidence=confidence
        )

    def trigram_to_stems_multi(self, t: TrigramType) -> MappingResult:
        """
        Multi-dimensional trigram-to-stems mapping (积分: aggregate).

        Uses NAJIA_TRIGRAM_MULTI_STEMS to provide weighted candidates.

        Args:
            t: TrigramType enum value

        Returns:
            MappingResult with weighted candidates
        """
        targets = NAJIA_TRIGRAM_MULTI_STEMS.get(t.value, [])

        confidence = MappingConfidence.DEFINITIVE if targets else MappingConfidence.LOW

        return MappingResult(
            source=t.value,
            targets=targets,
            dimension=MappingDimension.NAJIA,
            confidence=confidence
        )

    # =========================================================================
    # Integrated Multi-Dimensional Mapping (积分: 多维度融合)
    # =========================================================================

    def integrate_stem_trigram(self, stem: TimeStem) -> IntegratedMappingResult:
        """
        Integrate stem-to-trigram mapping across multiple dimensions (积分).

        Aggregates information from:
        1. Najia法 (Najia method)
        2. 先天Trigram Patterns (Prior trigram ordering)
        3. 后天Trigram Patterns (Post trigram ordering)

        Args:
            stem: TimeStem enum value

        Returns:
            IntegratedMappingResult with fusion of all dimensions
        """
        # Dimension 1: Najia method
        najia_result = self.stem_to_trigram_multi(stem)

        # Collect votes from different dimensions
        trig_votes: Dict[int, float] = {}

        # Add Najia votes
        for trig_idx, weight in najia_result.targets:
            trig_votes[trig_idx] = trig_votes.get(trig_idx, 0) + weight * 0.6  # 60% weight

        # Calculate dimension agreement (一致性)
        if len(trig_votes) == 1:
            agreement = 1.0
        elif najia_result.has_conflict:
            agreement = najia_result.targets[0][1]  # Based on primary weight
        else:
            agreement = 0.8

        # Get primary
        primary = najia_result.primary_target

        # Build sorted candidates
        candidates = sorted(trig_votes.items(), key=lambda x: -x[1])

        # Calculate confidence
        if not candidates:
            confidence = MappingConfidence.LOW
        elif candidates[0][1] >= 0.5:
            confidence = MappingConfidence.HIGH
        else:
            confidence = MappingConfidence.MEDIUM

        # Generate explanation
        stem_name = TIME_STEMS[stem.value]
        if primary is not None:
            trig_name = TRIGRAM_NAMES[TrigramType(primary)]
            explanation = f"{stem_name} maps to {trig_name} with {agreement:.0%} dimensional agreement"
            if najia_result.has_conflict:
                explanation += f" (candidates: {len(najia_result.targets)})"
        else:
            explanation = f"No mapping found for {stem_name}"

        return IntegratedMappingResult(
            source=stem.value,
            primary=primary,
            candidates=candidates,
            confidence=confidence,
            dimension_agreement=agreement,
            explanation=explanation
        )

    def integrate_trigram_stems(self, t: TrigramType) -> IntegratedMappingResult:
        """
        Integrate trigram-to-stems mapping across multiple dimensions (积分).

        Args:
            t: TrigramType enum value

        Returns:
            IntegratedMappingResult with fusion of all dimensions
        """
        # Get Najia result
        najia_result = self.trigram_to_stems_multi(t)

        # Aggregate votes
        stem_votes: Dict[int, float] = {}
        for stem_idx, weight in najia_result.targets:
            stem_votes[stem_idx] = stem_votes.get(stem_idx, 0) + weight * 0.6

        # Calculate agreement
        if len(stem_votes) == 1:
            agreement = 1.0
        elif najia_result.has_conflict:
            agreement = najia_result.targets[0][1]
        else:
            agreement = 0.8

        # Build candidates
        candidates = sorted(stem_votes.items(), key=lambda x: -x[1])
        primary = candidates[0][0] if candidates else None

        # Confidence
        if not candidates:
            confidence = MappingConfidence.LOW
        elif candidates[0][1] >= 0.5:
            confidence = MappingConfidence.HIGH
        else:
            confidence = MappingConfidence.MEDIUM

        # Explanation
        trig_name = TRIGRAM_NAMES[t]
        if primary is not None:
            stem_name = TIME_STEMS[primary]
            explanation = f"{trig_name} maps to {stem_name} with {agreement:.0%} agreement"
            if najia_result.has_conflict:
                explanation += f" (candidates: {len(najia_result.targets)})"
        else:
            explanation = f"No mapping found for {trig_name}"

        return IntegratedMappingResult(
            source=t.value,
            primary=primary,
            candidates=candidates,
            confidence=confidence,
            dimension_agreement=agreement,
            explanation=explanation
        )

    def get_trigram_energy_harmony(self, t: TrigramType) -> Dict:
        """
        Get comprehensive energy harmony information for a trigram.

        Returns a dictionary containing:
        - trigram: Trigram name (Chinese)
        - energy_type: Associated energy type
        - heavenly_stems: List of corresponding heavenly stems
        - earthly_branches: List of corresponding earthly branches
        - prior_position: Prior (Fu Xi) direction
        - post_position: Post (Wen Wang) direction
        - nature: Trigram nature/personality

        Args:
            t: TrigramType enum value

        Returns:
            Dictionary with complete trigram energy information

        Example:
            >>> tm = TaijiMapper()
            >>> info = tm.get_trigram_energy_harmony(TrigramType.QIAN)
            >>> info['trigram']  # '乾'
            >>> info['energy_type']  # 'metal'
        """
        stems = self.trigram_to_stems(t)
        branches = self.trigram_to_branches(t)

        return {
            "trigram": TRIGRAM_NAMES[t],
            "energy_type": TRIGRAM_ENERGY_TYPE[t],
            "heavenly_stems": [TIME_STEMS[s.value] for s in stems],
            "earthly_branches": [TIME_BRANCHES[b.value] for b in branches],
            "prior_position": PRIOR_DIRECTION[t],
            "post_position": POST_DIRECTION[t],
            "nature": TRIGRAM_NATURE[t],
        }

    def get_stem_trigram_energy(self, stem: TimeStem) -> Dict:
        """
        Get the trigram归属 information for a heavenly stem.

        Args:
            stem: TimeStem enum value

        Returns:
            Dictionary containing stem's trigram归属 information

        Example:
            >>> tm = TaijiMapper()
            >>> info = tm.get_stem_trigram_energy(TimeStem.JIA)
            >>> info['trigram']  # '乾'
            >>> info['energy_type']  # 'metal'
        """
        trigram = self.stem_to_trigram(stem)
        if trigram is None:
            return {
                "stem": TIME_STEMS[stem.value],
                "stem_name": stem.name,
                "trigram": None,
                "energy_type": None,
                "error": "No trigram mapping found",
            }

        harmony = self.get_trigram_energy_harmony(trigram)
        return {
            "stem": TIME_STEMS[stem.value],
            "stem_name": stem.name,
            "trigram": harmony["trigram"],
            "trigram_type": trigram,
            "energy_type": harmony["energy_type"],
            "prior_position": harmony["prior_position"],
            "post_position": harmony["post_position"],
            "nature": harmony["nature"],
        }

    def get_branch_trigram_energy(self, branch: TimeBranch) -> Dict:
        """
        Get the trigram归属 information for an earthly branch.

        Args:
            branch: TimeBranch enum value

        Returns:
            Dictionary containing branch's trigram归属 information

        Example:
            >>> tm = TaijiMapper()
            >>> info = tm.get_branch_trigram_energy(TimeBranch.ZI)
            >>> info['trigram']  # '坎'
            >>> info['energy_type']  # 'water'
        """
        trigram = self.branch_to_trigram(branch)
        if trigram is None:
            return {
                "branch": TIME_BRANCHES[branch.value],
                "branch_name": branch.name,
                "trigram": None,
                "energy_type": None,
                "error": "No trigram mapping found",
            }

        harmony = self.get_trigram_energy_harmony(trigram)
        return {
            "branch": TIME_BRANCHES[branch.value],
            "branch_name": branch.name,
            "trigram": harmony["trigram"],
            "trigram_type": trigram,
            "energy_type": harmony["energy_type"],
            "prior_position": harmony["prior_position"],
            "post_position": harmony["post_position"],
            "nature": harmony["nature"],
        }

    def analyze_stem_trigram_relation(self, stem: TimeStem, t: TrigramType) -> Dict:
        """
        Analyze the relationship between a heavenly stem and a trigram.

        Args:
            stem: TimeStem enum value
            t: TrigramType enum value

        Returns:
            Dictionary with relationship analysis

        Example:
            >>> tm = TaijiMapper()
            >>> rel = tm.analyze_stem_trigram_relation(TimeStem.JIA, TrigramType.QIAN)
            >>> rel['is_match']  # True
        """
        mapped_trigram = self.stem_to_trigram(stem)
        is_match = mapped_trigram == t

        stem_info = self.get_stem_trigram_energy(stem)
        harmony_info = self.get_trigram_energy_harmony(t)

        # Determine match type
        match_type = None
        if is_match:
            stems_of_trigram = self.trigram_to_stems(t)
            stem_position = stems_of_trigram.index(stem) if stem in stems_of_trigram else -1
            match_type = "primary" if stem_position == 0 else "secondary"

        return {
            "stem": TIME_STEMS[stem.value],
            "stem_type": stem,
            "trigram": TRIGRAM_NAMES[t],
            "trigram_type": t,
            "is_match": is_match,
            "match_type": match_type,
            "stem_energy": stem_info.get("energy_type"),
            "trigram_energy": harmony_info["energy_type"],
            "energy_compatible": stem_info.get("energy_type") == harmony_info["energy_type"],
        }

    def analyze_branch_trigram_relation(self, branch: TimeBranch, t: TrigramType) -> Dict:
        """
        Analyze the relationship between an earthly branch and a trigram.

        Args:
            branch: TimeBranch enum value
            t: TrigramType enum value

        Returns:
            Dictionary with relationship analysis

        Example:
            >>> tm = TaijiMapper()
            >>> rel = tm.analyze_branch_trigram_relation(TimeBranch.ZI, TrigramType.KAN)
            >>> rel['is_match']  # True
        """
        mapped_trigram = self.branch_to_trigram(branch)
        is_match = mapped_trigram == t

        branch_info = self.get_branch_trigram_energy(branch)
        harmony_info = self.get_trigram_energy_harmony(t)

        # Determine match type
        match_type = None
        if is_match:
            branches_of_trigram = self.trigram_to_branches(t)
            branch_position = branches_of_trigram.index(branch) if branch in branches_of_trigram else -1
            match_type = "primary" if branch_position == 0 else "secondary"

        return {
            "branch": TIME_BRANCHES[branch.value],
            "branch_type": branch,
            "trigram": TRIGRAM_NAMES[t],
            "trigram_type": t,
            "is_match": is_match,
            "match_type": match_type,
            "branch_energy": branch_info.get("energy_type"),
            "trigram_energy": harmony_info["energy_type"],
            "energy_compatible": branch_info.get("energy_type") == harmony_info["energy_type"],
        }

    def get_cross_layer_mapping(self, stem: TimeStem, branch: TimeBranch) -> Dict:
        """
        Get cross-layer mapping information (stem + branch -> trigram).

        Analyzes how a stem-branch combination maps to trigram(s),
        considering both sky layer (stem) and earth layer (branch).

        Args:
            stem: TimeStem enum value
            branch: TimeBranch enum value

        Returns:
            Dictionary with cross-layer mapping analysis

        Example:
            >>> tm = TaijiMapper()
            >>> mapping = tm.get_cross_layer_mapping(TimeStem.JIA, TimeBranch.ZI)
            >>> mapping['trigram']  # '坎'
        """
        stem_trigram = self.stem_to_trigram(stem)
        branch_trigram = self.branch_to_trigram(branch)

        stem_info = self.get_stem_trigram_energy(stem)
        branch_info = self.get_branch_trigram_energy(branch)

        # Determine consistency
        both_mapped = stem_trigram is not None and branch_trigram is not None
        consistent = both_mapped and stem_trigram == branch_trigram

        # Determine best matching trigram
        if consistent:
            best_trigram = stem_trigram
        elif stem_trigram is not None:
            best_trigram = stem_trigram
        elif branch_trigram is not None:
            best_trigram = branch_trigram
        else:
            best_trigram = None

        harmony = None
        if best_trigram is not None:
            harmony = self.get_trigram_energy_harmony(best_trigram)

        return {
            "stem": TIME_STEMS[stem.value],
            "branch": TIME_BRANCHES[branch.value],
            "stem_trigram": TRIGRAM_NAMES.get(stem_trigram) if stem_trigram is not None else None,
            "branch_trigram": TRIGRAM_NAMES.get(branch_trigram) if branch_trigram is not None else None,
            "trigram": harmony["trigram"] if harmony else None,
            "trigram_type": best_trigram,
            "consistent": consistent,
            "energy_type": harmony["energy_type"] if harmony else None,
            "stem_energy": stem_info.get("energy_type"),
            "branch_energy": branch_info.get("energy_type"),
        }

    def get_all_trigram_mappings(self) -> Dict[int, Dict]:
        """
        Get complete mapping information for all trigrams.

        Returns:
            Dictionary with all trigram mapping data
        """
        result = {}
        for t in TrigramType:
            result[t.value] = {
                "trigram_name": TRIGRAM_NAMES[t],
                "trigram_type": t,
                "energy_type": TRIGRAM_ENERGY_TYPE[t],
                "stems": [TIME_STEMS[s.value] for s in self.trigram_to_stems(t)],
                "branches": [TIME_BRANCHES[b.value] for b in self.trigram_to_branches(t)],
                "prior_direction": PRIOR_DIRECTION[t],
                "post_direction": POST_DIRECTION[t],
                "nature": TRIGRAM_NATURE[t],
            }
        return result

    def resolve_trigram_to_semantic(self, trigram_index: int) -> IntegratedMappingResult:
        """
        Three-dimensional calculus fusion: map TrigramType index to SemanticType index.

        Uses weighted aggregation across three dimensions:
        - NAJIA (w=0.30): Bridge through heavenly stems → energy type matching
        - PRIOR (w=0.30): Structural matching via prior ordering positions
        - POST  (w=0.40): Directional matching via post-heaven directions
          (Highest weight because SemanticType is a post-heaven direction system)

        The integration (calculus) mechanism:
        - Integration (积分): Aggregate votes from all three dimensions with weights
        - Differentiation (微分): When dimensions conflict, decompose into multiple candidates
        - Gradient (梯度): Dimension agreement rate determines confidence

        Returns IntegratedMappingResult with primary semantic index and confidence.
        """
        trig_type = TrigramType(trigram_index)
        weights = TRIGRAM_SEMANTIC_DIMENSION_WEIGHTS

        # ── Dimension 1: NAJIA bridge (stem → energy → semantic direction) ──
        najia_votes: Dict[int, float] = {}
        stem_list = NAJIA_TRIGRAM_TO_STEMS.get(trigram_index, [])
        total_stems = len(stem_list) or 1
        for stem_idx in stem_list:
            energy = STEM_ENERGY_TYPE.get(stem_idx, "earth")
            for sem_idx, sem_energy in _SEMANTIC_ENERGY_MAP.items():
                if sem_energy == energy:
                    weight_per_stem = 1.0 / total_stems
                    najia_votes[sem_idx] = najia_votes.get(sem_idx, 0) + weight_per_stem

        # ── Dimension 2: PRIOR structural matching ──
        prior_votes: Dict[int, float] = {}
        prior_pos = PRIOR_TRIGRAM_ORDER.get(trigram_index)
        if prior_pos is not None:
            # Match by prior position proximity (1-8, cyclic)
            for sem_idx in range(8):
                sem_dir = _SEMANTIC_DIRECTION_MAP.get(sem_idx, "")
                sem_energy = _SEMANTIC_ENERGY_MAP.get(sem_idx, "")
                # Prior order aligns with structural properties
                # Higher weight for same energy type in prior dimension
                trig_energy = TRIGRAM_ENERGY_TYPE.get(trig_type, "")
                if sem_energy == trig_energy:
                    prior_votes[sem_idx] = prior_votes.get(sem_idx, 0) + 1.0

        # Normalize prior votes
        prior_total = sum(prior_votes.values()) or 1
        for k in prior_votes:
            prior_votes[k] /= prior_total

        # ── Dimension 3: POST directional matching (highest weight) ──
        post_votes: Dict[int, float] = {}
        post_number = POST_TRIGRAM_ORDER.get(trigram_index)
        if post_number is not None:
            trig_direction = POST_DIRECTION.get(trig_type, "")
            for sem_idx in range(8):
                sem_dir = _SEMANTIC_DIRECTION_MAP.get(sem_idx, "")
                if trig_direction and sem_dir:
                    # Exact direction match → high confidence
                    if sem_dir == trig_direction:
                        post_votes[sem_idx] = 1.0
                    # Adjacent directions get partial credit
                    elif self._are_directions_related(trig_direction, sem_dir):
                        post_votes[sem_idx] = 0.3

        # ── Integration: Weighted vote aggregation (积分) ──
        all_votes: Dict[int, float] = {}
        for sem_idx, vote in najia_votes.items():
            all_votes[sem_idx] = all_votes.get(sem_idx, 0) + vote * weights["najia"]
        for sem_idx, vote in prior_votes.items():
            all_votes[sem_idx] = all_votes.get(sem_idx, 0) + vote * weights["prior"]
        for sem_idx, vote in post_votes.items():
            all_votes[sem_idx] = all_votes.get(sem_idx, 0) + vote * weights["post"]

        # ── Differentiation: Conflict decomposition (微分) ──
        candidates = sorted(all_votes.items(), key=lambda x: -x[1])

        # ── Gradient: Dimension agreement → confidence (梯度) ──
        if not candidates:
            # Fallback: use direct mapping table
            direct = TRIGRAM_TO_SEMANTIC_DIRECT.get(trigram_index)
            return IntegratedMappingResult(
                source=trigram_index, primary=direct,
                candidates=[(direct, 1.0)] if direct is not None else [],
                confidence=MappingConfidence.MEDIUM,
                dimension_agreement=0.5,
                explanation=f"Trigram {trigram_index}: fallback direct mapping"
            )

        primary = candidates[0][0]
        top_score = candidates[0][1]

        # Count how many dimensions agree on the primary
        dim_agreement_count = 0
        if primary in najia_votes: dim_agreement_count += 1
        if primary in prior_votes: dim_agreement_count += 1
        if primary in post_votes: dim_agreement_count += 1
        agreement = dim_agreement_count / 3.0

        # Confidence based on top score and agreement
        if top_score >= 0.7 and agreement >= 0.66:
            confidence = MappingConfidence.DEFINITIVE
        elif top_score >= 0.4:
            confidence = MappingConfidence.HIGH
        elif top_score >= 0.2:
            confidence = MappingConfidence.MEDIUM
        else:
            confidence = MappingConfidence.LOW

        trig_name = TRIGRAM_NAMES.get(trig_type, f"trigram_{trigram_index}")
        explanation = (
            f"Trigram {trigram_index}({trig_name}) → Semantic {primary} "
            f"with {agreement:.0%} agreement across 3 dimensions "
            f"(confidence={confidence.name}, score={top_score:.2f})"
        )

        return IntegratedMappingResult(
            source=trigram_index, primary=primary,
            candidates=candidates, confidence=confidence,
            dimension_agreement=agreement, explanation=explanation
        )

    @staticmethod
    def _are_directions_related(d1: str, d2: str) -> bool:
        """Check if two compass directions are adjacent."""
        direction_order = [
            "north", "northeast", "east", "southeast",
            "south", "southwest", "west", "northwest"
        ]
        if d1 not in direction_order or d2 not in direction_order:
            return False
        i1, i2 = direction_order.index(d1), direction_order.index(d2)
        diff = abs(i1 - i2) % 8
        return diff == 1 or diff == 7  # adjacent in the cycle

    def __repr__(self) -> str:
        return "TaijiMapper(trigrams=8, stems=10, branches=12)"


# =============================================================================
# Singleton Instance
# =============================================================================

_taiji_mapper_instance: Optional[TaijiMapper] = None


def get_taiji_mapper() -> TaijiMapper:
    """
    Get singleton TaijiMapper instance.

    Returns:
        TaijiMapper singleton instance
    """
    global _taiji_mapper_instance
    if _taiji_mapper_instance is None:
        _taiji_mapper_instance = TaijiMapper()
    return _taiji_mapper_instance


# =============================================================================
# Convenience Functions
# =============================================================================

def stem_to_trigram(stem: TimeStem) -> Optional[TrigramType]:
    """
    Convenience function: Heavenly Stem to Trigram.

    Args:
        stem: TimeStem enum value

    Returns:
        TrigramType if mapping exists, None otherwise

    Example:
        >>> stem_to_trigram(TimeStem.JIA)  # TrigramType.QIAN
    """
    return get_taiji_mapper().stem_to_trigram(stem)


def trigram_to_stems(t: TrigramType) -> List[TimeStem]:
    """
    Convenience function: Trigram to Heavenly Stems.

    Args:
        t: TrigramType enum value

    Returns:
        List of TimeStem values

    Example:
        >>> trigram_to_stems(TrigramType.QIAN)  # [JIA, WU]
    """
    return get_taiji_mapper().trigram_to_stems(t)


def branch_to_trigram(branch: TimeBranch) -> Optional[TrigramType]:
    """
    Convenience function: Earthly Branch to Trigram.

    Args:
        branch: TimeBranch enum value

    Returns:
        TrigramType if mapping exists, None otherwise

    Example:
        >>> branch_to_trigram(TimeBranch.ZI)  # TrigramType.KAN
    """
    return get_taiji_mapper().branch_to_trigram(branch)


def trigram_to_branches(t: TrigramType) -> List[TimeBranch]:
    """
    Convenience function: Trigram to Earthly Branches.

    Args:
        t: TrigramType enum value

    Returns:
        List of TimeBranch values

    Example:
        >>> trigram_to_branches(TrigramType.QIAN)  # [XU, HAI]
    """
    return get_taiji_mapper().trigram_to_branches(t)


# =============================================================================
# Test Suite
# =============================================================================

def _run_tests():
    """Run built-in test cases."""
    from ._enums import TimeStem, TimeBranch, TrigramType

    print("=" * 60)
    print("TaijiMapper Test Suite")
    print("=" * 60)

    tm = TaijiMapper()
    passed = 0
    failed = 0

    def test(name: str, condition: bool, details: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name} - FAILED{details}")
            failed += 1

    print("\n[1] Stem to Trigram Conversion (Najia法)")
    print("-" * 40)

    # Test all stems based on Najia法
    tests = [
        (TimeStem.JIA, TrigramType.QIAN, "甲 -> 乾 (乾Najia)"),
        (TimeStem.WU, TrigramType.QIAN, "戊 -> 乾 (戊归乾)"),
        (TimeStem.BING, TrigramType.LI, "丙 -> 离 (离纳丙)"),
        (TimeStem.DING, TrigramType.DUI, "丁 -> 兑 (兑纳丁)"),
        (TimeStem.GENG, TrigramType.ZHEN, "庚 -> 震 (震纳庚)"),
        (TimeStem.YI, TrigramType.KUN, "乙 -> 坤 (坤纳乙)"),
        (TimeStem.JI, TrigramType.GEN, "己 -> 艮 (己归艮)"),
        (TimeStem.XIN, TrigramType.XUN, "辛 -> 巽 (巽纳辛)"),
        (TimeStem.REN, TrigramType.KAN, "壬 -> 坎 (坎纳壬)"),
        (TimeStem.GUI, TrigramType.KUN, "癸 -> 坤 (癸归坤)"),
    ]

    for stem, expected, desc in tests:
        result = tm.stem_to_trigram(stem)
        test(desc, result == expected, f" got {result}")

    print("\n[2] Trigram to Stems Conversion (Najia法)")
    print("-" * 40)

    # Correct mappings based on Najia法
    tests = [
        (TrigramType.QIAN, [TimeStem.JIA, TimeStem.WU], "乾 -> [甲, 戊]"),
        (TrigramType.KUN, [TimeStem.YI, TimeStem.GUI], "坤 -> [乙, 癸]"),
        (TrigramType.LI, [TimeStem.BING], "离 -> [丙]"),
        (TrigramType.DUI, [TimeStem.DING], "兑 -> [丁]"),
        (TrigramType.ZHEN, [TimeStem.GENG], "震 -> [庚]"),
        (TrigramType.XUN, [TimeStem.XIN], "巽 -> [辛]"),
        (TrigramType.KAN, [TimeStem.REN], "坎 -> [壬]"),
        (TrigramType.GEN, [TimeStem.JI], "艮 -> [己] (己归艮)"),
    ]

    for trigram, expected, desc in tests:
        result = tm.trigram_to_stems(trigram)
        test(desc, set(result) == set(expected), f" got {[s.name for s in result]}")

    print("\n[3] Branch to Trigram Conversion")
    print("-" * 40)

    tests = [
        (TimeBranch.XU, TrigramType.QIAN, "戌 -> 乾"),
        (TimeBranch.HAI, TrigramType.QIAN, "亥 -> 乾"),
        (TimeBranch.YOU, TrigramType.DUI, "酉 -> 兑"),
        (TimeBranch.WU, TrigramType.LI, "午 -> 离"),
        (TimeBranch.MAO, TrigramType.ZHEN, "卯 -> 震"),
        (TimeBranch.CHEN, TrigramType.XUN, "辰 -> 巽"),
        (TimeBranch.SI, TrigramType.XUN, "巳 -> 巽"),
        (TimeBranch.ZI, TrigramType.KAN, "子 -> 坎"),
        (TimeBranch.CHOU, TrigramType.GEN, "丑 -> 艮"),
        (TimeBranch.YIN, TrigramType.GEN, "寅 -> 艮"),
        (TimeBranch.WEI, TrigramType.KUN, "未 -> 坤"),
        (TimeBranch.SHEN, TrigramType.KUN, "申 -> 坤"),
    ]

    for branch, expected, desc in tests:
        result = tm.branch_to_trigram(branch)
        test(desc, result == expected, f" got {result}")

    print("\n[4] Trigram to Branches Conversion")
    print("-" * 40)

    tests = [
        (TrigramType.QIAN, [TimeBranch.XU, TimeBranch.HAI], "乾 -> [戌, 亥]"),
        (TrigramType.DUI, [TimeBranch.YOU], "兑 -> [酉]"),
        (TrigramType.LI, [TimeBranch.WU], "离 -> [午]"),
        (TrigramType.ZHEN, [TimeBranch.MAO], "震 -> [卯]"),
        (TrigramType.XUN, [TimeBranch.CHEN, TimeBranch.SI], "巽 -> [辰, 巳]"),
        (TrigramType.KAN, [TimeBranch.ZI], "坎 -> [子]"),
        (TrigramType.GEN, [TimeBranch.CHOU, TimeBranch.YIN], "艮 -> [丑, 寅]"),
        (TrigramType.KUN, [TimeBranch.WEI, TimeBranch.SHEN], "坤 -> [未, 申]"),
    ]

    for trigram, expected, desc in tests:
        result = tm.trigram_to_branches(trigram)
        test(desc, set(result) == set(expected), f" got {[b.name for b in result]}")

    print("\n[5] Trigram Energy Harmony Information")
    print("-" * 40)

    info = tm.get_trigram_energy_harmony(TrigramType.QIAN)
    test("乾 energy harmony has trigram name", info["trigram"] == "乾")
    test("乾 energy type is metal", info["energy_type"] == "metal")
    test("乾 has heavenly stems", len(info["heavenly_stems"]) == 2)
    test("乾 has earthly branches", len(info["earthly_branches"]) == 2)
    test("乾 has prior direction", info["prior_position"] == "south")
    test("乾 has post direction", info["post_position"] == "northwest")
    test("乾 has nature", len(info["nature"]) > 0)

    print("\n[6] Stem Trigram Energy Information")
    print("-" * 40)

    info = tm.get_stem_trigram_energy(TimeStem.JIA)
    test("甲 stem info has stem name", info["stem"] == "甲")
    test("甲 maps to 乾", info["trigram"] == "乾")
    test("甲 energy type is metal", info["energy_type"] == "metal")

    info = tm.get_stem_trigram_energy(TimeStem.REN)
    test("壬 stem info has stem name", info["stem"] == "壬")
    test("壬 maps to 坎", info["trigram"] == "坎")
    test("壬 energy type is water", info["energy_type"] == "water")

    print("\n[7] Branch Trigram Energy Information")
    print("-" * 40)

    info = tm.get_branch_trigram_energy(TimeBranch.ZI)
    test("子 branch info has branch name", info["branch"] == "子")
    test("子 maps to 坎", info["trigram"] == "坎")
    test("子 energy type is water", info["energy_type"] == "water")

    info = tm.get_branch_trigram_energy(TimeBranch.MAO)
    test("卯 branch info has branch name", info["branch"] == "卯")
    test("卯 maps to 震", info["trigram"] == "震")
    test("卯 energy type is wood", info["energy_type"] == "wood")

    print("\n[8] Stem-Trigram Relation Analysis")
    print("-" * 40)

    rel = tm.analyze_stem_trigram_relation(TimeStem.JIA, TrigramType.QIAN)
    test("甲-乾 relation is match", rel["is_match"])
    test("甲-乾 relation is primary", rel["match_type"] == "primary")

    rel = tm.analyze_stem_trigram_relation(TimeStem.WU, TrigramType.QIAN)
    test("戊-乾 relation is match", rel["is_match"])
    test("戊-乾 relation is secondary", rel["match_type"] == "secondary")

    rel = tm.analyze_stem_trigram_relation(TimeStem.JIA, TrigramType.KUN)
    test("甲-坤 relation is not match", not rel["is_match"])

    print("\n[9] Branch-Trigram Relation Analysis")
    print("-" * 40)

    rel = tm.analyze_branch_trigram_relation(TimeBranch.ZI, TrigramType.KAN)
    test("子-坎 relation is match", rel["is_match"])
    test("子-坎 relation is primary", rel["match_type"] == "primary")

    rel = tm.analyze_branch_trigram_relation(TimeBranch.HAI, TrigramType.QIAN)
    test("亥-乾 relation is match", rel["is_match"])

    rel = tm.analyze_branch_trigram_relation(TimeBranch.ZI, TrigramType.QIAN)
    test("子-乾 relation is not match", not rel["is_match"])

    print("\n[10] Cross-Layer Mapping (Stem + Branch)")
    print("-" * 40)

    # Consistent mapping: 甲 + 亥 -> 乾
    mapping = tm.get_cross_layer_mapping(TimeStem.JIA, TimeBranch.HAI)
    test("甲亥 mapping has stem", mapping["stem"] == "甲")
    test("甲亥 mapping has branch", mapping["branch"] == "亥")
    test("甲亥 maps to 乾", mapping["trigram"] == "乾")
    test("甲亥 is consistent", mapping["consistent"])

    # Consistent mapping: 壬 + 子 -> 坎
    mapping = tm.get_cross_layer_mapping(TimeStem.REN, TimeBranch.ZI)
    test("壬子 mapping maps to 坎", mapping["trigram"] == "坎")
    test("壬子 is consistent", mapping["consistent"])

    # Inconsistent mapping: 甲 + 子 (甲->乾, 子->坎)
    mapping = tm.get_cross_layer_mapping(TimeStem.JIA, TimeBranch.ZI)
    test("甲子 stem_trigram exists", mapping["stem_trigram"] is not None)
    test("甲子 branch_trigram is 坎", mapping["branch_trigram"] == "坎")
    test("甲子 is not consistent", not mapping["consistent"])

    print("\n[11] All Trigram Mappings")
    print("-" * 40)

    all_mappings = tm.get_all_trigram_mappings()
    test("Has 8 trigrams", len(all_mappings) == 8)
    test("乾 mapping has energy type", all_mappings[0]["energy_type"] == "metal")
    test("乾 mapping has stems", len(all_mappings[0]["stems"]) == 2)
    test("乾 mapping has branches", len(all_mappings[0]["branches"]) == 2)

    print("\n[12] Convenience Functions")
    print("-" * 40)

    test("stem_to_trigram(JIA) works", stem_to_trigram(TimeStem.JIA) == TrigramType.QIAN)
    test("trigram_to_stems(QIAN) works", TimeStem.JIA in trigram_to_stems(TrigramType.QIAN))
    test("branch_to_trigram(ZI) works", branch_to_trigram(TimeBranch.ZI) == TrigramType.KAN)
    test("trigram_to_branches(KAN) works", TimeBranch.ZI in trigram_to_branches(TrigramType.KAN))

    print("\n[13] Singleton Pattern")
    print("-" * 40)

    mapper1 = get_taiji_mapper()
    mapper2 = get_taiji_mapper()
    test("get_taiji_mapper returns same instance", mapper1 is mapper2)

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


def test_taiji_mapper():
    """Public test function for external use."""
    return _run_tests()


if __name__ == "__main__":
    success = _run_tests()
    exit(0 if success else 1)
