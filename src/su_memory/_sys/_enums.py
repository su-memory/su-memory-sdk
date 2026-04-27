"""
Enumeration Type System

This module defines the complete enumeration types for the semantic memory system,
using modern technical terminology to represent traditional concepts.

Architecture Layers (核心演化脉络: 无极→Core Principle→Dual Forces→Triad System→Four Symbols→Energy System→Trigram Patterns→干支):
- YinYang Layer (Duality): The fundamental duality - all systems are built on this
- ThreePowers Layer (Triad System): Tian (Sky), Di (Earth), Ren (Human) - spatial framework
- FourSymbols Layer (Four Symbols): Shao Yang, Tai Yang, Shao Yin, Tai Yin - temporal rhythm
- FiveElements Layer (Energy System): Wood, Fire, Earth, Metal, Water - energy operation rules
- Trigrams Layer (Trigram Patterns): Eight trigrams - concrete manifestation of yin-yang
- Spacetime Layer (时空): TimeStems, TimeBranches - quantitative measurement

Core Principle: Duality平衡为万物之本，平衡则化生，失衡则衰败

All enums use IntEnum with consecutive integer values starting from 0.
"""

from enum import IntEnum, Enum

# ============================================================
# Yin-Yang Layer (Duality层) - Foundation
# ============================================================

class YinYang(Enum):
    """Yin-Yang duality enumeration"""
    YIN = 0   # 阴 - receptive, passive, cold, inward
    YANG = 1  # 阳 - creative, active, warm, outward


class ThreePowers(Enum):
    """
    Three Powers (Triad System) enumeration.

    Represents the three fundamental forces:
    - TIAN: Pure yang, sky, heaven, time, qi
    - DI: Pure yin, earth, ground, space, form
    - REN: Yin-yang harmony, human, balance carrier
    """
    TIAN = 0  # 天 - sky/heaven (pure yang)
    REN = 1   # 人 - human (yin-yang harmony)
    DI = 2    # 地 - earth (pure yin)


# ============================================================
# Four Symbols Layer (Four Symbols层) - Temporal Rhythm
# ============================================================

class FourSymbols(IntEnum):
    """
    Four Symbols (Four Symbols) enumeration.

    Represents the four stages of yin-yang transformation:
    - SHAO_YANG: Spring (少阳) - yin declining, yang rising, birth phase
    - TAI_YANG: Summer (太阳) - yang peak, outward expansion, growth phase
    - SHAO_YIN: Autumn (少阴) - yang declining, yin rising, harvest phase
    - TAI_YIN: Winter (太阴) - yin peak, inward storage, rest phase

    Corresponds to Five Elements and seasons:
    - SHAO_YANG -> 木 (Wood) -> Spring
    - TAI_YANG -> 火 (Fire) -> Summer
    - SHAO_YIN -> 金 (Metal) -> Autumn
    - TAI_YIN -> 水 (Water) -> Winter
    """
    SHAO_YANG = 0  # 少阳 - Spring, Wood, birth (阴消阳长)
    TAI_YANG = 1   # 太阳 - Summer, Fire, growth (阳气极盛)
    SHAO_YIN = 2  # 少阴 - Autumn, Metal, harvest (阳消阴长)
    TAI_YIN = 3    # 太阴 - Winter, Water, storage (阴气极盛)


class Season(IntEnum):
    """
    Four Seasons (四季) enumeration.

    Spring -> Summer -> Autumn -> Winter
    With Earth (土) as center, governing late summer/transitions
    """
    SPRING = 0     # 春 - Wood, growth
    SUMMER = 1     # 夏 - Fire, flourishing
    LATE_SUMMER = 2  # 长夏 - Earth, transformation/balance
    AUTUMN = 3     # 秋 - Metal,收敛
    WINTER = 4     # 冬 - Water, 闭藏


# ============================================================
# Sky Layer - Temporal System
# Ten Heavenly Stems (Shi Gan)
# ============================================================

class TimeStem(IntEnum):
    """
    Ten Heavenly Stems enumeration.

    Represents the ten temporal stems in the cyclical time system:
    - JIA/YI: Wood (yang/yin)
    - BING/DING: Fire (yang/yin)
    - WU/JI: Earth (yang/yin)
    - GENG/XIN: Metal (yang/yin)
    - REN/GUI: Water (yang/yin)
    """
    JIA = 0    # 甲 - yang wood
    YI = 1     # 乙 - yin wood
    BING = 2   # 丙 - yang fire
    DING = 3   # 丁 - yin fire
    WU = 4     # 戊 - yang earth
    JI = 5     # 己 - yin earth
    GENG = 6   # 庚 - yang metal
    XIN = 7    # 辛 - yin metal
    REN = 8    # 壬 - yang water
    GUI = 9    # 癸 - yin water


class TimeBranch(IntEnum):
    """
    Twelve Earthly Branches enumeration.

    Represents the twelve temporal branches in the cyclical time system:
    - ZI/CHOU: Water/Earth
    - YIN/MAO: Wood
    - CHEN/SI: Earth/Fire
    - WU/WEI: Fire/Earth
    - SHEN/YOU: Metal
    - XU/HAI: Earth/Water
    """
    ZI = 0     # 子 - yang water
    CHOU = 1   # 丑 - yin earth
    YIN = 2    # 寅 - yang wood
    MAO = 3    # 卯 - yin wood
    CHEN = 4   # 辰 - yang earth
    SI = 5     # 巳 - yin fire
    WU = 6     # 午 - yang fire
    WEI = 7    # 未 - yin earth
    SHEN = 8   # 申 - yang metal
    YOU = 9    # 酉 - yin metal
    XU = 10    # 戌 - yang earth
    HAI = 11   # 亥 - yin water


class BranchRelation(IntEnum):
    """
    Earthly Branch relationship types.

    Defines the various relationships between earthly branches:
    - LIU_HE: Six Harmonies (合)
    - SAN_HE: Three Combinations (三合)
    - LIU_CHONG: Six Conflicts (六冲)
    - SAN_XING: Three Punishments (三刑)
    - LIU_HAI: Six Harms (六害)
    - PO: Broken relationship (破)
    """
    LIU_HE = 1      # 六合 - Six Harmonies
    SAN_HE = 2      # 三合 - Three Combinations
    LIU_CHONG = 3   # 六冲 - Six Conflicts
    SAN_XING = 4    # 三刑 - Three Punishments
    LIU_HAI = 5     # 六害 - Six Harms
    PO = 6          # 破 - Broken relationship


# ============================================================
# Earth Layer - Semantic Category System
# Eight Trigrams (Ba Gua)
# ============================================================

class TrigramType(IntEnum):
    """
    Eight Trigrams enumeration.

    Represents the eight fundamental trigrams in the semantic system:
    - QIAN/KUN: Heaven/Earth (Creative/Receptive)
    - ZHEN/XUN: Thunder/Wind (Dynamic/Penetrating)
    - KAN/LI: Water/Fire (Abyss/Light)
    - GEN/DUI: Mountain/Lake (Steady/Joyful)
    """
    QIAN = 0   # 乾 - Heaven (sky/northwest/active)
    KUN = 1    # 坤 - Earth (ground/southwest/receptive)
    ZHEN = 2   # 震 - Thunder (east/dynamic/trigger)
    XUN = 3    # 巽 - Wind (southeast/penetrating/entry)
    KAN = 4    # 坎 - Water (north/hidden/risk)
    LI = 5     # 离 - Fire (south/bright/connection)
    GEN = 6    # 艮 - Mountain (northeast/steady/stop)
    DUI = 7    # 兑 - Lake (west/joyful/exchange)


class TrigramRelation(IntEnum):
    """
    Trigram relationship types.

    Defines the various relationships between trigrams:
    - CUO: Opposite trigram (all yin/yang reversed)
    - HU: Mutual trigram (upper/lower exchanged)
    - ZONG: Reversed trigram (rotated 180 degrees)
    - BAN: Half trigram (main body retained)
    - JIA: Cross trigram (yin/yang exchanged)
    """
    CUO = 1    # 错卦 - Opposite trigram
    HU = 2     # 互卦 - Mutual trigram
    ZONG = 3   # 综卦 - Reversed trigram
    BAN = 4    # 半卦 - Half trigram
    JIA = 5    # 交卦 - Cross trigram


# ============================================================
# Human Layer - Energy System
# Five Elements (Wu Xing)
# ============================================================

class EnergyType(IntEnum):
    """
    Five Elements enumeration.

    Represents the five fundamental energy types in the system:
    - WOOD: Growth, expansion, flexibility
    - FIRE: Energy, passion, transformation
    - EARTH: Stability, nourishment, transformation
    - METAL: Structure, clarity, precision
    - WATER: Flow, wisdom, adaptation
    """
    WOOD = 0   # 木 - Wood
    FIRE = 1   # 火 - Fire
    EARTH = 2  # 土 - Earth
    METAL = 3  # 金 - Metal
    WATER = 4  # 水 - Water


class EnergyRelation(IntEnum):
    """
    Five Elements relationship types.

    Defines the interactions between energy types:
    - ENHANCE: Generating cycle (相生)
    - SUPPRESS: Controlling cycle (相克)
    - OVERCONSTRAINT: Excessive control (相乘)
    - REVERSE: Reverse control (相侮)
    - SAME: Same category (同类)
    """
    ENHANCE = 1        # 相生 - Generating
    SUPPRESS = 2       # 相克 - Controlling
    OVERCONSTRAINT = 3 # 相乘 - Overconstraint
    REVERSE = 4        # 相侮 - Reverse control
    SAME = 5           # 同类 - Same category


class StrengthState(IntEnum):
    """
    Energy strength state enumeration.

    Represents the five strength states of energy:
    - WANG: Strong旺 (at peak strength)
    - XIANG: Balanced相 (helping phase)
    - XIU: Rested休 (resting phase)
    - QIU: Confined囚 (imprisoned phase)
    - SI: Declined死 (weakened phase)
    """
    WANG = 0   # 旺 - Strong
    XIANG = 1  # 相 - Balanced
    XIU = 2    # 休 - Rested
    QIU = 3    # 囚 - Confined
    SI = 4     # 死 - Declined


class EnergyPattern(IntEnum):
    """
    Energy pattern configuration types.

    Represents the different energy pattern configurations:
    - ZHI_HUA: Regulation pattern (制化格)
    - CONG_WANG: Following strength pattern (从旺格)
    - ZHUAN_WANG: Dedicated strength pattern (专旺格)
    - FAN_WANG: Reverse pattern (反局格)
    - PEI_HE: Coordination pattern (配合格)
    """
    ZHI_HUA = 0      # 制化格 - Regulation pattern
    CONG_WANG = 1    # 从旺格 - Following strength pattern
    ZHUAN_WANG = 2   # 专旺格 - Dedicated strength pattern
    FAN_WANG = 3     # 反局格 - Reverse pattern
    PEI_HE = 4       # 配合格 - Coordination pattern


# ============================================================
# Backward Compatibility Aliases
# ============================================================
# These aliases are provided for backward compatibility with _terms.py
# They map to the new enum-based system

# Note: The following aliases are deprecated and will be removed in future
# Users should migrate to the new IntEnum types above
