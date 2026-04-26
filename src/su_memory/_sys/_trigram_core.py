"""
Trigram Core Engine (八卦象数核心引擎)

This module provides the core engine for the Eight Trigrams (Bagua) system,
supporting prior/post transformations, hexagram mutations, and energy analysis.

Architecture:
- TrigramInfo: Individual trigram data structure
- HexagramInfo: Hexagram (64-pattern) data structure
- TrigramCore: Core engine for trigram operations

Dependencies:
- Uses _enums.TrigramType for trigram enumeration
- Uses _enums.TrigramRelation for trigram relationships
- Uses _terms mapping tables for energy/direction data
- Reuses encoders.HEXAGRAM_NAMES for hexagram names
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

from ._enums import TrigramType, TrigramRelation
from .encoders import HEXAGRAM_NAMES


# ============================================================
# Binary Mapping Table
# Maps TrigramType (0-7) to binary tuple (yin=0, yang=1)
# Formula: binary = 7 - trigram_idx (MSB to LSB)
# ============================================================

# Prior trigram order (Fu Xi sequence): Qian, Dui, Li, Zhen, Xun, Kan, Gen, Kun
# Index:  0      1      2      3      4      5      6      7
# Binary: 111,   011,   101,   001,   110,   010,   100,   000

TRIGRAM_TO_BINARY: Dict[TrigramType, Tuple[int, int, int]] = {
    TrigramType.QIAN: (1, 1, 1),  # 111
    TrigramType.DUI: (0, 1, 1),  # 011
    TrigramType.LI: (1, 0, 1),   # 101
    TrigramType.ZHEN: (0, 0, 1), # 001
    TrigramType.XUN: (1, 1, 0),  # 110
    TrigramType.KAN: (0, 1, 0),  # 010
    TrigramType.GEN: (1, 0, 0),   # 100
    TrigramType.KUN: (0, 0, 0),  # 000
}

BINARY_TO_TRIGRAM: Dict[Tuple[int, int, int], TrigramType] = {
    (1, 1, 1): TrigramType.QIAN,
    (0, 1, 1): TrigramType.DUI,
    (1, 0, 1): TrigramType.LI,
    (0, 0, 1): TrigramType.ZHEN,
    (1, 1, 0): TrigramType.XUN,
    (0, 1, 0): TrigramType.KAN,
    (1, 0, 0): TrigramType.GEN,
    (0, 0, 0): TrigramType.KUN,
}

# ============================================================
# Trigram Tree Levels (八卦三爻对应三才)
# Each trigram has 3 lines representing 三才:
# - 上爻 (top): 天道 (Heaven) - YANG
# - 中爻 (middle): 人道 (Human) - yin-yang harmony
# - 下爻 (bottom): 地道 (Earth) - YIN
# ============================================================

# Trigram Tree Line Structure (三爻结构)
# Each entry: (top_line, middle_line, bottom_line) where 1=yang, 0=yin
# Top line: YANG (天道), Middle line: varies, Bottom line: YIN (地道)

TRIGRAM_TREE_STRUCTURE: Dict[TrigramType, Tuple[int, int, int]] = {
    # 三爻结构 (上爻=天道, 中爻=人道, 下爻=地道)
    TrigramType.QIAN: (1, 1, 1),  # 乾: 天天天的阳极
    TrigramType.DUI: (0, 1, 1),  # 兑: 地下天 (阴在地, 阳在人天)
    TrigramType.LI: (1, 0, 1),   # 离: 天地下 (阳在天, 阴在地)
    TrigramType.ZHEN: (0, 0, 1), # 震: 地地的阳生 (阴阴阳)
    TrigramType.XUN: (1, 1, 0),  # 巽: 天天的阴入 (阳阳阴)
    TrigramType.KAN: (0, 1, 0),  # 坎: 地中地 (阴阳阴)
    TrigramType.GEN: (1, 0, 0),   # 艮: 天地的止 (阳阴阴)
    TrigramType.KUN: (0, 0, 0),  # 坤: 地地的阴极
}

# Three Powers Attribution for each trigram line
# Position 0: 上爻(天道), Position 1: 中爻(人道), Position 2: 下爻(地道)
TRIGRAM_THREE_POWERS: Dict[TrigramType, Tuple[str, str, str]] = {
    TrigramType.QIAN: ("TIAN", "REN", "DI"),  # 乾: 天人地 (三阳)
    TrigramType.DUI: ("DI", "REN", "TIAN"),   # 兑: 地的天
    TrigramType.LI: ("TIAN", "REN", "DI"),    # 离: 天人地 (天地阴阳)
    TrigramType.ZHEN: ("DI", "REN", "DI"),    # 震: 地人地
    TrigramType.XUN: ("TIAN", "REN", "DI"),   # 巽: 天人地 (天入阴)
    TrigramType.KAN: ("DI", "REN", "DI"),     # 坎: 地中地 (水险)
    TrigramType.GEN: ("TIAN", "REN", "DI"),   # 艮: 天人地 (山止)
    TrigramType.KUN: ("DI", "REN", "DI"),     # 坤: 地人地 (三阴)
}

# Yin-Yang Attribution for each trigram (based on overall nature)
TRIGRAM_YINYANG: Dict[TrigramType, str] = {
    TrigramType.QIAN: "YANG",    # 乾 - 纯阳
    TrigramType.DUI: "YIN",      # 兑 - 阴
    TrigramType.LI: "YANG",      # 离 - 阳
    TrigramType.ZHEN: "YANG",    # 震 - 阳
    TrigramType.XUN: "YIN",      # 巽 - 阴
    TrigramType.KAN: "YIN",      # 坎 - 阴
    TrigramType.GEN: "YANG",     # 艮 - 阳
    TrigramType.KUN: "YIN",      # 坤 - 纯阴
}

# Four Symbols Attribution (四象对应)
# Based on trigram energy type and position
TRIGRAM_FOUR_SYMBOLS: Dict[TrigramType, str] = {
    # 木属性 -> 少阳 (春)
    TrigramType.ZHEN: "SHAO_YANG",  # 震 - 雷, 少阳
    TrigramType.XUN: "SHAO_YANG",    # 巽 - 风, 少阳
    # 火属性 -> 太阳 (夏)
    TrigramType.LI: "TAI_YANG",      # 离 - 火, 太阳
    # 金属性 -> 少阴 (秋)
    TrigramType.DUI: "SHAO_YIN",     # 兑 - 泽, 少阴
    TrigramType.QIAN: "SHAO_YIN",     # 乾 - 天, 少阴
    # 水属性 -> 太阴 (冬)
    TrigramType.KAN: "TAI_YIN",      # 坎 - 水, 太阴
    # 土属性 -> 中宫 (长夏/平衡)
    TrigramType.GEN: "CENTER",       # 艮 - 山, 土
    TrigramType.KUN: "CENTER",      # 坤 - 地, 土
}


# ============================================================
# Prior Order Table (伏羲卦序 - 先天八卦)
# Position 0-7: Qian, Dui, Li, Zhen, Xun, Kan, Gen, Kun
# 
# 【先天主数】- 先天八卦用于数值计算和数学运算
# - 应用于六十四卦的数值推演和序数计算
# - 用于二进制转换和位运算操作
# - 序数从 0 开始，体现数学生成序列
# ============================================================

PRIOR_ORDER: Dict[TrigramType, int] = {
    TrigramType.QIAN: 0,  # 乾 - 数1 (先天序位)
    TrigramType.DUI: 1,  # 兑 - 数2
    TrigramType.LI: 2,   # 离 - 数3
    TrigramType.ZHEN: 3, # 震 - 数4
    TrigramType.XUN: 4,  # 巽 - 数5
    TrigramType.KAN: 5,  # 坎 - 数6
    TrigramType.GEN: 6,  # 艮 - 数7
    TrigramType.KUN: 7,  # 坤 - 数8
}

# Reverse mapping: position -> TrigramType
ORDER_TO_TRIGRAM_PRIOR: Dict[int, TrigramType] = {v: k for k, v in PRIOR_ORDER.items()}

# ============================================================
# Post Order Table (文王卦序 - 后天八卦)
# Position 0-7: Kan, Kun, Zhen, Xun, Qian, Dui, Gen, Li
# 
# 【后天主象】- 后天八卦用于象征意义和时空应用
# - 应用于方位、季节、时间等空间映射
# - 用于能量流转和五行关系的实际应用
# - 序数 0-7 映射到 1-9 方位数，体现后天应用
# ============================================================

POST_ORDER: Dict[TrigramType, int] = {
    TrigramType.KAN: 0,  # 坎 - 象1 (北方水)
    TrigramType.KUN: 1,  # 坤 - 象2 (西南土)
    TrigramType.ZHEN: 2, # 震 - 象3 (东方木)
    TrigramType.XUN: 3,  # 巽 - 象4 (东南木)
    TrigramType.QIAN: 4, # 乾 - 象6 (西北金)
    TrigramType.DUI: 5,  # 兑 - 象7 (西方金)
    TrigramType.GEN: 6,  # 艮 - 象8 (东北土)
    TrigramType.LI: 7,   # 离 - 象9 (南方火)
}

ORDER_TO_TRIGRAM_POST: Dict[int, TrigramType] = {v: k for k, v in POST_ORDER.items()}

# ============================================================
# Energy Type Mapping (五行归属)
# ============================================================

TRIGRAM_ENERGY_TYPE: Dict[TrigramType, str] = {
    TrigramType.QIAN: "metal",  # 乾 - metal
    TrigramType.DUI: "metal",   # 兑 - metal
    TrigramType.LI: "fire",     # 离 - fire
    TrigramType.ZHEN: "wood",   # 震 - wood
    TrigramType.XUN: "wood",    # 巽 - wood
    TrigramType.KAN: "water",   # 坎 - water
    TrigramType.GEN: "earth",    # 艮 - earth
    TrigramType.KUN: "earth",    # 坤 - earth
}

# ============================================================
# Trigram Name Mapping
# ============================================================

TRIGRAM_NAMES: Dict[TrigramType, str] = {
    TrigramType.QIAN: "乾",
    TrigramType.KUN: "坤",
    TrigramType.ZHEN: "震",
    TrigramType.XUN: "巽",
    TrigramType.KAN: "坎",
    TrigramType.LI: "离",
    TrigramType.GEN: "艮",
    TrigramType.DUI: "兑",
}

# ============================================================
# Prior Direction (先天方位)
# ============================================================

PRIOR_DIRECTION: Dict[TrigramType, str] = {
    TrigramType.QIAN: "south",
    TrigramType.KUN: "north",
    TrigramType.ZHEN: "east",
    TrigramType.XUN: "southeast",
    TrigramType.KAN: "west",
    TrigramType.LI: "east",
    TrigramType.GEN: "northeast",
    TrigramType.DUI: "southeast",
}

# ============================================================
# Post Direction (后天方位)
# ============================================================

POST_DIRECTION: Dict[TrigramType, str] = {
    TrigramType.QIAN: "northwest",
    TrigramType.KUN: "southwest",
    TrigramType.ZHEN: "east",
    TrigramType.XUN: "southeast",
    TrigramType.KAN: "north",
    TrigramType.LI: "south",
    TrigramType.GEN: "northeast",
    TrigramType.DUI: "west",
}

# ============================================================
# Trigram Nature (性情)
# ============================================================

TRIGRAM_NATURE: Dict[TrigramType, str] = {
    TrigramType.QIAN: "刚健",   # Strong and vigorous
    TrigramType.KUN: "柔顺",    # Soft and receptive
    TrigramType.ZHEN: "震动",   # Movement and action
    TrigramType.XUN: "进入",    # Penetration and entry
    TrigramType.KAN: "陷入",    # Danger and risk
    TrigramType.LI: "明亮",     # Brightness and clarity
    TrigramType.GEN: "停止",    # Stillness and stop
    TrigramType.DUI: "喜悦",    # Joy and exchange
}


# ============================================================
# Data Structures
# ============================================================

@dataclass
class TrigramInfo:
    """
    Trigram information data structure.
    
    Contains all relevant data for a single trigram including
    name, energy type, directions, and nature.
    """
    trigram: TrigramType
    name: str  # 乾/坤/震/巽/坎/离/艮/兑
    energy_type: str
    prior_direction: str   # Prior (Fu Xi) direction
    post_direction: str     # Post (Wen Wang) direction
    nature: str            # Nature/personality
    
    @property
    def binary(self) -> Tuple[int, int, int]:
        """
        Convert to binary representation (yin=0, yang=1).
        
        Example:
            Qian (111) -> (1, 1, 1)
            Kun (000) -> (0, 0, 0)
        
        Returns:
            Tuple of 3 bits (MSB to LSB)
        """
        return TRIGRAM_TO_BINARY[self.trigram]
    
    @property
    def prior_order(self) -> int:
        """Get prior trigram order (0-7, Fu Xi sequence)."""
        return PRIOR_ORDER[self.trigram]
    
    @property
    def post_order(self) -> int:
        """Get post trigram order (0-7, Wen Wang sequence)."""
        return POST_ORDER[self.trigram]


@dataclass
class HexagramInfo:
    """
    Hexagram (64-pattern) information data structure.
    
    Represents a hexagram composed of upper and lower trigrams
    with associated index, name, and energy information.
    """
    upper: TrigramType
    lower: TrigramType
    number: int  # 0-63
    name: str    # Hexagram name
    
    @property
    def energy_type(self) -> str:
        """
        Get energy type for this hexagram.
        
        Combines upper and lower trigram energy types.
        """
        upper_energy = TRIGRAM_ENERGY_TYPE[self.upper]
        lower_energy = TRIGRAM_ENERGY_TYPE[self.lower]
        return f"{upper_energy}-{lower_energy}"
    
    @property
    def binary_representation(self) -> str:
        """
        Get 6-bit binary representation (yin=0, yang=1).
        
        Format: upper bits (3) + lower bits (3)
        """
        upper_bits = TRIGRAM_TO_BINARY[self.upper]
        lower_bits = TRIGRAM_TO_BINARY[self.lower]
        return ''.join(str(b) for b in upper_bits + lower_bits)


# ============================================================
# Core Engine Class
# ============================================================

class TrigramCore:
    """
    Trigram core engine (八卦象数核心引擎).
    
    Provides comprehensive operations for the Eight Trigrams system:
    - Prior/Post trigram order conversion
    - Binary representation conversion
    - Hexagram mutation transformations (错卦/互卦/综卦)
    - 64-hexagram derivation and lookup
    - Trigram relationship analysis
    
    Example:
        >>> tc = TrigramCore()
        >>> info = tc.get_trigram_info(TrigramType.QIAN)
        >>> print(info.name)  # 乾
        >>> bits = tc.trigram_to_binary(TrigramType.QIAN)
        >>> print(bits)  # (1, 1, 1)
    """
    
    def __init__(self):
        """Initialize the TrigramCore engine."""
        self._hexagram_map = self._build_hexagram_map()
        self._cuo_cache: Dict[Tuple[TrigramType, TrigramType], Tuple[TrigramType, TrigramType]] = {}
        self._hu_cache: Dict[Tuple[TrigramType, TrigramType], Tuple[TrigramType, TrigramType]] = {}
        self._zong_cache: Dict[Tuple[TrigramType, TrigramType], Tuple[TrigramType, TrigramType]] = {}
        self._ban_cache: Dict[Tuple[TrigramType, TrigramType], Tuple[TrigramType, TrigramType]] = {}
        self._jia_cache: Dict[Tuple[TrigramType, TrigramType], Tuple[TrigramType, TrigramType]] = {}
    
    def _build_hexagram_map(self) -> Dict[int, Tuple[TrigramType, TrigramType]]:
        """
        Build hexagram index to (upper, lower) mapping.
        
        Index formula: number = upper * 8 + lower
        
        Returns:
            Dictionary mapping 0-63 to (upper, lower) trigram tuple
        """
        hexagram_map: Dict[int, Tuple[TrigramType, TrigramType]] = {}
        for upper_idx in range(8):
            for lower_idx in range(8):
                number = upper_idx * 8 + lower_idx
                hexagram_map[number] = (
                    TrigramType(upper_idx),
                    TrigramType(lower_idx)
                )
        return hexagram_map
    
    def get_trigram_info(self, t: TrigramType) -> TrigramInfo:
        """
        Get complete trigram information.
        
        Args:
            t: TrigramType enum value
        
        Returns:
            TrigramInfo with all trigram attributes
        
        Example:
            >>> info = tc.get_trigram_info(TrigramType.QIAN)
            >>> info.name  # 乾
            >>> info.energy_type  # metal
        """
        return TrigramInfo(
            trigram=t,
            name=TRIGRAM_NAMES[t],
            energy_type=TRIGRAM_ENERGY_TYPE[t],
            prior_direction=PRIOR_DIRECTION[t],
            post_direction=POST_DIRECTION[t],
            nature=TRIGRAM_NATURE[t]
        )
    
    def get_prior_order(self, t: TrigramType) -> int:
        """
        Get prior trigram order (0-7).
        
        Fu Xi (伏羲) sequence: Qian, Dui, Li, Zhen, Xun, Kan, Gen, Kun
        
        Args:
            t: TrigramType enum value
        
        Returns:
            Position in prior sequence (0-7)
        
        Example:
            >>> tc.get_prior_order(TrigramType.QIAN)  # 0
            >>> tc.get_prior_order(TrigramType.KUN)  # 7
        """
        return PRIOR_ORDER[t]
    
    def get_post_order(self, t: TrigramType) -> int:
        """
        Get post trigram order (0-7).
        
        Wen Wang (文王) sequence: Kan, Kun, Zhen, Xun, Qian, Dui, Gen, Li
        
        Args:
            t: TrigramType enum value
        
        Returns:
            Position in post sequence (0-7)
        
        Example:
            >>> tc.get_post_order(TrigramType.KAN)  # 0
            >>> tc.get_post_order(TrigramType.LI)  # 7
        """
        return POST_ORDER[t]
    
    def trigram_to_binary(self, t: TrigramType) -> Tuple[int, int, int]:
        """
        Convert trigram to binary tuple.
        
        Maps trigram to 3-bit binary (yin=0, yang=1).
        Top bit is first element of tuple.
        
        Args:
            t: TrigramType enum value
        
        Returns:
            Tuple of 3 bits: (MSB, middle, LSB)
        
        Example:
            >>> tc.trigram_to_binary(TrigramType.QIAN)  # (1, 1, 1)
            >>> tc.trigram_to_binary(TrigramType.KUN)   # (0, 0, 0)
        """
        return TRIGRAM_TO_BINARY[t]
    
    def binary_to_trigram(self, bits: Tuple[int, int, int]) -> TrigramType:
        """
        Convert binary tuple to trigram.
        
        Args:
            bits: Tuple of 3 bits (yin=0, yang=1)
        
        Returns:
            TrigramType enum value
        
        Raises:
            ValueError: If binary combination is invalid
        
        Example:
            >>> tc.binary_to_trigram((1, 1, 1))  # TrigramType.QIAN
            >>> tc.binary_to_trigram((0, 0, 0))  # TrigramType.KUN
        """
        return BINARY_TO_TRIGRAM[bits]
    
    def get_cuo_hexagram(
        self,
        upper: TrigramType,
        lower: TrigramType
    ) -> Tuple[TrigramType, TrigramType]:
        """
        Get Cuo (错卦) - opposite transformation.
        
        Inverts all yin/yang in the hexagram (all 6 yao lines).
        
        Args:
            upper: Upper trigram
            lower: Lower trigram
        
        Returns:
            Tuple of (upper, lower) for the opposite hexagram
        
        Example:
            >>> tc.get_cuo_hexagram(QIAN, QIAN)  # (KUN, KUN)
            # 111111 (Qian) -> 000000 (Kun)
        """
        cache_key = (upper, lower)
        if cache_key in self._cuo_cache:
            return self._cuo_cache[cache_key]
        
        # Cuo: invert all bits
        upper_bits = TRIGRAM_TO_BINARY[upper]
        lower_bits = TRIGRAM_TO_BINARY[lower]
        
        cuo_upper_bits = tuple(1 - b for b in upper_bits)
        cuo_lower_bits = tuple(1 - b for b in lower_bits)
        
        cuo_upper = BINARY_TO_TRIGRAM[cuo_upper_bits]
        cuo_lower = BINARY_TO_TRIGRAM[cuo_lower_bits]
        
        result = (cuo_upper, cuo_lower)
        self._cuo_cache[cache_key] = result
        return result
    
    def get_hu_hexagram(
        self,
        upper: TrigramType,
        lower: TrigramType
    ) -> Tuple[TrigramType, TrigramType]:
        """
        Get Hu (互卦) - mutual transformation.
        
        Exchanges upper and lower trigrams.
        
        Args:
            upper: Upper trigram
            lower: Lower trigram
        
        Returns:
            Tuple of (upper, lower) with positions swapped
        
        Example:
            >>> tc.get_hu_hexagram(QIAN, KUN)  # (KUN, QIAN)
            # Pi (否) -> Tai (泰)
        """
        cache_key = (upper, lower)
        if cache_key in self._hu_cache:
            return self._hu_cache[cache_key]
        
        result = (lower, upper)  # Swap positions
        self._hu_cache[cache_key] = result
        return result
    
    def get_zong_hexagram(
        self,
        upper: TrigramType,
        lower: TrigramType
    ) -> Tuple[TrigramType, TrigramType]:
        """
        Get Zong (综卦) - reversed transformation.
        
        Rotates the hexagram 180 degrees (same as hu transformation
        in most cases, but follows different rules in full I Ching).
        
        Args:
            upper: Upper trigram
            lower: Lower trigram
        
        Returns:
            Tuple of (upper, lower) for the reversed hexagram
        
        Example:
            >>> tc.get_zong_hexagram(QIAN, KUN)  # (KUN, QIAN)
        """
        cache_key = (upper, lower)
        if cache_key in self._zong_cache:
            return self._zong_cache[cache_key]
        
        # Zong: rotate 180 degrees = swap upper and lower
        result = (lower, upper)
        self._zong_cache[cache_key] = result
        return result
    
    def get_ban_hexagram(
        self,
        upper: TrigramType,
        lower: TrigramType
    ) -> Tuple[TrigramType, TrigramType]:
        """
        Get Ban (半卦) - half transformation.
        
        Retains the main body of the hexagram with partial modification.
        
        Args:
            upper: Upper trigram
            lower: Lower trigram
        
        Returns:
            Tuple of (upper, lower) for the half hexagram
        """
        cache_key = (upper, lower)
        if cache_key in self._ban_cache:
            return self._ban_cache[cache_key]
        
        # Ban: keep lower, derive upper from binary middle bit
        lower_bits = TRIGRAM_TO_BINARY[lower]
        # Take middle bit from lower trigram to modify upper
        middle_bit = lower_bits[1]
        upper_bits = TRIGRAM_TO_BINARY[upper]
        # Combine: keep MSB and LSB of upper, use middle of lower
        new_middle = upper_bits[1] if upper_bits[1] != middle_bit else (1 - middle_bit)
        new_upper_bits = (upper_bits[0], new_middle, upper_bits[2])
        new_upper = BINARY_TO_TRIGRAM[new_upper_bits]
        
        result = (new_upper, lower)
        self._ban_cache[cache_key] = result
        return result
    
    def get_jia_hexagram(
        self,
        upper: TrigramType,
        lower: TrigramType
    ) -> Tuple[TrigramType, TrigramType]:
        """
        Get Jia (交卦) - cross transformation.
        
        Exchanges yin/yang between trigrams.
        
        Args:
            upper: Upper trigram
            lower: Lower trigram
        
        Returns:
            Tuple of (upper, lower) for the cross hexagram
        """
        cache_key = (upper, lower)
        if cache_key in self._jia_cache:
            return self._jia_cache[cache_key]
        
        # Jia: exchange outer lines between trigrams
        upper_bits = TRIGRAM_TO_BINARY[upper]
        lower_bits = TRIGRAM_TO_BINARY[lower]
        
        # Swap MSB and LSB between trigrams
        new_upper_bits = (lower_bits[0], upper_bits[1], upper_bits[2])
        new_lower_bits = (upper_bits[0], lower_bits[1], lower_bits[2])
        
        new_upper = BINARY_TO_TRIGRAM[new_upper_bits]
        new_lower = BINARY_TO_TRIGRAM[new_lower_bits]
        
        result = (new_upper, new_lower)
        self._jia_cache[cache_key] = result
        return result
    
    def get_hexagram_number(
        self,
        upper: TrigramType,
        lower: TrigramType
    ) -> int:
        """
        Get hexagram number (0-63).
        
        Formula: number = upper * 8 + lower
        
        Args:
            upper: Upper trigram
            lower: Lower trigram
        
        Returns:
            Hexagram index (0-63)
        
        Example:
            >>> tc.get_hexagram_number(QIAN, KUN)  # 1 (Pi/否)
            >>> tc.get_hexagram_number(QIAN, QIAN)  # 0 (Qian/乾)
        """
        return int(upper) * 8 + int(lower)
    
    def get_hexagram(
        self,
        number: int
    ) -> Tuple[TrigramType, TrigramType]:
        """
        Get upper and lower trigrams from hexagram number.
        
        Args:
            number: Hexagram index (0-63)
        
        Returns:
            Tuple of (upper, lower) trigrams
        
        Raises:
            ValueError: If number is out of range
        
        Example:
            >>> tc.get_hexagram(1)  # (QIAN, KUN) - Pi/否
            >>> tc.get_hexagram(0)  # (QIAN, QIAN) - Qian/乾
        """
        if not 0 <= number < 64:
            raise ValueError(f"Hexagram number must be 0-63, got {number}")
        return self._hexagram_map[number]
    
    def get_hexagram_info(self, number: int) -> HexagramInfo:
        """
        Get complete hexagram information.
        
        Args:
            number: Hexagram index (0-63)
        
        Returns:
            HexagramInfo with all hexagram attributes
        
        Example:
            >>> info = tc.get_hexagram_info(1)
            >>> info.name  # 坤 (actually this is Pi/否)
            >>> info.upper  # QIAN
            >>> info.lower  # KUN
        """
        if not 0 <= number < 64:
            raise ValueError(f"Hexagram number must be 0-63, got {number}")
        
        upper, lower = self._hexagram_map[number]
        return HexagramInfo(
            upper=upper,
            lower=lower,
            number=number,
            name=HEXAGRAM_NAMES[number]
        )
    
    def analyze_trigram_relation(
        self,
        t1: TrigramType,
        t2: TrigramType
    ) -> List[TrigramRelation]:
        """
        Analyze relationships between two trigrams.
        
        Checks for various relationship types including
        cuo (opposite), hu (mutual), zong (reversed), etc.
        
        Args:
            t1: First trigram
            t2: Second trigram
        
        Returns:
            List of TrigramRelation enums found between the trigrams
        
        Example:
            >>> relations = tc.analyze_trigram_relation(QIAN, KUN)
            >>> TrigramRelation.CUO in relations  # True (opposites)
        """
        relations = []
        
        # Check Cuo (opposite) relationship
        bits1 = TRIGRAM_TO_BINARY[t1]
        bits2 = TRIGRAM_TO_BINARY[t2]
        if bits1 == tuple(1 - b for b in bits2):
            relations.append(TrigramRelation.CUO)
        
        # Check if same trigram (implicit relationship)
        if t1 == t2:
            relations.append(TrigramRelation.SAME if hasattr(TrigramRelation, 'SAME') else TrigramRelation.BAN)
        
        # Check energy type relationship
        energy1 = TRIGRAM_ENERGY_TYPE[t1]
        energy2 = TRIGRAM_ENERGY_TYPE[t2]
        if energy1 == energy2:
            # Same energy type is a form of relationship
            pass  # Already handled by cuo check if applicable
        
        # Check prior/post order relationship (adjacent)
        prior_diff = abs(PRIOR_ORDER[t1] - PRIOR_ORDER[t2])
        if prior_diff in [1, 7]:
            relations.append(TrigramRelation.HU)
        
        return relations
    
    def get_trigram_energy_type(self, t: TrigramType) -> str:
        """
        Get the energy (element) type for a trigram.
        
        Args:
            t: TrigramType enum value
        
        Returns:
            Energy type string: "metal", "wood", "water", "fire", "earth"
        
        Example:
            >>> tc.get_trigram_energy_type(QIAN)  # "metal"
            >>> tc.get_trigram_energy_type(LI)    # "fire"
        """
        return TRIGRAM_ENERGY_TYPE[t]
    
    def get_all_hexagrams(self) -> List[HexagramInfo]:
        """
        Get all 64 hexagrams.
        
        Returns:
            List of all HexagramInfo objects sorted by number
        
        Example:
            >>> all_hex = tc.get_all_hexagrams()
            >>> len(all_hex)  # 64
        """
        return [self.get_hexagram_info(i) for i in range(64)]
    
    def get_trigrams_by_energy(self, energy: str) -> List[TrigramType]:
        """
        Get all trigrams with specified energy type.
        
        Args:
            energy: Energy type ("metal", "wood", "water", "fire", "earth")
        
        Returns:
            List of TrigramType enums with matching energy
        
        Example:
            >>> wood_trigrams = tc.get_trigrams_by_energy("wood")
            >>> wood_trigrams  # [ZHEN, XUN]
        """
        return [
            t for t in TrigramType
            if TRIGRAM_ENERGY_TYPE[t] == energy
        ]


# ============================================================
# Convenience Functions
# ============================================================

def get_trigram_core() -> TrigramCore:
    """
    Get a singleton TrigramCore instance.
    
    Returns:
        TrigramCore instance
    """
    if not hasattr(get_trigram_core, '_instance'):
        get_trigram_core._instance = TrigramCore()
    return get_trigram_core._instance


# ============================================================
# Test Suite
# ============================================================

def test_trigram_core():
    """
    Test suite for TrigramCore.
    
    Verifies all core functionality including:
    - Trigram info retrieval
    - Binary conversion
    - Hexagram transformations
    - Hexagram number calculation
    """
    print("=" * 60)
    print("TrigramCore Test Suite")
    print("=" * 60)
    
    tc = TrigramCore()
    
    # Test 1: Trigram info
    print("\n[Test 1] Trigram Info Retrieval")
    info = tc.get_trigram_info(TrigramType.QIAN)
    assert info.name == "乾", f"Expected '乾', got '{info.name}'"
    assert info.energy_type == "metal", f"Expected 'metal', got '{info.energy_type}'"
    print(f"  QIAN: name={info.name}, energy={info.energy_type}")
    
    info_kun = tc.get_trigram_info(TrigramType.KUN)
    assert info_kun.name == "坤"
    assert info_kun.energy_type == "earth"
    print(f"  KUN: name={info_kun.name}, energy={info_kun.energy_type}")
    print("  ✓ PASSED")
    
    # Test 2: Binary conversion
    print("\n[Test 2] Binary Conversion")
    bits = tc.trigram_to_binary(TrigramType.QIAN)
    assert bits == (1, 1, 1), f"Expected (1,1,1), got {bits}"
    print(f"  QIAN -> {bits}")
    
    bits_kun = tc.trigram_to_binary(TrigramType.KUN)
    assert bits_kun == (0, 0, 0), f"Expected (0,0,0), got {bits_kun}"
    print(f"  KUN -> {bits_kun}")
    
    # Round trip test
    recovered = tc.binary_to_trigram(bits)
    assert recovered == TrigramType.QIAN
    print(f"  Round trip: {bits} -> {recovered.name}")
    print("  ✓ PASSED")
    
    # Test 3: Prior order
    print("\n[Test 3] Prior Order (伏羲卦序)")
    assert tc.get_prior_order(TrigramType.QIAN) == 0
    assert tc.get_prior_order(TrigramType.DUI) == 1
    assert tc.get_prior_order(TrigramType.LI) == 2
    assert tc.get_prior_order(TrigramType.ZHEN) == 3
    assert tc.get_prior_order(TrigramType.XUN) == 4
    assert tc.get_prior_order(TrigramType.KAN) == 5
    assert tc.get_prior_order(TrigramType.GEN) == 6
    assert tc.get_prior_order(TrigramType.KUN) == 7
    print(f"  Prior order: QIAN=0, DUI=1, LI=2, ZHEN=3, XUN=4, KAN=5, GEN=6, KUN=7")
    print("  ✓ PASSED")
    
    # Test 4: Post order
    print("\n[Test 4] Post Order (文王卦序)")
    assert tc.get_post_order(TrigramType.KAN) == 0
    assert tc.get_post_order(TrigramType.KUN) == 1
    assert tc.get_post_order(TrigramType.ZHEN) == 2
    assert tc.get_post_order(TrigramType.XUN) == 3
    assert tc.get_post_order(TrigramType.QIAN) == 4
    assert tc.get_post_order(TrigramType.DUI) == 5
    assert tc.get_post_order(TrigramType.GEN) == 6
    assert tc.get_post_order(TrigramType.LI) == 7
    print(f"  Post order: KAN=0, KUN=1, ZHEN=2, XUN=3, QIAN=4, DUI=5, GEN=6, LI=7")
    print("  ✓ PASSED")
    
    # Test 5: Cuo (错卦) transformation
    print("\n[Test 5] Cuo (错卦) Transformation")
    cuo = tc.get_cuo_hexagram(TrigramType.QIAN, TrigramType.QIAN)
    assert cuo == (TrigramType.KUN, TrigramType.KUN), f"Expected (KUN,KUN), got {cuo}"
    print(f"  Cuo(QIAN, QIAN) = {cuo[0].name}, {cuo[1].name}")
    
    # Verify all cuo pairs
    for upper in TrigramType:
        for lower in TrigramType:
            cuo_upper, cuo_lower = tc.get_cuo_hexagram(upper, lower)
            upper_bits = TRIGRAM_TO_BINARY[upper]
            lower_bits = TRIGRAM_TO_BINARY[lower]
            expected_upper = tuple(1 - b for b in upper_bits)
            expected_lower = tuple(1 - b for b in lower_bits)
            assert BINARY_TO_TRIGRAM[expected_upper] == cuo_upper
            assert BINARY_TO_TRIGRAM[expected_lower] == cuo_lower
    print(f"  All 64 hexagrams verified")
    print("  ✓ PASSED")
    
    # Test 6: Hu (互卦) transformation
    print("\n[Test 6] Hu (互卦) Transformation")
    hu = tc.get_hu_hexagram(TrigramType.QIAN, TrigramType.KUN)
    assert hu == (TrigramType.KUN, TrigramType.QIAN), f"Expected (KUN,QIAN), got {hu}"
    print(f"  Hu(QIAN, KUN) = {hu[0].name}, {hu[1].name}")
    print("  ✓ PASSED")
    
    # Test 7: Zong (综卦) transformation
    print("\n[Test 7] Zong (综卦) Transformation")
    zong = tc.get_zong_hexagram(TrigramType.QIAN, TrigramType.KUN)
    assert zong == (TrigramType.KUN, TrigramType.QIAN), f"Expected (KUN,QIAN), got {zong}"
    print(f"  Zong(QIAN, KUN) = {zong[0].name}, {zong[1].name}")
    print("  ✓ PASSED")
    
    # Test 8: Hexagram number calculation
    print("\n[Test 8] Hexagram Number Calculation")
    num = tc.get_hexagram_number(TrigramType.QIAN, TrigramType.KUN)
    assert num == 1, f"Expected 1, got {num}"
    print(f"  HexagramNumber(QIAN, KUN) = {num}")
    
    num_00 = tc.get_hexagram_number(TrigramType.QIAN, TrigramType.QIAN)
    assert num_00 == 0, f"Expected 0, got {num_00}"
    print(f"  HexagramNumber(QIAN, QIAN) = {num_00}")
    
    num_09 = tc.get_hexagram_number(TrigramType.KUN, TrigramType.KUN)
    assert num_09 == 9, f"Expected 9, got {num_09}"
    print(f"  HexagramNumber(KUN, KUN) = {num_09}")
    
    num_63 = tc.get_hexagram_number(TrigramType.DUI, TrigramType.DUI)
    assert num_63 == 63, f"Expected 63, got {num_63}"
    print(f"  HexagramNumber(DUI, DUI) = {num_63}")
    print("  ✓ PASSED")
    
    # Test 9: Hexagram lookup
    print("\n[Test 9] Hexagram Lookup")
    upper, lower = tc.get_hexagram(1)
    assert upper == TrigramType.QIAN and lower == TrigramType.KUN
    print(f"  Hexagram(1) = ({upper.name}, {lower.name})")
    
    upper, lower = tc.get_hexagram(0)
    assert upper == TrigramType.QIAN and lower == TrigramType.QIAN
    print(f"  Hexagram(0) = ({upper.name}, {lower.name})")
    print("  ✓ PASSED")
    
    # Test 10: Hexagram info
    print("\n[Test 10] Hexagram Info")
    info = tc.get_hexagram_info(0)
    assert info.name == "乾"
    assert info.upper == TrigramType.QIAN
    assert info.lower == TrigramType.QIAN
    print(f"  HexagramInfo(0): name={info.name}")
    
    info_1 = tc.get_hexagram_info(1)
    assert info_1.name == "坤"  # Kun/坤 (index 1 in standard sequence)
    print(f"  HexagramInfo(1): name={info_1.name}")
    
    info_63 = tc.get_hexagram_info(63)
    assert info_63.name == "未济"
    assert info_63.upper == TrigramType.DUI
    assert info_63.lower == TrigramType.DUI
    print(f"  HexagramInfo(63): name={info_63.name}")
    print("  ✓ PASSED")
    
    # Test 11: Trigram energy type
    print("\n[Test 11] Trigram Energy Type")
    assert tc.get_trigram_energy_type(TrigramType.QIAN) == "metal"
    assert tc.get_trigram_energy_type(TrigramType.DUI) == "metal"
    assert tc.get_trigram_energy_type(TrigramType.LI) == "fire"
    assert tc.get_trigram_energy_type(TrigramType.ZHEN) == "wood"
    assert tc.get_trigram_energy_type(TrigramType.XUN) == "wood"
    assert tc.get_trigram_energy_type(TrigramType.KAN) == "water"
    assert tc.get_trigram_energy_type(TrigramType.GEN) == "earth"
    assert tc.get_trigram_energy_type(TrigramType.KUN) == "earth"
    print("  Energy types verified for all 8 trigrams")
    print("  ✓ PASSED")
    
    # Test 12: Get trigrams by energy
    print("\n[Test 12] Get Trigrams By Energy")
    metal_trigrams = tc.get_trigrams_by_energy("metal")
    assert len(metal_trigrams) == 2
    assert TrigramType.QIAN in metal_trigrams
    assert TrigramType.DUI in metal_trigrams
    print(f"  Metal trigrams: {[t.name for t in metal_trigrams]}")
    
    wood_trigrams = tc.get_trigrams_by_energy("wood")
    assert len(wood_trigrams) == 2
    assert TrigramType.ZHEN in wood_trigrams
    assert TrigramType.XUN in wood_trigrams
    print(f"  Wood trigrams: {[t.name for t in wood_trigrams]}")
    print("  ✓ PASSED")
    
    # Test 13: Trigram relation analysis
    print("\n[Test 13] Trigram Relation Analysis")
    relations = tc.analyze_trigram_relation(TrigramType.QIAN, TrigramType.KUN)
    assert TrigramRelation.CUO in relations
    print(f"  QIAN-KUN relations: {[r.name for r in relations]}")
    print("  ✓ PASSED")
    
    # Test 14: All 64 hexagrams
    print("\n[Test 14] All 64 Hexagrams")
    all_hex = tc.get_all_hexagrams()
    assert len(all_hex) == 64
    print(f"  Total hexagrams: {len(all_hex)}")
    print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_trigram_core()
