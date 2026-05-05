"""
SemanticFramework Module - Dynamic Transformation Rule System

Corresponds to the "Transformation Layer" in the four-dimensional system
Uses 64-pattern system with 384 components as carriers
Reveals the complete rules from origin to transformation

Core: Three Principles (constancy / transformation / simplification)
      Primary/Internal/Resulting pattern system
      Primary/Response/Active component system
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# Eight Trigram Base Types
# ============================================================

class SemanticType(Enum):
    """Semantic system type definitions"""
    CAT_CREATIVE = 0   # Creative/Generative
    CAT_LAKE = 1       # Lake/Open
    CAT_LIGHT = 2      # Light/Illumination
    CAT_THUNDER = 3    # Thunder/Activation
    CAT_WIND = 4       # Wind/Penetration
    CAT_ABYSS = 5      # Abyss/Depth
    CAT_MOUNTAIN = 6   # Mountain/Stillness
    CAT_RECEPTIVE = 7  # Receptive/Sustaining

    @property
    def name_en(self) -> str:
        return ["Creative", "Lake", "Light", "Thunder", "Wind", "Abyss", "Mountain", "Receptive"][self.value]

    @property
    def energy_type(self) -> str:
        return ["metal", "metal", "fire", "wood", "wood", "water", "earth", "earth"][self.value]

    @property
    def direction(self) -> str:
        return ["NW", "W", "S", "E", "SE", "N", "NE", "SW"][self.value]

    @property
    def characteristic(self) -> str:
        return ["strength", "joy", "clarity", "movement", "infiltration", "danger", "stillness", "receptivity"][self.value]

    @property
    def enhance(self) -> 'SemanticType':
        """Energy enhancement relationships"""
        mapping = {0: 2, 2: 5, 5: 7, 7: 3, 3: 0, 1: 0, 4: 3, 6: 5}
        return SemanticType(mapping.get(self.value, 0))

    @property
    def suppress(self) -> 'SemanticType':
        """Energy suppression relationships"""
        mapping = {0: 4, 2: 1, 5: 0, 3: 7, 7: 5, 4: 2, 6: 0, 1: 6}
        return SemanticType(mapping.get(self.value, 0))


# Prior trigram mapping (ontological positioning)
PRIOR_SEMANTIC_MAP = {
    SemanticType.CAT_CREATIVE: {"position": "S", "polarity": "yang"},
    SemanticType.CAT_RECEPTIVE: {"position": "N", "polarity": "yin"},
    SemanticType.CAT_LIGHT: {"position": "E", "polarity": "yin"},
    SemanticType.CAT_ABYSS: {"position": "W", "polarity": "yang"},
    SemanticType.CAT_THUNDER: {"position": "NE", "polarity": "yang"},
    SemanticType.CAT_WIND: {"position": "SW", "polarity": "yin"},
    SemanticType.CAT_MOUNTAIN: {"position": "NW", "polarity": "yang"},
    SemanticType.CAT_LAKE: {"position": "SE", "polarity": "yin"},
}

# Post trigram mapping (temporal-spatial application)
POST_SEMANTIC_MAP = {
    SemanticType.CAT_THUNDER: {"position": "E", "season": "Spring", "month": "2-3"},
    SemanticType.CAT_WIND: {"position": "SE", "season": "LateSpring", "month": "3-4"},
    SemanticType.CAT_LIGHT: {"position": "S", "season": "Summer", "month": "5-6"},
    SemanticType.CAT_LAKE: {"position": "W", "season": "Autumn", "month": "8-9"},
    SemanticType.CAT_CREATIVE: {"position": "NW", "season": "LateAutumn", "month": "9-10"},
    SemanticType.CAT_MOUNTAIN: {"position": "NE", "season": "WinterSpring", "month": "12-1"},
    SemanticType.CAT_RECEPTIVE: {"position": "SW", "season": "LateSummer", "month": "6-7"},
    SemanticType.CAT_ABYSS: {"position": "N", "season": "Winter", "month": "11-12"},
}


# ============================================================
# 64-Pattern Table (Simplified)
# ============================================================

PATTERN_NAMES = [
    "Qian", "Kun", "Zhun", "Meng", "Xu", "Song", "Shi", "Bi",
    "XiaoXu", "Lv", "Tai", "Pi", "TongRen", "DaYou", "Qian", "Yu",
    "Sui", "Gu", "Lin", "Guan", "ShiKe", "Bi", "Bo", "Fu",
    "WuWang", "DaXu", "Yi", "DaGuo", "Kan", "Li", "Xian", "Heng",
    "Dun", "DaZhuang", "Jin", "MingYi", "JiaRen", "Kui", "Jian", "Xie",
    "Sun", "Yi", "Guai", "Gou", "Cui", "Sheng", "Kun", "Jing",
    "Ge", "Ding", "Zhen", "Gen", "Jian", "GuiMei", "Feng", "Lv",
    "Xun", "Dui", "Huan", "Jie", "ZhongFu", "XiaoGuo", "JiJi", "WeiJi"
]

# Complete 64-pattern upper/lower mapping (from encoders.py)
from .encoders import HEXAGRAM_TRIGRAMS_BELOW, HEXAGRAM_TRIGRAMS_ABOVE

# Three-dimensional calculus resolver for TrigramType→SemanticType mapping
# Uses integration across NAJIA, PRIOR, POST dimensions with weighted voting
# Replaces direct SemanticType() cast which had 25% accuracy (only indices 0,6 matched)
_TRIGRAM_RESOLVER = None

def _get_trigram_resolver():
    """Lazy-load the three-dimensional resolver to avoid circular imports."""
    global _TRIGRAM_RESOLVER
    if _TRIGRAM_RESOLVER is None:
        from ._dimension_map import TaijiMapper
        _TRIGRAM_RESOLVER = TaijiMapper()
    return _TRIGRAM_RESOLVER

def _build_hexagram_trigrams():
    """Build complete 64-pattern upper/lower mapping using 3D calculus fusion."""
    result = []
    for i in range(64):
        # Resolve via three-dimensional weighted voting (integration/differentiation/gradient)
        upper_result = _get_trigram_resolver().resolve_trigram_to_semantic(HEXAGRAM_TRIGRAMS_ABOVE[i])
        lower_result = _get_trigram_resolver().resolve_trigram_to_semantic(HEXAGRAM_TRIGRAMS_BELOW[i])
        upper = SemanticType(upper_result.primary) if upper_result.primary is not None else SemanticType.CAT_CREATIVE
        lower = SemanticType(lower_result.primary) if lower_result.primary is not None else SemanticType.CAT_CREATIVE
        result.append((upper, lower))
    return result

HEXAGRAM_TRIGRAMS = _build_hexagram_trigrams()


# ============================================================
# Component System
# ============================================================

@dataclass
class ComponentPosition:
    """Component position information"""
    index: int
    name: str
    polarity: str
    dignity: str
    position_nature: str


class Pattern:
    """64-pattern object"""

    def __init__(self, number: int, upper: SemanticType, lower: SemanticType,
                 pattern_name: str = ""):
        self.number = number
        self.upper = upper
        self.lower = lower
        self.name = pattern_name or PATTERN_NAMES[number] if number < 64 else "Unknown"
        self.energy_type = upper.energy_type  # Upper pattern energy type

    @property
    def pattern_representation(self) -> str:
        return f"{self.lower.name_en}/{self.upper.name_en}"

    @property
    def pattern_type(self) -> str:
        """Pattern classification"""
        if self.upper == SemanticType.CAT_CREATIVE or self.lower == SemanticType.CAT_CREATIVE:
            return "CreativeSystem"
        if self.upper == SemanticType.CAT_RECEPTIVE or self.lower == SemanticType.CAT_RECEPTIVE:
            return "ReceptiveSystem"
        return "Other"

    def get_polarity_string(self) -> str:
        """Get pattern polarity sequence (bottom to top)"""
        base = self.number
        result = ""
        for i in range(6):
            result += "yang" if (base + i) % 2 == 0 else "yin"
        return result

    def get_base_info(self) -> dict:
        return {
            "number": self.number,
            "name": self.name,
            "pattern": self.pattern_representation,
            "upper": self.upper.name_en,
            "lower": self.lower.name_en,
            "energy_type": self.energy_type,
            "upper_energy": self.upper.energy_type,
            "lower_energy": self.lower.energy_type,
            "type": self.pattern_type,
            "prior_position": PRIOR_SEMANTIC_MAP.get(self.lower, {}).get("position", ""),
            "post_position": POST_SEMANTIC_MAP.get(self.lower, {}).get("position", ""),
        }


# Quick pattern creation
def create_pattern(upper_idx: int, lower_idx: int) -> Pattern:
    """Create pattern by upper/lower index"""
    upper = SemanticType(upper_idx % 8)
    lower = SemanticType(lower_idx % 8)
    number = upper_idx * 8 + lower_idx
    return Pattern(number, upper, lower)


# Hexagram heavenly stem assignment
def get_heavenly_stem(pattern: Pattern) -> str:
    """Get pattern's heavenly stem assignment"""
    stem_map = {
        SemanticType.CAT_CREATIVE: "Jia", SemanticType.CAT_LAKE: "Ding",
        SemanticType.CAT_LIGHT: "Ji", SemanticType.CAT_THUNDER: "Geng",
        SemanticType.CAT_WIND: "Xin", SemanticType.CAT_ABYSS: "Wu",
        SemanticType.CAT_MOUNTAIN: "Bing", SemanticType.CAT_RECEPTIVE: "Gui",
    }
    return stem_map.get(pattern.upper, "Jia")


@dataclass
class TrigramInfo:
    """Internal pattern information - inherent development"""
    upper: SemanticType
    lower: SemanticType
    name: str


# ============================================================
# Three Principles Encoding
# ============================================================

class FrameworkRule:
    """Three principles - constancy/transformation/simplification"""

    @staticmethod
    def constancy() -> str:
        """Constancy - invariant rules"""
        return "Energy enhancement/suppression, polarity matching, and hierarchy mapping remain constant"

    @staticmethod
    def transformation(energy_state: str, component_active: bool) -> str:
        """Transformation - dynamic changes"""
        if component_active:
            return f"Active component triggered, {energy_state} field transformation"
        return f"{energy_state} energy flow in progress"

    @staticmethod
    def simplification(core_energy: str) -> str:
        """Simplification - complexity reduction"""
        return f"Capture core {core_energy} essence, grasp the essential"


@dataclass
class MemoryPattern:
    """Memory's semantic framework annotation"""
    pattern: Pattern
    primary_yao: int
    responding_yao: int
    active_components: List[int]
    current_pattern: Pattern
    internal_pattern: Optional[Pattern]
    resulting_pattern: Optional[Pattern]

    def get_trend(self) -> str:
        """Get transformation trend"""
        if self.internal_pattern and self.resulting_pattern:
            return f"{self.current_pattern.name}->{self.internal_pattern.name}->{self.resulting_pattern.name}"
        return self.current_pattern.name

    def get_state(self) -> str:
        """Get current state"""
        return f"{self.pattern.energy_type} energy {'active' if self.active_components else 'stable'}"


# ============================================================
# Primary/Response Component Calculation
# ============================================================

def _trigram_to_bits(t: SemanticType) -> List[int]:
    """Trigram to 3-bit binary (prior sequence)"""
    bits_map = {
        SemanticType.CAT_CREATIVE: [1,1,1],
        SemanticType.CAT_LAKE:  [0,1,1],
        SemanticType.CAT_LIGHT:   [1,0,1],
        SemanticType.CAT_THUNDER: [0,0,1],
        SemanticType.CAT_WIND:  [1,1,0],
        SemanticType.CAT_ABYSS:  [0,1,0],
        SemanticType.CAT_MOUNTAIN:  [1,0,0],
        SemanticType.CAT_RECEPTIVE:  [0,0,0],
    }
    return bits_map.get(t, [0,0,0])


def compute_shi_ying(upper: SemanticType, lower: SemanticType) -> Tuple[int, int]:
    """
    Calculate primary and responding component positions

    Jingfang Eight Palace rules (simplified):
    - Same upper/lower (pure patterns) -> primary=6, response=3
    - Different upper/lower -> determine primary based on binary differences
    - Primary and response are 3 positions apart

    Returns:
        (primary_position, responding_position)  # 1-6
    """
    if upper == lower:
        return (6, 3)

    upper_bits = _trigram_to_bits(upper)
    lower_bits = _trigram_to_bits(lower)

    # Find first different bit from bottom
    diff_pos = 1
    for i in range(3):
        if upper_bits[i] != lower_bits[i]:
            diff_pos = i + 1
            break

    shi = max(1, min(6, diff_pos))
    ying = shi + 3 if shi <= 3 else shi - 3
    return (shi, ying)


def predict_active_components(pattern_index: int, query_context: str = "") -> List[int]:
    """
    Predict active component positions

    Rules:
    1. Primary component is active (core change)
    2. Polarity boundary is active (upper/lower junction, positions 3-4)
    3. If specific semantics present (e.g., "change", "sudden"), add first component

    Returns:
        Active component positions [1-6]
    """
    if pattern_index < 0 or pattern_index >= 64:
        return [1]

    upper, lower = HEXAGRAM_TRIGRAMS[pattern_index]
    shi, _ = compute_shi_ying(upper, lower)

    active_components = [shi]

    # Upper/lower junction (positions 3-4)
    if shi not in (3, 4):
        active_components.append(3)

    # Semantic triggers
    change_keywords = ["change", "sudden", "transform", "break", "intense", "rapid", "new"]
    if query_context and any(kw in query_context for kw in change_keywords):
        if 1 not in active_components:
            active_components.append(1)

    return sorted(set(active_components))


def _get_yao_line(pattern_index: int, yao_pos: int) -> int:
    """Get polarity value of component at position (0=yin, 1=yang)"""
    if pattern_index < 0 or pattern_index >= 64:
        return 0
    upper, lower = HEXAGRAM_TRIGRAMS[pattern_index]
    upper_bits = _trigram_to_bits(upper)
    lower_bits = _trigram_to_bits(lower)
    # Positions 1-3 map to lower, 4-6 map to upper
    all_bits = lower_bits + upper_bits  # [1,2,3,4,5,6]
    if 1 <= yao_pos <= 6:
        return all_bits[yao_pos - 1]
    return 0


def _bits_to_trigram(bits: List[int]) -> SemanticType:
    """3-bit binary to trigram"""
    bits_map = {
        (1,1,1): SemanticType.CAT_CREATIVE,
        (0,1,1): SemanticType.CAT_LAKE,
        (1,0,1): SemanticType.CAT_LIGHT,
        (0,0,1): SemanticType.CAT_THUNDER,
        (1,1,0): SemanticType.CAT_WIND,
        (0,1,0): SemanticType.CAT_ABYSS,
        (1,0,0): SemanticType.CAT_MOUNTAIN,
        (0,0,0): SemanticType.CAT_RECEPTIVE,
    }
    return bits_map.get(tuple(bits), SemanticType.CAT_RECEPTIVE)


def _find_pattern_by_trigrams(upper: SemanticType, lower: SemanticType) -> int:
    """Find 64-pattern index by upper/lower"""
    for i in range(64):
        if HEXAGRAM_TRIGRAMS[i] == (upper, lower):
            return i
    return 0


def _compute_internal_trigrams(pattern_index: int) -> Tuple[SemanticType, SemanticType]:
    """Compute internal pattern (positions 2-3-4 as lower, 3-4-5 as upper)"""
    upper, lower = HEXAGRAM_TRIGRAMS[pattern_index]
    upper_bits = _trigram_to_bits(upper)
    lower_bits = _trigram_to_bits(lower)
    all_bits = lower_bits + upper_bits  # [comp1, comp2, comp3, comp4, comp5, comp6]

    internal_lower_bits = [all_bits[1], all_bits[2], all_bits[3]]  # 2-3-4
    internal_upper_bits = [all_bits[2], all_bits[3], all_bits[4]]  # 3-4-5

    internal_lower = _bits_to_trigram(internal_lower_bits)
    internal_upper = _bits_to_trigram(internal_upper_bits)
    return (internal_upper, internal_lower)


def _compute_resulting_pattern(pattern_index: int, active_components: List[int]) -> int:
    """Compute resulting pattern (polarity swap of active components)"""
    if not active_components:
        return pattern_index

    upper, lower = HEXAGRAM_TRIGRAMS[pattern_index]
    upper_bits = _trigram_to_bits(upper)
    lower_bits = _trigram_to_bits(lower)
    all_bits = lower_bits + upper_bits

    for pos in active_components:
        if 1 <= pos <= 6:
            all_bits[pos - 1] = 1 - all_bits[pos - 1]

    new_lower = _bits_to_trigram(all_bits[0:3])
    new_upper = _bits_to_trigram(all_bits[3:6])
    return _find_pattern_by_trigrams(new_upper, new_lower)


# ============================================================
# SemanticFramework Three-Layer Inference Engine
# ============================================================

class PatternInference:
    """
    Three-layer inference engine

    Current Pattern -> Internal Pattern -> Resulting Pattern
    """

    def __init__(self):
        pass

    def create_memory_pattern(self, pattern_index: int, content: str = "") -> MemoryPattern:
        """
        Create complete semantic framework annotation for a memory
        """
        idx = pattern_index % 64
        upper, lower = HEXAGRAM_TRIGRAMS[idx]
        current_pattern = Pattern(idx, upper, lower)

        # Primary/Response components
        shi, ying = compute_shi_ying(upper, lower)

        # Active components
        active = predict_active_components(idx, content)

        # Internal pattern
        internal_upper, internal_lower = _compute_internal_trigrams(idx)
        internal_idx = _find_pattern_by_trigrams(internal_upper, internal_lower)
        internal_pattern = Pattern(internal_idx, internal_upper, internal_lower)

        # Resulting pattern
        resulting_idx = _compute_resulting_pattern(idx, active)
        resulting_upper, resulting_lower = HEXAGRAM_TRIGRAMS[resulting_idx]
        resulting_pattern = Pattern(resulting_idx, resulting_upper, resulting_lower)

        return MemoryPattern(
            pattern=current_pattern,
            primary_yao=shi,
            responding_yao=ying,
            active_components=active,
            current_pattern=current_pattern,
            internal_pattern=internal_pattern,
            resulting_pattern=resulting_pattern,
        )

    def three_layer_retrieve(self, query_index: int,
                              candidate_indices: List[int],
                              top_k: int = 8) -> List[Dict]:
        """
        Three-layer inference retrieval

        1. Current Layer: Direct match of candidate's current pattern
        2. Internal Layer: Candidate's internal = Query's current (inherent relation)
        3. Resulting Layer: Candidate's resulting = Query's current (temporal relation)

        Layer weights: Current 0.5, Internal 0.3, Resulting 0.2
        """
        query_idx = query_index % 64
        query_my = self.create_memory_pattern(query_idx)

        results = []
        for cand_idx in candidate_indices:
            cidx = cand_idx % 64
            cand_my = self.create_memory_pattern(cidx)

            # Current layer: direct match
            current_score = 1.0 if cidx == query_idx else 0.0
            # Partial match: share upper or lower
            if current_score == 0:
                q_upper, q_lower = HEXAGRAM_TRIGRAMS[query_idx]
                c_upper, c_lower = HEXAGRAM_TRIGRAMS[cidx]
                if q_upper == c_upper:
                    current_score = 0.4
                elif q_lower == c_lower:
                    current_score = 0.4
                # Same energy type
                elif query_my.pattern.energy_type == cand_my.pattern.energy_type:
                    current_score = 0.2

            # Internal layer: candidate's internal = query's current (or reverse)
            internal_score = 0.0
            if cand_my.internal_pattern and cand_my.internal_pattern.number == query_idx:
                internal_score = 1.0
            elif query_my.internal_pattern and query_my.internal_pattern.number == cidx:
                internal_score = 0.8
            elif cand_my.internal_pattern and query_my.internal_pattern and cand_my.internal_pattern.number == query_my.internal_pattern.number:
                internal_score = 0.5

            # Resulting layer: candidate's resulting = query's current (or reverse)
            resulting_score = 0.0
            if cand_my.resulting_pattern and cand_my.resulting_pattern.number == query_idx:
                resulting_score = 1.0
            elif query_my.resulting_pattern and query_my.resulting_pattern.number == cidx:
                resulting_score = 0.8
            elif cand_my.resulting_pattern and query_my.resulting_pattern and cand_my.resulting_pattern.number == query_my.resulting_pattern.number:
                resulting_score = 0.5

            total = 0.5 * current_score + 0.3 * internal_score + 0.2 * resulting_score

            trend = cand_my.current_pattern.name
            if cand_my.internal_pattern:
                trend += f"->{cand_my.internal_pattern.name}"
            if cand_my.resulting_pattern:
                trend += f"->{cand_my.resulting_pattern.name}"

            results.append({
                "index": cidx,
                "score": round(total, 4),
                "layer_scores": {
                    "current": round(current_score, 4),
                    "internal": round(internal_score, 4),
                    "resulting": round(resulting_score, 4),
                },
                "trend": trend,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_trend_analysis(self, memory_pattern: MemoryPattern) -> Dict:
        """
        Get memory's trend analysis
        """
        my = memory_pattern
        current = {"name": my.current_pattern.name, "energy_type": my.current_pattern.energy_type,
                    "type": my.current_pattern.pattern_type}

        internal = None
        if my.internal_pattern:
            internal = {"name": my.internal_pattern.name, "energy_type": my.internal_pattern.energy_type,
                        "type": my.internal_pattern.pattern_type}

        future = None
        if my.resulting_pattern:
            future = {"name": my.resulting_pattern.name, "energy_type": my.resulting_pattern.energy_type,
                       "type": my.resulting_pattern.pattern_type}

        state = my.get_state()

        # Primary/Response relation recommendations
        if my.primary_yao == 6:
            recommendation = "Pure pattern, maximum energy, suitable as core memory"
        elif my.primary_yao <= 3:
            recommendation = "Primary in lower pattern, suitable as foundational memory"
        else:
            recommendation = "Primary in upper pattern, suitable as applied memory"

        return {
            "current": current,
            "internal": internal,
            "future": future,
            "state": state,
            "recommendation": recommendation,
            "trend_path": my.get_trend(),
        }
