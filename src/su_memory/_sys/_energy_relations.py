"""
Energy Relations Module - Five Elements Enhance and Suppress System

This module implements the core logic for Five Elements (Wu Xing) relationships:
- Enhance (相生): Wood -> Fire -> Earth -> Metal -> Water -> Wood
- Suppress (相克): Wood -> Earth -> Water -> Fire -> Metal -> Wood

核心演化脉络: 无极→太极→两仪(阴阳)→三才→四象→五行→八卦→天干地支

【后天主象】- 五行生克关系以后天八卦象征意义为基础
- 五行能量流转遵循后天八卦的时空顺序
- 方位对应：坎北水、坤西南土、震东木、巽东南木、乾西北金、兑西金、艮东北土、离南火
- 应用场景：时空索引、能量传播、五行养生、方位调理

【四象体系】
- 少阳 -> 木 -> 春 -> 生发
- 太阳 -> 火 -> 夏 -> 炎盛
- 少阴 -> 金 -> 秋 -> 收敛
- 太阴 -> 水 -> 冬 -> 闭藏
- 中宫 -> 土 -> 长夏 -> 化育

Modern Terminology Mapping:
- Enhance -> EnergyEnhance / ShengRelationship
- Suppress -> EnergySuppress / KeRelationship
- Overconstraint -> OverConstraint / ChengRelationship
- Reverse -> ReverseConstraint / WuRelationship

All external APIs use modern technical terms while maintaining internal philosophical logic.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# ============================================================
# Five Elements Energy Types (现代化术语)
# ============================================================

class EnergyType(Enum):
    """Five Elements energy types"""
    WOOD = "wood"      # 木
    FIRE = "fire"      # 火
    EARTH = "earth"    # 土
    METAL = "metal"    # 金
    WATER = "water"     # 水
    
    @classmethod
    def from_string(cls, value: str) -> 'EnergyType':
        """Create from string value"""
        value = value.lower().strip()
        for e in cls:
            if e.value == value:
                return e
        raise ValueError(f"Unknown energy type: {value}")
    
    @classmethod
    def all_values(cls) -> List[str]:
        """Get all energy type values"""
        return [e.value for e in cls]


# ============================================================
# Energy Relation Types (现代化术语)
# ============================================================

class RelationType(Enum):
    """Energy relation types"""
    ENHANCE = "enhance"          # 相生
    SUPPRESS = "suppress"         # 相克
    OVERCONSTRAINT = "overconstraint"  # 相乘
    REVERSE = "reverse"          # 相侮
    SAME = "same"                 # 同类
    NEUTRAL = "neutral"          # 中性


# ============================================================
# Energy Enhance Mapping (相生: 木生火、火生土、土生金、金生水、水生木)
# ============================================================

# Forward enhance: element -> what it generates
ENERGY_ENHANCE: Dict[str, str] = {
    "wood": "fire",      # 木生火
    "fire": "earth",     # 火生土
    "earth": "metal",    # 土生金
    "metal": "water",    # 金生水
    "water": "wood",     # 水生木
}

# Reverse enhance: element -> what generates it
ENERGY_ENHANCED_BY: Dict[str, str] = {
    "fire": "wood",      # 火被木生
    "earth": "fire",     # 土被火生
    "metal": "earth",    # 金被土生
    "water": "metal",    # 水被金生
    "wood": "water",     # 木被水生
}


# ============================================================
# Four Symbols Mapping (四象对应)
# ============================================================

# Energy to Four Symbols mapping
ENERGY_TO_FOUR_SYMBOLS: Dict[str, str] = {
    "wood": "SHAO_YANG",    # 木 -> 少阳 (春)
    "fire": "TAI_YANG",     # 火 -> 太阳 (夏)
    "earth": "CENTER",      # 土 -> 中宫 (长夏)
    "metal": "SHAO_YIN",    # 金 -> 少阴 (秋)
    "water": "TAI_YIN",     # 水 -> 太阴 (冬)
}

# Four Symbols to Energy mapping
FOUR_SYMBOLS_TO_ENERGY: Dict[str, str] = {
    "SHAO_YANG": "wood",   # 少阳 -> 木
    "TAI_YANG": "fire",    # 太阳 -> 火
    "CENTER": "earth",     # 中宫 -> 土
    "SHAO_YIN": "metal",   # 少阴 -> 金
    "TAI_YIN": "water",    # 太阴 -> 水
}

# Four Symbols to Season mapping
FOUR_SYMBOLS_TO_SEASON: Dict[str, str] = {
    "SHAO_YANG": "spring",     # 少阳 -> 春
    "TAI_YANG": "summer",      # 太阳 -> 夏
    "CENTER": "late_summer",    # 中宫 -> 长夏
    "SHAO_YIN": "autumn",      # 少阴 -> 秋
    "TAI_YIN": "winter",       # 太阴 -> 冬
}

# Season to Energy mapping (季节能量)
SEASON_ENERGY_MAP: Dict[str, str] = {
    "spring": "wood",      # 春 -> 木
    "summer": "fire",      # 夏 -> 火
    "late_summer": "earth", # 长夏 -> 土
    "autumn": "metal",     # 秋 -> 金
    "winter": "water",     # 冬 -> 水
}


# ============================================================
# Energy Suppress Mapping (相克: 木克土、土克水、水克火、火克金、金克木)
# ============================================================

# Forward suppress: element -> what it controls
ENERGY_SUPPRESS: Dict[str, str] = {
    "wood": "earth",     # 木克土
    "earth": "water",    # 土克水
    "water": "fire",     # 水克火
    "fire": "metal",     # 火克金
    "metal": "wood",     # 金克木
}

# Reverse suppress: element -> what controls it
ENERGY_SUPPRESSED_BY: Dict[str, str] = {
    "earth": "wood",     # 土被木克
    "water": "earth",    # 水被土克
    "fire": "water",     # 火被水克
    "metal": "fire",     # 金被火克
    "wood": "metal",     # 木被金克
}


# ============================================================
# Energy Relation Strength (关系强度)
# ============================================================

# Strength multipliers for different relations
RELATION_STRENGTH = {
    RelationType.ENHANCE: 1.2,        # 相生增强 20%
    RelationType.SUPPRESS: 0.8,       # 相克削弱 20%
    RelationType.OVERCONSTRAINT: 0.6, # 相乘削弱 40%
    RelationType.REVERSE: 0.4,        # 相侮削弱 60%
    RelationType.SAME: 1.1,          # 同类增强 10%
    RelationType.NEUTRAL: 1.0,       # 中性不变
}


# ============================================================
# Data Classes
# ============================================================

@dataclass
class EnergyRelation:
    """Energy relation result"""
    source: str           # Source element
    target: str           # Target element
    relation: RelationType  # Relation type
    strength: float       # Relation strength (0.0 - 2.0)
    description: str      # Human-readable description
    
    @property
    def is_enhancing(self) -> bool:
        """Check if this is an enhancing relationship"""
        return self.relation == RelationType.ENHANCE
    
    @property
    def is_suppressing(self) -> bool:
        """Check if this is a suppressing relationship"""
        return self.relation in [RelationType.SUPPRESS, RelationType.OVERCONSTRAINT, RelationType.REVERSE]
    
    @property
    def boost_factor(self) -> float:
        """Get the boost factor for link weights"""
        if self.relation == RelationType.ENHANCE:
            return 1.2  # 增强 20%
        elif self.relation == RelationType.SUPPRESS:
            return 0.8  # 削弱 20%
        elif self.relation == RelationType.OVERCONSTRAINT:
            return 0.6  # 削弱 40%
        elif self.relation == RelationType.REVERSE:
            return 0.4  # 削弱 60%
        elif self.relation == RelationType.SAME:
            return 1.1  # 增强 10%
        return 1.0  # 中性不变


@dataclass
class MemoryNodeEnergy:
    """Memory node with energy attributes"""
    node_id: str
    energy_type: str
    intensity: float = 1.0
    stem_idx: Optional[int] = None
    branch_idx: Optional[int] = None


# ============================================================
# Core Functions
# ============================================================

def get_enhance_relation(source: str, target: str) -> bool:
    """
    Check if source energy enhances target energy.
    
    Args:
        source: Source energy type (e.g., "wood")
        target: Target energy type (e.g., "fire")
    
    Returns:
        True if source enhances target
    """
    return ENERGY_ENHANCE.get(source) == target


def get_suppress_relation(source: str, target: str) -> bool:
    """
    Check if source energy suppresses target energy.
    
    Args:
        source: Source energy type (e.g., "wood")
        target: Target energy type (e.g., "earth")
    
    Returns:
        True if source suppresses target
    """
    return ENERGY_SUPPRESS.get(source) == target


def analyze_relation(source: str, target: str) -> EnergyRelation:
    """
    Analyze the energy relation between two elements.
    
    Args:
        source: Source energy type
        target: Target energy type
    
    Returns:
        EnergyRelation with detailed information
    """
    # Same type
    if source == target:
        return EnergyRelation(
            source=source,
            target=target,
            relation=RelationType.SAME,
            strength=RELATION_STRENGTH[RelationType.SAME],
            description=f"{source} and {target} are the same type (同类)"
        )
    
    # Enhance relationship
    if get_enhance_relation(source, target):
        return EnergyRelation(
            source=source,
            target=target,
            relation=RelationType.ENHANCE,
            strength=RELATION_STRENGTH[RelationType.ENHANCE],
            description=f"{source} enhances {target} (相生)"
        )
    
    # Suppress relationship
    if get_suppress_relation(source, target):
        return EnergyRelation(
            source=source,
            target=target,
            relation=RelationType.SUPPRESS,
            strength=RELATION_STRENGTH[RelationType.SUPPRESS],
            description=f"{source} suppresses {target} (相克)"
        )
    
    # Check for reverse relationships
    # Source is suppressed by target
    if ENERGY_SUPPRESSED_BY.get(source) == target:
        return EnergyRelation(
            source=source,
            target=target,
            relation=RelationType.REVERSE,
            strength=RELATION_STRENGTH[RelationType.REVERSE],
            description=f"{target} suppresses {source} (反克: 相侮)"
        )
    
    # Source is enhanced by target
    if ENERGY_ENHANCED_BY.get(source) == target:
        return EnergyRelation(
            source=source,
            target=target,
            relation=RelationType.ENHANCE,
            strength=RELATION_STRENGTH[RelationType.ENHANCE],
            description=f"{target} enhances {source} (反生)"
        )
    
    # Neutral - no direct relationship
    return EnergyRelation(
        source=source,
        target=target,
        relation=RelationType.NEUTRAL,
        strength=RELATION_STRENGTH[RelationType.NEUTRAL],
        description=f"{source} and {target} have no direct relationship (无关)"
    )


def calculate_link_weight(
    source_energy: str,
    target_energy: str,
    base_weight: float = 1.0
) -> float:
    """
    Calculate the link weight between two memory nodes based on energy relations.
    
    When two memory nodes have an enhancing (相生) relationship, their link
    weight is increased. When they have a suppressing (相克) relationship,
    their link weight is decreased.
    
    Args:
        source_energy: Source node's energy type
        target_energy: Target node's energy type
        base_weight: Base link weight (default: 1.0)
    
    Returns:
        Adjusted link weight
    """
    relation = analyze_relation(source_energy, target_energy)
    return base_weight * relation.boost_factor


def get_cycle_sequence(start: str, steps: int = 5) -> List[str]:
    """
    Get the enhance cycle sequence starting from an element.
    
    Args:
        start: Starting element
        steps: Number of steps (default: 5 for full cycle)
    
    Returns:
        List of elements in sequence
    """
    sequence = [start]
    current = start
    
    for _ in range(steps - 1):
        next_element = ENERGY_ENHANCE.get(current)
        if next_element is None:
            break
        sequence.append(next_element)
        current = next_element
    
    return sequence


def get_suppress_chain(start: str, steps: int = 5) -> List[str]:
    """
    Get the suppress chain starting from an element.
    
    Args:
        start: Starting element
        steps: Number of steps (default: 5 for full cycle)
    
    Returns:
        List of elements in sequence
    """
    chain = [start]
    current = start
    
    for _ in range(steps - 1):
        next_element = ENERGY_SUPPRESS.get(current)
        if next_element is None:
            break
        chain.append(next_element)
        current = next_element
    
    return chain


def analyze_balance(energy_distribution: Dict[str, float]) -> Dict[str, Any]:
    """
    Analyze the balance of energy distribution.
    
    Args:
        energy_distribution: Dict mapping energy types to their intensities
    
    Returns:
        Analysis result with balance status and suggestions
    """
    if not energy_distribution:
        return {
            "status": "empty",
            "dominant": None,
            "weak": None,
            "ratio": {},
            "suggestions": []
        }
    
    total = sum(energy_distribution.values())
    ratio = {k: v / total for k, v in energy_distribution.items() if total > 0}
    
    # Find dominant and weak energies
    dominant = max(energy_distribution, key=energy_distribution.get)
    weak = min(energy_distribution, key=energy_distribution.get)
    
    # Analyze status
    max_ratio = ratio.get(dominant, 0)
    
    if max_ratio > 0.5:
        status = "concentrated"  # 能量集中
        suggestions = [f"{dominant} is dominant, consider balancing"]
    elif max_ratio < 0.1:
        status = "dispersed"  # 能量分散
        suggestions = ["Energy is dispersed, no clear pattern"]
    else:
        status = "balanced"  # 能量平衡
        suggestions = []
    
    # Check for conflicts
    if dominant and weak:
        if get_suppress_relation(dominant, weak):
            suggestions.append(f"Warning: {dominant} suppresses {weak}")
    
    return {
        "status": status,
        "dominant": dominant,
        "weak": weak,
        "ratio": ratio,
        "suggestions": suggestions
    }


def get_affinity_score(energy1: str, energy2: str) -> float:
    """
    Calculate affinity score between two energy types.
    
    Args:
        energy1: First energy type
        energy2: Second energy type
    
    Returns:
        Affinity score (0.0 - 2.0), higher means more compatible
    """
    relation = analyze_relation(energy1, energy2)
    
    if relation.relation == RelationType.ENHANCE:
        return 1.5  # High affinity - complementary
    elif relation.relation == RelationType.SAME:
        return 1.2  # Medium-high affinity - same type
    elif relation.relation == RelationType.SUPPRESS:
        return 0.6  # Low affinity - conflict
    elif relation.relation == RelationType.REVERSE:
        return 0.3  # Very low affinity - strong conflict
    else:
        return 1.0  # Neutral


# ============================================================
# Convenience Functions
# ============================================================

def is_enhancing(energy1: str, energy2: str) -> bool:
    """Check if energy1 enhances energy2"""
    return get_enhance_relation(energy1, energy2)


def is_suppressing(energy1: str, energy2: str) -> bool:
    """Check if energy1 suppresses energy2"""
    return get_suppress_relation(energy1, energy2)


def get_enhanced_energy(energy: str) -> Optional[str]:
    """Get the energy that this element enhances"""
    return ENERGY_ENHANCE.get(energy)


def get_enhancing_energy(energy: str) -> Optional[str]:
    """Get the energy that enhances this element"""
    return ENERGY_ENHANCED_BY.get(energy)


def get_suppressed_energy(energy: str) -> Optional[str]:
    """Get the energy that this element suppresses"""
    return ENERGY_SUPPRESS.get(energy)


def get_suppressing_energy(energy: str) -> Optional[str]:
    """Get the energy that suppresses this element"""
    return ENERGY_SUPPRESSED_BY.get(energy)


# ============================================================
# Unit Tests
# ============================================================

def test_energy_relations():
    """Test Five Elements relationship functions"""
    print("=" * 60)
    print("Testing Energy Relations Module")
    print("=" * 60)
    
    # Test 1: Enhance relationships
    print("\n[Test 1] Enhance Relationships (相生)")
    test_cases = [
        ("wood", "fire", True),    # 木生火
        ("fire", "earth", True),   # 火生土
        ("earth", "metal", True),  # 土生金
        ("metal", "water", True),  # 金生水
        ("water", "wood", True),   # 水生木
        ("fire", "wood", False),   # 火不生木
        ("wood", "earth", False),  # 木不生土
    ]
    
    all_passed = True
    for source, target, expected in test_cases:
        result = get_enhance_relation(source, target)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"  {status} {source} -> {target}: {result} (expected: {expected})")
    
    # Test 2: Suppress relationships
    print("\n[Test 2] Suppress Relationships (相克)")
    test_cases = [
        ("wood", "earth", True),   # 木克土
        ("earth", "water", True),  # 土克水
        ("water", "fire", True),   # 水克火
        ("fire", "metal", True),   # 火克金
        ("metal", "wood", True),   # 金克木
        ("fire", "earth", False),  # 火不克土
        ("earth", "wood", False),  # 土不克木 (木克土是其反向)
    ]
    
    for source, target, expected in test_cases:
        result = get_suppress_relation(source, target)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"  {status} {source} -克> {target}: {result} (expected: {expected})")
    
    # Test 3: Relation analysis
    print("\n[Test 3] Relation Analysis")
    test_cases = [
        ("wood", "fire", RelationType.ENHANCE, 1.2),      # 木生火
        ("wood", "earth", RelationType.SUPPRESS, 0.8),    # 木克土
        ("fire", "fire", RelationType.SAME, 1.1),          # 火同火
        ("wood", "metal", RelationType.REVERSE, 0.4),      # 金克木 (相侮)
    ]
    
    for source, target, exp_rel, exp_str in test_cases:
        result = analyze_relation(source, target)
        rel_ok = result.relation == exp_rel
        str_ok = abs(result.strength - exp_str) < 0.01
        status = "✓" if rel_ok and str_ok else "✗"
        if not (rel_ok and str_ok):
            all_passed = False
        print(f"  {status} {source} <-> {target}: {result.relation.value} ({result.strength})")
        print(f"      Description: {result.description}")
    
    # Test 4: Link weight calculation
    print("\n[Test 4] Link Weight Calculation")
    test_cases = [
        ("wood", "fire", 1.0, 1.2),    # 相生增强
        ("wood", "earth", 1.0, 0.8),   # 相克削弱
        ("fire", "fire", 1.0, 1.1),    # 同类增强
        ("wood", "metal", 1.0, 0.4),    # 相侮削弱
    ]
    
    for source, target, base, expected in test_cases:
        result = calculate_link_weight(source, target, base)
        status = "✓" if abs(result - expected) < 0.01 else "✗"
        if abs(result - expected) >= 0.01:
            all_passed = False
        print(f"  {status} Link({source}, {target}, base={base}): {result}")
    
    # Test 5: Cycle sequences
    print("\n[Test 5] Cycle Sequences")
    enhance_seq = get_cycle_sequence("wood", 5)
    print(f"  Enhance cycle from wood: {' -> '.join(enhance_seq)}")
    assert enhance_seq == ["wood", "fire", "earth", "metal", "water"]
    
    suppress_seq = get_suppress_chain("wood", 5)
    print(f"  Suppress chain from wood: {' -> '.join(suppress_seq)}")
    assert suppress_seq == ["wood", "earth", "water", "fire", "metal"]
    
    # Test 6: Balance analysis
    print("\n[Test 6] Balance Analysis")
    dist = {"wood": 0.3, "fire": 0.2, "earth": 0.2, "metal": 0.15, "water": 0.15}
    result = analyze_balance(dist)
    print(f"  Status: {result['status']}")
    print(f"  Dominant: {result['dominant']}")
    print(f"  Ratio: {result['ratio']}")
    
    # Test 7: Four Symbols mapping
    print("\n[Test 7] Four Symbols Mapping (四象映射)")
    four_symbol_tests = [
        ("wood", "SHAO_YANG"),   # 木 -> 少阳
        ("fire", "TAI_YANG"),     # 火 -> 太阳
        ("earth", "CENTER"),      # 土 -> 中宫
        ("metal", "SHAO_YIN"),    # 金 -> 少阴
        ("water", "TAI_YIN"),     # 水 -> 太阴
    ]
    
    for energy, expected_symbol in four_symbol_tests:
        result = ENERGY_TO_FOUR_SYMBOLS.get(energy)
        status = "✓" if result == expected_symbol else "✗"
        if result != expected_symbol:
            all_passed = False
        print(f"  {status} {energy} -> {result} (expected: {expected_symbol})")
    
    # Test 8: Season mapping
    print("\n[Test 8] Season Energy Mapping (季节能量)")
    season_tests = [
        ("spring", "wood"),      # 春 -> 木
        ("summer", "fire"),       # 夏 -> 火
        ("late_summer", "earth"), # 长夏 -> 土
        ("autumn", "metal"),      # 秋 -> 金
        ("winter", "water"),      # 冬 -> 水
    ]
    
    for season, expected_energy in season_tests:
        result = SEASON_ENERGY_MAP.get(season)
        status = "✓" if result == expected_energy else "✗"
        if result != expected_energy:
            all_passed = False
        print(f"  {status} {season} -> {result} (expected: {expected_energy})")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    test_energy_relations()
