"""
Unified Information Unit - Three Talents Integration

This module implements the core data structure for the Three Talents Integration (Triad System合一)
system, unifying temporal (sky), spatial (earth), and elemental (human) information into
a single cohesive unit.

Architecture:
- Heaven Layer (Sky/Temporal): Time stems and branches
- Earth Layer (Spatial): Trigrams and hexagrams
- Human Layer (Element): Energy types and strength states

Usage:
    >>> factory = UnifiedInfoFactory()
    >>> unit = factory.create_from_temporal_code(0, 0)  # 甲子
    >>> print(unit.heaven_layer)
    >>> print(unit.earth_layer)
    >>> print(unit.human_layer)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import uuid
import time

# Import enums from _enums.py
from ._enums import TimeStem, TimeBranch, TrigramType, EnergyType, StrengthState

# Import mapping tables from _terms.py
from ._terms import (
    ENERGY_DIRECTION,
    ENERGY_COLOR,
    ENERGY_ORGAN,
    ENERGY_EMOTION,
    ENERGY_INDUSTRY,
    TIME_STEMS,
    TIME_BRANCHES,
    TRIGRAM_ENERGY_MAP,
)

# Import core engines
from ._temporal_core import TemporalCore
from ._trigram_core import TrigramCore
from ._energy_core import EnergyCore


# =============================================================================
# Energy Names Mapping
# =============================================================================

ENERGY_NAMES = {
    0: "wood",
    1: "fire",
    2: "earth",
    3: "metal",
    4: "water",
}


# =============================================================================
# Unified Information Unit Data Class
# =============================================================================

@dataclass
class UnifiedInfoUnit:
    """
    Unified Information Unit (统一信息单元)
    
    The core data structure for Three Talents Integration (Triad System合一) system,
    containing information from three layers:
    
    Heaven Layer (天层/Temporal):
        - temporal_stem: Heavenly stem (0-9)
        - temporal_branch: Earthly branch (0-11)
        - cyclic_code: Position in 60-cycle (0-59)
        - temporal_intensity: Temporal intensity factor
    
    Earth Layer (地层/Spatial):
        - trigram: Trigram type (0-7)
        - hexagram_index: Hexagram index (0-63)
        - prior_trigram: Prior trigram (Fu Xi sequence)
        - post_trigram: Post trigram (Wen Wang sequence)
    
    Human Layer (人层/Element):
        - energy_type: Energy type (0-4)
        - energy_intensity: Energy intensity factor
        - strength_state: Strength state (旺相休囚死)
    
    Extended Attributes:
        - direction: Associated directions
        - colors: Associated colors
        - organs: Associated organs
        - emotions: Associated emotions
        - industries: Associated industries
    
    Relationship Information:
        - related_units: Related information units
        - causal_chain: Causal chain identifiers
    
    Attributes:
        id: Unique identifier for this unit
        content: Text content or description
        timestamp: Unix timestamp when created
        metadata: Additional metadata dictionary
    
    Example:
        >>> unit = UnifiedInfoUnit(
        ...     id="test-001",
        ...     content="甲子",
        ...     timestamp=int(time.time()),
        ...     temporal_stem=0,
        ...     temporal_branch=0,
        ...     energy_type=3
        ... )
        >>> unit.direction
        ['west', 'northwest']
    """
    
    # ========== Basic Information ==========
    id: str
    content: str
    timestamp: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # ========== Heaven Layer (Temporal) ==========
    temporal_stem: Optional[int] = None  # 0-9
    temporal_branch: Optional[int] = None  # 0-11
    cyclic_code: Optional[int] = None  # 0-59
    temporal_intensity: float = 1.0
    
    # ========== Earth Layer (Spatial) ==========
    trigram: Optional[int] = None  # 0-7
    hexagram_index: Optional[int] = None  # 0-63
    prior_trigram: Optional[int] = None  # 0-7
    post_trigram: Optional[int] = None  # 0-7
    
    # ========== Human Layer (Element) ==========
    energy_type: Optional[int] = None  # 0-4
    energy_intensity: float = 1.0
    strength_state: Optional[int] = None  # 0-4
    
    # ========== Extended Attributes ==========
    direction: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    organs: List[str] = field(default_factory=list)
    emotions: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    
    # ========== Relationship Information ==========
    related_units: List[str] = field(default_factory=list)
    causal_chain: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Auto-fill extended attributes based on energy type."""
        if self.energy_type is not None:
            self._fill_energy_attributes()
    
    def _fill_energy_attributes(self):
        """
        Fill extended attributes based on energy type.
        
        This method automatically populates direction, colors, organs,
        emotions, and industries based on the energy_type value.
        """
        if 0 <= self.energy_type < 5:
            energy_name = ENERGY_NAMES[self.energy_type]
            self.direction = ENERGY_DIRECTION.get(energy_name, [])
            self.colors = ENERGY_COLOR.get(energy_name, [])
            organ = ENERGY_ORGAN.get(energy_name, "")
            self.organs = [organ] if organ else []
            emotion = ENERGY_EMOTION.get(energy_name, "")
            self.emotions = [emotion] if emotion else []
            self.industries = ENERGY_INDUSTRY.get(energy_name, [])
    
    # ========== Layer Properties ==========
    
    @property
    def heaven_layer(self) -> Dict[str, Any]:
        """
        Get Heaven Layer (Temporal) information.
        
        Returns:
            Dictionary containing temporal information:
            - stem: Heavenly stem index (0-9)
            - branch: Earthly branch index (0-11)
            - cycle_position: Position in 60-cycle (0-59)
            - intensity: Temporal intensity factor
            - stem_name: Name of the heavenly stem
            - branch_name: Name of the earthly branch
        """
        result = {
            "stem": self.temporal_stem,
            "branch": self.temporal_branch,
            "cycle_position": self.cyclic_code,
            "intensity": self.temporal_intensity,
        }
        if self.temporal_stem is not None:
            result["stem_name"] = TIME_STEMS[self.temporal_stem] if 0 <= self.temporal_stem < 10 else None
        if self.temporal_branch is not None:
            result["branch_name"] = TIME_BRANCHES[self.temporal_branch] if 0 <= self.temporal_branch < 12 else None
        return result
    
    @property
    def earth_layer(self) -> Dict[str, Any]:
        """
        Get Earth Layer (Spatial) information.
        
        Returns:
            Dictionary containing spatial information:
            - trigram: Trigram type index (0-7)
            - hexagram_index: Hexagram index (0-63)
            - prior_trigram: Prior trigram index (Fu Xi)
            - post_trigram: Post trigram index (Wen Wang)
            - trigram_name: Name of the trigram
        """
        result = {
            "trigram": self.trigram,
            "hexagram_index": self.hexagram_index,
            "prior_trigram": self.prior_trigram,
            "post_trigram": self.post_trigram,
        }
        if self.trigram is not None and 0 <= self.trigram < 8:
            result["trigram_name"] = TrigramType(self.trigram).name
        return result
    
    @property
    def human_layer(self) -> Dict[str, Any]:
        """
        Get Human Layer (Element) information.
        
        Returns:
            Dictionary containing elemental information:
            - energy_type: Energy type index (0-4)
            - intensity: Energy intensity factor
            - strength_state: Strength state index (0-4)
            - energy_name: Name of the energy type
            - attributes: Extended attributes dictionary
        """
        result = {
            "energy_type": self.energy_type,
            "intensity": self.energy_intensity,
            "strength_state": self.strength_state,
            "attributes": {
                "direction": self.direction,
                "colors": self.colors,
                "organs": self.organs,
                "emotions": self.emotions,
                "industries": self.industries,
            }
        }
        if self.energy_type is not None and 0 <= self.energy_type < 5:
            result["energy_name"] = ENERGY_NAMES[self.energy_type]
        if self.strength_state is not None and 0 <= self.strength_state < 5:
            result["strength_name"] = StrengthState(self.strength_state).name
        return result
    
    # ========== Serialization Methods ==========
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the unit to a dictionary.
        
        Returns:
            Dictionary representation of the unit with three-layer structure:
            - id: Unique identifier
            - content: Text content
            - timestamp: Creation timestamp
            - heaven: Heaven layer data
            - earth: Earth layer data
            - human: Human layer data
            - relations: Relationship data
        """
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "heaven": self.heaven_layer,
            "earth": self.earth_layer,
            "human": self.human_layer,
            "relations": {
                "related_units": self.related_units,
                "causal_chain": self.causal_chain,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedInfoUnit':
        """
        Create a UnifiedInfoUnit from a dictionary.
        
        Args:
            data: Dictionary containing unit data
            
        Returns:
            UnifiedInfoUnit instance
            
        Example:
            >>> data = unit.to_dict()
            >>> restored = UnifiedInfoUnit.from_dict(data)
        """
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", int(time.time())),
            metadata=data.get("metadata", {}),
            temporal_stem=data.get("heaven", {}).get("stem"),
            temporal_branch=data.get("heaven", {}).get("branch"),
            cyclic_code=data.get("heaven", {}).get("cycle_position"),
            temporal_intensity=data.get("heaven", {}).get("intensity", 1.0),
            trigram=data.get("earth", {}).get("trigram"),
            hexagram_index=data.get("earth", {}).get("hexagram_index"),
            prior_trigram=data.get("earth", {}).get("prior_trigram"),
            post_trigram=data.get("earth", {}).get("post_trigram"),
            energy_type=data.get("human", {}).get("energy_type"),
            energy_intensity=data.get("human", {}).get("intensity", 1.0),
            strength_state=data.get("human", {}).get("strength_state"),
            direction=data.get("human", {}).get("attributes", {}).get("direction", []),
            colors=data.get("human", {}).get("attributes", {}).get("colors", []),
            organs=data.get("human", {}).get("attributes", {}).get("organs", []),
            emotions=data.get("human", {}).get("attributes", {}).get("emotions", []),
            industries=data.get("human", {}).get("attributes", {}).get("industries", []),
            related_units=data.get("relations", {}).get("related_units", []),
            causal_chain=data.get("relations", {}).get("causal_chain", []),
        )
    
    def __repr__(self) -> str:
        """String representation of the unit."""
        return (
            f"UnifiedInfoUnit(id={self.id[:8]}..., "
            f"stem={self.temporal_stem}, "
            f"branch={self.temporal_branch}, "
            f"energy={self.energy_type})"
        )


# =============================================================================
# Unified Information Factory
# =============================================================================

class UnifiedInfoFactory:
    """
    Factory for creating and parsing UnifiedInfoUnit instances.
    
    This factory uses TemporalCore, TrigramCore, and EnergyCore engines
    to create unified information units from various input types.
    
    Attributes:
        _temporal_core: Temporal encoding engine
        _trigram_core: Trigram encoding engine
        _energy_core: Energy calculation engine
    
    Example:
        >>> factory = UnifiedInfoFactory()
        >>> unit = factory.create_from_temporal_code(0, 0)  # 甲子
        >>> print(unit.heaven_layer)
        >>> print(unit.earth_layer)
        >>> print(unit.human_layer)
    """
    
    def __init__(self):
        """Initialize the factory with core engines."""
        self._temporal_core = TemporalCore()
        self._trigram_core = TrigramCore()
        self._energy_core = EnergyCore()
    
    def create_from_content(
        self,
        content: str,
        stem_idx: int = 0,
        branch_idx: int = 0,
        hexagram_idx: int = 0,
        energy_type: int = 2
    ) -> UnifiedInfoUnit:
        """
        Create a unified information unit from content and indices.
        
        Args:
            content: Text content or description
            stem_idx: Heavenly stem index (0-9)
            branch_idx: Earthly branch index (0-11)
            hexagram_idx: Hexagram index (0-63)
            energy_type: Energy type index (0-4, default: 2=earth)
        
        Returns:
            UnifiedInfoUnit instance with all three layers populated
        
        Example:
            >>> factory = UnifiedInfoFactory()
            >>> unit = factory.create_from_content("测试内容", 0, 0, 0, 3)
            >>> print(unit.content)
            '测试内容'
        """
        # Get temporal information
        stem_idx = stem_idx % 10
        branch_idx = branch_idx % 12
        cyclic_code = self._temporal_core.get_cycle_index(
            TimeStem(stem_idx),
            TimeBranch(branch_idx)
        )
        
        # Get spatial information from hexagram
        upper_trigram, lower_trigram = self._trigram_core.get_hexagram(hexagram_idx)
        prior_idx = self._trigram_core.get_prior_order(TrigramType(lower_trigram))
        post_idx = self._trigram_core.get_post_order(TrigramType(lower_trigram))
        
        # Get energy state
        energy_idx = energy_type % 5
        energy_name = ENERGY_NAMES[energy_idx]
        
        # Create the unit
        return UnifiedInfoUnit(
            id=str(uuid.uuid4()),
            content=content,
            timestamp=int(time.time()),
            temporal_stem=stem_idx,
            temporal_branch=branch_idx,
            cyclic_code=cyclic_code,
            trigram=int(lower_trigram),
            hexagram_index=hexagram_idx,
            prior_trigram=prior_idx,
            post_trigram=post_idx,
            energy_type=energy_idx,
            strength_state=StrengthState.XIANG.value,
        )
    
    def create_from_temporal_code(
        self,
        stem_idx: int,
        branch_idx: int
    ) -> UnifiedInfoUnit:
        """
        Create a unified information unit from stem-branch code.
        
        Args:
            stem_idx: Heavenly stem index (0-9)
            branch_idx: Earthly branch index (0-11)
        
        Returns:
            UnifiedInfoUnit with temporal and derived earth/human layers
        
        Example:
            >>> factory = UnifiedInfoFactory()
            >>> unit = factory.create_from_temporal_code(0, 0)  # 甲子
            >>> unit.temporal_stem
            0
            >>> unit.temporal_branch
            0
            >>> unit.cyclic_code
            0
        """
        # Normalize indices
        stem_idx = stem_idx % 10
        branch_idx = branch_idx % 12
        
        # Get temporal information
        cyclic_code = self._temporal_core.get_cycle_index(
            TimeStem(stem_idx),
            TimeBranch(branch_idx)
        )
        
        # Get temporal code name
        code_name = self._temporal_core.get_cycle_name(cyclic_code)
        
        # Get energy type from branch
        branch = TimeBranch(branch_idx)
        energy_name = self._temporal_core.get_branch_energy(branch)
        energy_idx = list(ENERGY_NAMES.values()).index(energy_name)
        
        # Get strength state (default to balanced)
        strength_state = StrengthState.XIANG.value
        
        # Map branch to trigram (simplified mapping)
        trigram_idx = self._map_branch_to_trigram(branch_idx)
        hexagram_idx = self._trigram_core.get_hexagram_number(
            TrigramType(trigram_idx),
            TrigramType(0)
        )
        
        # Get prior/post trigrams
        prior_idx = self._trigram_core.get_prior_order(TrigramType(trigram_idx))
        post_idx = self._trigram_core.get_post_order(TrigramType(trigram_idx))
        
        return UnifiedInfoUnit(
            id=str(uuid.uuid4()),
            content=code_name,
            timestamp=int(time.time()),
            temporal_stem=stem_idx,
            temporal_branch=branch_idx,
            cyclic_code=cyclic_code,
            trigram=trigram_idx,
            hexagram_index=hexagram_idx,
            prior_trigram=prior_idx,
            post_trigram=post_idx,
            energy_type=energy_idx,
            strength_state=strength_state,
        )
    
    def create_from_hexagram(
        self,
        hexagram_idx: int
    ) -> UnifiedInfoUnit:
        """
        Create a unified information unit from a hexagram.
        
        Args:
            hexagram_idx: Hexagram index (0-63)
        
        Returns:
            UnifiedInfoUnit with earth layer populated
        
        Example:
            >>> factory = UnifiedInfoFactory()
            >>> unit = factory.create_from_hexagram(0)  # 乾
            >>> unit.trigram
            0
            >>> unit.hexagram_index
            0
        """
        # Normalize index
        hexagram_idx = hexagram_idx % 64
        
        # Get trigram information
        upper_trigram, lower_trigram = self._trigram_core.get_hexagram(hexagram_idx)
        lower_idx = int(lower_trigram)
        
        # Get prior/post orders
        prior_idx = self._trigram_core.get_prior_order(TrigramType(lower_idx))
        post_idx = self._trigram_core.get_post_order(TrigramType(lower_idx))
        
        # Get energy type from trigram
        trigram = TrigramType(lower_idx)
        energy_name = self._trigram_core.get_trigram_energy_type(trigram)
        energy_idx = list(ENERGY_NAMES.values()).index(energy_name)
        
        return UnifiedInfoUnit(
            id=str(uuid.uuid4()),
            content=f"Hexagram_{hexagram_idx}",
            timestamp=int(time.time()),
            trigram=lower_idx,
            hexagram_index=hexagram_idx,
            prior_trigram=prior_idx,
            post_trigram=post_idx,
            energy_type=energy_idx,
            strength_state=StrengthState.XIANG.value,
        )
    
    def create_random(
        self,
        content: str = ""
    ) -> UnifiedInfoUnit:
        """
        Create a random unified information unit.
        
        Args:
            content: Optional text content
        
        Returns:
            UnifiedInfoUnit with random temporal, spatial, and elemental values
        
        Example:
            >>> factory = UnifiedInfoFactory()
            >>> unit = factory.create_random("随机信息")
            >>> 0 <= unit.temporal_stem <= 9
            True
        """
        import random
        
        stem_idx = random.randint(0, 9)
        branch_idx = random.randint(0, 11)
        hexagram_idx = random.randint(0, 63)
        energy_idx = random.randint(0, 4)
        
        return self.create_from_content(
            content=content or f"Random_{int(time.time())}",
            stem_idx=stem_idx,
            branch_idx=branch_idx,
            hexagram_idx=hexagram_idx,
            energy_type=energy_idx
        )
    
    def _map_branch_to_trigram(self, branch_idx: int) -> int:
        """
        Map earthly branch to corresponding trigram.
        
        This is a simplified mapping based on the five elements
        association of branches.
        
        Args:
            branch_idx: Earthly branch index (0-11)
        
        Returns:
            Corresponding trigram index (0-7)
        """
        # Simplified branch to trigram mapping
        branch_trigram_map = {
            0: 4,   # 子 -> 坎 (water)
            1: 6,   # 丑 -> 艮 (earth)
            2: 3,   # 寅 -> 巽 (wood)
            3: 3,   # 卯 -> 巽 (wood)
            4: 6,   # 辰 -> 艮 (earth)
            5: 5,   # 巳 -> 离 (fire)
            6: 5,   # 午 -> 离 (fire)
            7: 6,   # 未 -> 艮 (earth)
            8: 7,   # 申 -> 兑 (metal)
            9: 7,   # 酉 -> 兑 (metal)
            10: 6,  # 戌 -> 艮 (earth)
            11: 4,  # 亥 -> 坎 (water)
        }
        return branch_trigram_map.get(branch_idx, 0)


# =============================================================================
# Utility Functions
# =============================================================================

def create_unified_unit(
    stem_idx: int = 0,
    branch_idx: int = 0,
    content: str = ""
) -> UnifiedInfoUnit:
    """
    Convenience function to create a unified information unit.
    
    Args:
        stem_idx: Heavenly stem index (0-9)
        branch_idx: Earthly branch index (0-11)
        content: Optional text content
    
    Returns:
        UnifiedInfoUnit instance
    
    Example:
        >>> unit = create_unified_unit(0, 0, "甲子")
        >>> unit.temporal_stem
        0
    """
    factory = UnifiedInfoFactory()
    return factory.create_from_temporal_code(stem_idx, branch_idx)


# =============================================================================
# Test Suite
# =============================================================================

def run_tests():
    """
    Run built-in test cases for UnifiedInfoUnit.
    
    Returns:
        True if all tests pass, False otherwise
    """
    print("=" * 60)
    print("UnifiedInfoUnit Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Create from temporal code
    print("\n[Test 1] Create from temporal code (甲子)...")
    try:
        factory = UnifiedInfoFactory()
        unit = factory.create_from_temporal_code(0, 0)  # 甲子
        assert unit.temporal_stem == 0, f"Expected stem 0, got {unit.temporal_stem}"
        assert unit.temporal_branch == 0, f"Expected branch 0, got {unit.temporal_branch}"
        assert unit.cyclic_code == 0, f"Expected cyclic 0, got {unit.cyclic_code}"
        print(f"  PASS: stem={unit.temporal_stem}, branch={unit.temporal_branch}, cyclic={unit.cyclic_code}")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        tests_failed += 1
    
    # Test 2: Extended attributes
    print("\n[Test 2] Extended attributes auto-fill...")
    try:
        unit = UnifiedInfoUnit(
            id="test-002",
            content="测试",
            timestamp=int(time.time()),
            energy_type=3  # metal
        )
        assert len(unit.direction) > 0, "Direction should not be empty"
        assert len(unit.colors) > 0, "Colors should not be empty"
        assert "lung" in unit.organs, f"Expected 'lung' in organs, got {unit.organs}"
        print(f"  PASS: direction={unit.direction}, colors={unit.colors}, organs={unit.organs}")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        tests_failed += 1
    
    # Test 3: Three-layer interface
    print("\n[Test 3] Three-layer interface...")
    try:
        unit = factory.create_from_temporal_code(4, 6)  # 戊午
        heaven = unit.heaven_layer
        earth = unit.earth_layer
        human = unit.human_layer
        assert "stem" in heaven, "Heaven layer missing 'stem'"
        assert "trigram" in earth, "Earth layer missing 'trigram'"
        assert "energy_type" in human, "Human layer missing 'energy_type'"
        print(f"  PASS: heaven={heaven}, earth={earth}, human={human}")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        tests_failed += 1
    
    # Test 4: Serialization
    print("\n[Test 4] Serialization and deserialization...")
    try:
        original = factory.create_from_content("测试内容", 1, 2, 5, 2)
        data = original.to_dict()
        restored = UnifiedInfoUnit.from_dict(data)
        assert restored.id == original.id, f"ID mismatch: {restored.id} != {original.id}"
        assert restored.temporal_stem == original.temporal_stem
        assert restored.energy_type == original.energy_type
        print(f"  PASS: Original ID={original.id[:8]}, Restored ID={restored.id[:8]}")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        tests_failed += 1
    
    # Test 5: Create from hexagram
    print("\n[Test 5] Create from hexagram...")
    try:
        unit = factory.create_from_hexagram(0)  # 乾
        assert unit.trigram == 0, f"Expected trigram 0, got {unit.trigram}"
        assert unit.hexagram_index == 0, f"Expected hexagram 0, got {unit.hexagram_index}"
        print(f"  PASS: trigram={unit.trigram}, hexagram={unit.hexagram_index}")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        tests_failed += 1
    
    # Test 6: Random creation
    print("\n[Test 6] Random unified unit creation...")
    try:
        unit = factory.create_random("随机测试")
        assert unit.temporal_stem is not None
        assert unit.temporal_branch is not None
        assert unit.hexagram_index is not None
        assert unit.energy_type is not None
        print(f"  PASS: random stem={unit.temporal_stem}, branch={unit.temporal_branch}")
        print(f"        hexagram={unit.hexagram_index}, energy={unit.energy_type}")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        tests_failed += 1
    
    # Test 7: All five energy types
    print("\n[Test 7] All five energy types extended attributes...")
    try:
        for energy_idx in range(5):
            unit = UnifiedInfoUnit(
                id=f"test-{energy_idx}",
                content="测试",
                timestamp=int(time.time()),
                energy_type=energy_idx
            )
            assert len(unit.direction) > 0, f"Energy {energy_idx} missing directions"
            assert len(unit.colors) > 0, f"Energy {energy_idx} missing colors"
            print(f"  Energy {energy_idx} ({ENERGY_NAMES[energy_idx]}): "
                  f"dir={unit.direction}, colors={unit.colors}")
        print("  PASS: All energy types have extended attributes")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Test Summary: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    return tests_failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
