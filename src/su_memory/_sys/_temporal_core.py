"""
Temporal Core Engine - Stem-Branch Cyclical System

This module implements the core temporal encoding engine for the Sky Layer (Temporal Layer),
providing comprehensive functionality for:

- Sixty-cycle (Sixty Cycle) encoding and computation
- Heavenly Stem (Heavenly Stems) relationships (Wu He / Five Combinations, Xiang Chong / Six Conflicts)
- Earthly Branch (Earthly Branches) relationships:
  - Liu He (六合) - Six Harmonies
  - San He (三合) - Three Combinations
  - Liu Chong (六冲) - Six Conflicts
  - San Xing (三刑) - Three Punishments
  - Liu Hai (六害) - Six Harms
  - Po (破) - Broken relationships
- Hidden Stem (藏干) extraction for each branch

Architecture: Sky Layer (Tian) - Temporal System
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict
from ._enums import TimeStem, TimeBranch, BranchRelation
from ._terms import (
    STEM_HE_MAP,
    STEM_CHONG_MAP,
    BRANCH_HE_MAP,
    BRANCH_CHONG_MAP,
    BRANCH_SANHE_MAP,
    BRANCH_HIDDEN_STEM_MAP,
    TIME_STEMS,
    TIME_BRANCHES,
    TIME_BRANCH_ENERGY,
)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class StemBranchCode:
    """
    Stem-Branch Code (干支编码)
    
    Represents a complete cyclical time code combining a heavenly stem
    and an earthly branch.
    
    Attributes:
        stem: The heavenly stem (Heavenly Stems) - 0-9
        branch: The earthly branch (Earthly Branches) - 0-11
        cycle_index: Position in the 60-cycle (0-59)
    
    Example:
        >>> code = StemBranchCode(TimeStem.JIA, TimeBranch.ZI, 0)
        >>> code.name
        '甲子'
        >>> code.polarity
        'yang'
    """
    stem: TimeStem
    branch: TimeBranch
    cycle_index: int  # 0-59
    
    def __post_init__(self):
        """Validate cycle index range."""
        if not 0 <= self.cycle_index < 60:
            raise ValueError(f"Cycle index must be 0-59, got {self.cycle_index}")
    
    @property
    def polarity(self) -> str:
        """
        Get polarity (Duality属性).
        
        Yang: even stem values (0, 2, 4, 6, 8)
        Yin: odd stem values (1, 3, 5, 7, 9)
        
        Returns:
            'yang' or 'yin'
        """
        return "yang" if self.stem.value % 2 == 0 else "yin"
    
    @property
    def name(self) -> str:
        """
        Get the combined stem-branch name.
        
        Returns:
            Combined name like '甲子', '乙丑', etc.
        """
        return f"{TIME_STEMS[self.stem.value]}{TIME_BRANCHES[self.branch.value]}"
    
    @property
    def hidden_stems(self) -> List[TimeStem]:
        """
        Get the hidden stems (Earthly Branches藏干) within this branch.
        
        Each earthly branch contains one or more hidden heavenly stems
        with different strength levels.
        
        Returns:
            List of TimeStem values hidden in this branch
            
        Example:
            >>> StemBranchCode(TimeStem.JIA, TimeBranch.ZI, 0).hidden_stems
            [TimeStem.GUI]  # 子藏癸
        """
        stem_indices = BRANCH_HIDDEN_STEM_MAP.get(self.branch.value, [])
        return [TimeStem(idx) for idx in stem_indices]
    
    @property
    def energy_type(self) -> str:
        """
        Get the primary energy type (Energy System) of the branch.
        
        Returns:
            Energy type string: 'wood', 'fire', 'earth', 'metal', 'water'
        """
        return TIME_BRANCH_ENERGY.get(TIME_BRANCHES[self.branch.value], "earth")
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"StemBranchCode({self.stem.name}, {self.branch.name}, {self.cycle_index})"


# =============================================================================
# Branch Energy Type Mapping
# =============================================================================

# Branch to primary energy (本气) mapping
BRANCH_PRIMARY_ENERGY: Dict[TimeBranch, str] = {
    TimeBranch.ZI: "water",     # 子 - yang water
    TimeBranch.CHOU: "earth",   # 丑 - yin earth
    TimeBranch.YIN: "wood",     # 寅 - yang wood
    TimeBranch.MAO: "wood",     # 卯 - yin wood
    TimeBranch.CHEN: "earth",   # 辰 - yang earth
    TimeBranch.SI: "fire",      # 巳 - yin fire
    TimeBranch.WU: "fire",      # 午 - yang fire
    TimeBranch.WEI: "earth",    # 未 - yin earth
    TimeBranch.SHEN: "metal",   # 申 - yang metal
    TimeBranch.YOU: "metal",    # 酉 - yin metal
    TimeBranch.XU: "earth",     # 戌 - yang earth
    TimeBranch.HAI: "water",    # 亥 - yin water
}


# =============================================================================
# Core Engine
# =============================================================================

class TemporalCore:
    """
    Temporal Core Engine (时序干支核心引擎)
    
    The main engine for computing stem-branch relationships and
    performing temporal encoding/decoding operations.
    
    Features:
        - Sixty-cycle (Sixty Cycle) encoding
        - Heavenly stem relationship analysis (合/冲)
        - Earthly branch relationship analysis (六合/三合/六冲/三刑/六害/破)
        - Hidden stem extraction
        - Cycle distance computation
        - Trigram (三合局) detection
    
    Example:
        >>> tc = TemporalCore()
        >>> code = tc.create_code(0, 0)  # Create 甲子
        >>> code.name
        '甲子'
        >>> tc.analyze_stem_relation(TimeStem.JIA, TimeStem.JI)
        <BranchRelation.LIU_HE: 1>
    """
    
    def __init__(self):
        """Initialize the temporal core engine."""
        self._cycle: List[Tuple[TimeStem, TimeBranch]] = self._build_cycle()
        # Build reverse lookup: (stem, branch) -> index
        self._cycle_index_map: Dict[Tuple[int, int], int] = self._build_index_map()
    
    def _build_cycle(self) -> List[Tuple[TimeStem, TimeBranch]]:
        """
        Build the sixty-cycle (Sixty Cycle) sequence.
        
        The cycle combines heavenly stems (10) and earthly branches (12)
        using the pattern: stem[i % 10], branch[i % 12] for i in 0..59.
        
        Returns:
            List of (TimeStem, TimeBranch) tuples for indices 0-59
        """
        cycle = []
        for i in range(60):
            stem = TimeStem(i % 10)
            branch = TimeBranch(i % 12)
            cycle.append((stem, branch))
        return cycle
    
    def _build_index_map(self) -> Dict[Tuple[int, int], int]:
        """
        Build reverse lookup map from (stem_value, branch_value) to cycle index.
        
        Returns:
            Dictionary mapping (stem_idx, branch_idx) -> cycle_position
        """
        index_map = {}
        for idx, (stem, branch) in enumerate(self._cycle):
            index_map[(stem.value, branch.value)] = idx
        return index_map
    
    def create_code(self, stem_idx: int, branch_idx: int) -> StemBranchCode:
        """
        Create a StemBranchCode from stem and branch indices.
        
        Args:
            stem_idx: Heavenly stem index (0-9)
            branch_idx: Earthly branch index (0-11)
        
        Returns:
            StemBranchCode instance with correct cycle_index
            
        Example:
            >>> tc = TemporalCore()
            >>> code = tc.create_code(0, 0)  # 甲子
            >>> code.name
            '甲子'
        """
        stem = TimeStem(stem_idx % 10)
        branch = TimeBranch(branch_idx % 12)
        cycle_index = self._cycle_index_map.get((stem.value, branch.value), 0)
        return StemBranchCode(stem=stem, branch=branch, cycle_index=cycle_index)
    
    def get_cycle_name(self, index: int) -> str:
        """
        Get the stem-branch name at the specified cycle position.
        
        Args:
            index: Cycle position (0-59), wrapped modulo 60
        
        Returns:
            Stem-branch name like '甲子'
            
        Example:
            >>> tc = TemporalCore()
            >>> tc.get_cycle_name(0)  # 甲子
            '甲子'
            >>> tc.get_cycle_name(59)  # 癸亥
            '癸亥'
        """
        idx = index % 60
        stem, branch = self._cycle[idx]
        return f"{TIME_STEMS[stem.value]}{TIME_BRANCHES[branch.value]}"
    
    def get_cycle_index(self, stem: TimeStem, branch: TimeBranch) -> int:
        """
        Get the cycle position for a given stem-branch combination.
        
        Args:
            stem: The heavenly stem
            branch: The earthly branch
        
        Returns:
            Cycle position (0-59), or -1 if combination doesn't exist in cycle
            
        Example:
            >>> tc = TemporalCore()
            >>> tc.get_cycle_index(TimeStem.JIA, TimeBranch.ZI)
            0
        """
        return self._cycle_index_map.get((stem.value, branch.value), -1)
    
    def analyze_stem_relation(self, s1: TimeStem, s2: TimeStem) -> Optional[BranchRelation]:
        """
        Analyze the relationship between two heavenly stems.
        
        Checks for:
        - Liu He (六合): Wu He (五合) combinations - harmonious union
        - Li uChong (六冲): Xiang Chong (相冲) combinations - conflict
        
        Args:
            s1: First heavenly stem
            s2: Second heavenly stem
        
        Returns:
            BranchRelation.LIU_HE for harmonious combination,
            BranchRelation.LIU_CHONG for conflict,
            None if no special relationship
            
        Example:
            >>> tc = TemporalCore()
            >>> tc.analyze_stem_relation(TimeStem.JIA, TimeStem.JI)
            <BranchRelation.LIU_HE: 1>  # 甲己合
            >>> tc.analyze_stem_relation(TimeStem.JIA, TimeStem.GENG)
            <BranchRelation.LIU_CHONG: 3>  # 甲庚冲
        """
        # Check Wu He (五合) - Five Combinations
        if STEM_HE_MAP.get(s1.value) == s2.value:
            return BranchRelation.LIU_HE
        if STEM_HE_MAP.get(s2.value) == s1.value:
            return BranchRelation.LIU_HE
        
        # Check Xiang Chong (相冲) - Six Conflicts
        if STEM_CHONG_MAP.get(s1.value) == s2.value:
            return BranchRelation.LIU_CHONG
        if STEM_CHONG_MAP.get(s2.value) == s1.value:
            return BranchRelation.LIU_CHONG
        
        return None
    
    def analyze_branch_relation(self, b1: TimeBranch, b2: TimeBranch) -> List[BranchRelation]:
        """
        Analyze all relationships between two earthly branches.
        
        Checks for:
        - Liu He (六合): Six Harmonies
        - San He (三合): Three Combinations (checked separately with is_same_trigram)
        - Liu Chong (六冲): Six Conflicts
        - San Xing (三刑): Three Punishments
        - Liu Hai (六害): Six Harms
        - Po (破): Broken relationships
        
        Args:
            b1: First earthly branch
            b2: Second earthly branch
        
        Returns:
            List of BranchRelation values (may be empty)
            
        Example:
            >>> tc = TemporalCore()
            >>> tc.analyze_branch_relation(TimeBranch.ZI, TimeBranch.CHOU)
            [<BranchRelation.LIU_HE: 1>]  # 子丑合
        """
        relations = []
        v1, v2 = b1.value, b2.value
        
        # Liu He (六合) - Six Harmonies
        if BRANCH_HE_MAP.get(v1) == v2 or BRANCH_HE_MAP.get(v2) == v1:
            relations.append(BranchRelation.LIU_HE)
        
        # Liu Chong (六冲) - Six Conflicts
        if BRANCH_CHONG_MAP.get(v1) == v2 or BRANCH_CHONG_MAP.get(v2) == v1:
            relations.append(BranchRelation.LIU_CHONG)
        
        # San Xing (三刑) - Three Punishments
        if self._is_san_xing(b1, b2):
            relations.append(BranchRelation.SAN_XING)
        
        # Liu Hai (六害) - Six Harms
        if self._is_liu_hai(b1, b2):
            relations.append(BranchRelation.LIU_HAI)
        
        # Po (破) - Broken relationships
        if self._is_po(b1, b2):
            relations.append(BranchRelation.PO)
        
        return relations
    
    def _is_san_xing(self, b1: TimeBranch, b2: TimeBranch) -> bool:
        """
        Check if two branches form a San Xing (三刑) relationship.
        
        San Xing patterns:
        - 寅巳申: 3 branches mutually punish
        - 子卯: 子刑卯, 卯刑子
        - 丑戌未: 3 branches mutually punish
        
        Args:
            b1: First branch
            b2: Second branch
        
        Returns:
            True if San Xing relationship exists
        """
        v1, v2 = b1.value, b2.value
        
        # 寅巳申 (寅-巳, 寅-申, 巳-申)
        if (v1, v2) in [(2, 5), (5, 2), (2, 8), (8, 2), (5, 8), (8, 5)]:
            return True
        
        # 子卯 (子-卯)
        if (v1, v2) in [(0, 3), (3, 0)]:
            return True
        
        # 丑戌未 (丑-戌, 戌-丑, 丑-未, 未-丑, 戌-未, 未-戌)
        if (v1, v2) in [(1, 10), (10, 1), (1, 7), (7, 1), (10, 7), (7, 10)]:
            return True
        
        return False
    
    def _is_liu_hai(self, b1: TimeBranch, b2: TimeBranch) -> bool:
        """
        Check if two branches form a Liu Hai (六害) relationship.
        
        Liu Hai patterns:
        子未害, 丑午害, 寅巳害, 卯辰害, 申亥害, 酉戌害
        
        Args:
            b1: First branch
            b2: Second branch
        
        Returns:
            True if Liu Hai relationship exists
        """
        # Define Liu Hai pairs
        liu_hai_pairs = [
            (0, 7),   # 子-未
            (7, 0),   # 未-子
            (1, 6),   # 丑-午
            (6, 1),   # 午-丑
            (2, 5),   # 寅-巳
            (5, 2),   # 巳-寅
            (3, 4),   # 卯-辰
            (4, 3),   # 辰-卯
            (8, 11),  # 申-亥
            (11, 8),  # 亥-申
            (9, 10),  # 酉-戌
            (10, 9),  # 戌-酉
        ]
        return (b1.value, b2.value) in liu_hai_pairs
    
    def _is_po(self, b1: TimeBranch, b2: TimeBranch) -> bool:
        """
        Check if two branches form a Po (破) relationship.
        
        Po patterns:
        子酉破, 寅亥破, 卯午破, 辰丑破, 巳申破, 申巳破, 戌卯破, 亥寅破, 丑辰破, 酉子破, 午卯破
        
        Args:
            b1: First branch
            b2: Second branch
        
        Returns:
            True if Po relationship exists
        """
        # Define Po pairs
        po_pairs = [
            (0, 9), (9, 0),   # 子-酉
            (2, 11), (11, 2), # 寅-亥
            (3, 6), (6, 3),   # 卯-午
            (4, 1), (1, 4),   # 辰-丑
            (5, 8), (8, 5),   # 巳-申, 申-巳
            (10, 3), (3, 10), # 戌-卯
        ]
        return (b1.value, b2.value) in po_pairs
    
    def get_hidden_stems(self, branch: TimeBranch) -> List[TimeStem]:
        """
        Get the hidden stems (藏干) within an earthly branch.
        
        Each earthly branch contains one primary (本气), one secondary (中气),
        and one residual (余气) stem.
        
        Args:
            branch: The earthly branch
        
        Returns:
            List of hidden TimeStem values
            
        Example:
            >>> tc = TemporalCore()
            >>> tc.get_hidden_stems(TimeBranch.YIN)  # 寅
            [<TimeStem.JIA: 0>, <TimeStem.BING: 2>, <TimeStem.WU: 4>]
            # 寅藏: 甲(本气), 丙(中气), 戊(余气)
        """
        stem_indices = BRANCH_HIDDEN_STEM_MAP.get(branch.value, [])
        return [TimeStem(idx) for idx in stem_indices]
    
    def get_cycle_distance(self, idx1: int, idx2: int) -> int:
        """
        Calculate the circular distance between two cycle positions.
        
        Since the cycle wraps around at 60, the distance is calculated
        as the minimum of: |idx1 - idx2| and 60 - |idx1 - idx2|.
        
        The result is in range 0-30.
        
        Args:
            idx1: First cycle position (0-59)
            idx2: Second cycle position (0-59)
        
        Returns:
            Circular distance (0-30)
            
        Example:
            >>> tc = TemporalCore()
            >>> tc.get_cycle_distance(0, 59)  # Circular wrap
            1
            >>> tc.get_cycle_distance(0, 30)  # Maximum distance
            30
        """
        idx1 = idx1 % 60
        idx2 = idx2 % 60
        diff = abs(idx1 - idx2)
        return min(diff, 60 - diff)
    
    def is_same_trigram(self, code1: StemBranchCode, code2: StemBranchCode, 
                        code3: Optional[StemBranchCode] = None) -> Tuple[bool, Optional[str]]:
        """
        Check if one or more codes form a San He (三合局) trigram.
        
        The San He (三合局) patterns are:
        - 申子辰 (Shen-Zi-Chen): Water trigram
        - 亥卯未 (Hai-Mao-Wei): Wood trigram
        - 寅午戌 (Yin-Wu-Xu): Fire trigram
        - 巳酉丑 (Si-You-Chou): Metal trigram
        
        Args:
            code1: First stem-branch code
            code2: Second stem-branch code
            code3: Optional third stem-branch code (if None, checks if code1 and code2
                   are part of the same trigram)
        
        Returns:
            Tuple of (is_trigram, energy_type) where:
            - is_trigram: True if a valid San He pattern is formed
            - energy_type: The energy type of the trigram ('water', 'wood', 'fire', 'metal')
                          or None if not a trigram
            
        Example:
            >>> tc = TemporalCore()
            >>> code_sh = tc.create_code(6, 8)   # 庚申
            >>> code_zi = tc.create_code(8, 0)   # 壬子
            >>> code_ch = tc.create_code(4, 4)   # 戊辰
            >>> tc.is_same_trigram(code_sh, code_zi, code_ch)
            (True, 'water')  # 申子辰合水局
        """
        # If only 2 codes provided, check if they could be part of a trigram
        if code3 is None:
            # For two codes, we check if they're adjacent in a trigram
            branch_set = frozenset([code1.branch.value, code2.branch.value])
            for trigram_branches, energy in BRANCH_SANHE_MAP.items():
                if branch_set.issubset(trigram_branches):
                    return True, energy
            return False, None
        
        # For 3 codes, check if they form a complete trigram
        branch_set = frozenset([code1.branch.value, code2.branch.value, code3.branch.value])
        for trigram_branches, energy in BRANCH_SANHE_MAP.items():
            if branch_set == trigram_branches:
                return True, energy
        
        return False, None
    
    def is_same_trigram_set(self, codes: List[StemBranchCode]) -> Tuple[bool, Optional[str]]:
        """
        Check if a list of codes form a San He (三合局) trigram.
        
        Args:
            codes: List of StemBranchCode instances
        
        Returns:
            Tuple of (is_trigram, energy_type)
        """
        if len(codes) < 2:
            return False, None
        
        branch_set = frozenset(c.branch.value for c in codes)
        
        # Check if subset matches (for partial trigrams)
        for trigram_branches, energy in BRANCH_SANHE_MAP.items():
            if branch_set.issubset(trigram_branches):
                return True, energy
        
        return False, None
    
    def get_branch_energy(self, branch: TimeBranch) -> str:
        """
        Get the primary energy type (本气) of an earthly branch.
        
        Args:
            branch: The earthly branch
        
        Returns:
            Energy type: 'wood', 'fire', 'earth', 'metal', 'water'
            
        Example:
            >>> tc = TemporalCore()
            >>> tc.get_branch_energy(TimeBranch.ZI)
            'water'
            >>> tc.get_branch_energy(TimeBranch.MAO)
            'wood'
        """
        return BRANCH_PRIMARY_ENERGY.get(branch, "earth")
    
    def get_stem_energy(self, stem: TimeStem) -> str:
        """
        Get the energy type of a heavenly stem.
        
        Args:
            stem: The heavenly stem
        
        Returns:
            Energy type: 'wood', 'fire', 'earth', 'metal', 'water'
        """
        energies = ["wood", "wood", "fire", "fire", "earth", "earth", 
                   "metal", "metal", "water", "water"]
        return energies[stem.value]
    
    def is_stem_yang(self, stem: TimeStem) -> bool:
        """
        Check if a stem is Yang (阳).
        
        Args:
            stem: The heavenly stem
        
        Returns:
            True if Yang (even index), False if Yin (odd index)
        """
        return stem.value % 2 == 0
    
    def is_branch_yang(self, branch: TimeBranch) -> bool:
        """
        Check if a branch is Yang (阳).
        
        Args:
            branch: The earthly branch
        
        Returns:
            True if Yang (even index), False if Yin (odd index)
        """
        return branch.value % 2 == 0
    
    def get_san_he_branches(self, energy_type: str) -> List[TimeBranch]:
        """
        Get the three branches that form a San He trigram for a given energy type.
        
        Args:
            energy_type: 'water', 'wood', 'fire', or 'metal'
        
        Returns:
            List of three TimeBranch values forming the trigram
            
        Example:
            >>> tc = TemporalCore()
            >>> tc.get_san_he_branches('water')
            [<TimeBranch.SHEN: 8>, <TimeBranch.ZI: 0>, <TimeBranch.CHEN: 4>]
        """
        trigram_map = {
            "water": [TimeBranch.SHEN, TimeBranch.ZI, TimeBranch.CHEN],
            "wood": [TimeBranch.HAI, TimeBranch.MAO, TimeBranch.WEI],
            "fire": [TimeBranch.YIN, TimeBranch.WU, TimeBranch.XU],
            "metal": [TimeBranch.SI, TimeBranch.YOU, TimeBranch.CHOU],
        }
        return trigram_map.get(energy_type, [])
    
    def __repr__(self) -> str:
        return f"TemporalCore(cycle_length=60)"


# =============================================================================
# Convenience Functions
# =============================================================================

def create_stem_branch(stem_idx: int, branch_idx: int) -> StemBranchCode:
    """
    Convenience function to create a StemBranchCode.
    
    Args:
        stem_idx: Heavenly stem index (0-9)
        branch_idx: Earthly branch index (0-11)
    
    Returns:
        StemBranchCode instance
        
    Example:
        >>> code = create_stem_branch(0, 0)  # 甲子
        >>> code.name
        '甲子'
    """
    core = TemporalCore()
    return core.create_code(stem_idx, branch_idx)


def get_cycle_name(index: int) -> str:
    """
    Convenience function to get the name at a cycle position.
    
    Args:
        index: Cycle position (0-59)
    
    Returns:
        Stem-branch name
    """
    core = TemporalCore()
    return core.get_cycle_name(index)


# =============================================================================
# Test Suite
# =============================================================================

def _run_tests():
    """Run built-in test cases."""
    print("=" * 60)
    print("TemporalCore Test Suite")
    print("=" * 60)
    
    tc = TemporalCore()
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
    
    print("\n[1] StemBranchCode Creation Tests")
    print("-" * 40)
    
    # Test 甲子 creation
    code = tc.create_code(0, 0)  # 甲子
    test("create_code(0, 0) creates 甲子", code.name == "甲子")
    test("甲子 cycle_index is 0", code.cycle_index == 0)
    test("甲子 polarity is yang", code.polarity == "yang")
    
    # Test 癸亥 creation (index 59)
    code = tc.create_code(9, 11)  # 癸亥
    test("create_code(9, 11) creates 癸亥", code.name == "癸亥")
    test("癸亥 cycle_index is 59", code.cycle_index == 59)
    test("癸亥 polarity is yin", code.polarity == "yin")
    
    print("\n[2] Cycle Navigation Tests")
    print("-" * 40)
    
    test("get_cycle_name(0) returns 甲子", tc.get_cycle_name(0) == "甲子")
    test("get_cycle_name(59) returns 癸亥", tc.get_cycle_name(59) == "癸亥")
    test("get_cycle_name(60) wraps to 甲子", tc.get_cycle_name(60) == "甲子")
    
    idx = tc.get_cycle_index(TimeStem.JIA, TimeBranch.ZI)
    test("get_cycle_index(JIA, ZI) returns 0", idx == 0)
    
    idx = tc.get_cycle_index(TimeStem.GUI, TimeBranch.HAI)
    test("get_cycle_index(GUI, HAI) returns 59", idx == 59)
    
    print("\n[3] Heavenly Stem Relation Tests")
    print("-" * 40)
    
    # Wu He (五合) - Five Combinations
    rel = tc.analyze_stem_relation(TimeStem.JIA, TimeStem.JI)
    test("甲己合 (JIA-JI)", rel == BranchRelation.LIU_HE)
    
    rel = tc.analyze_stem_relation(TimeStem.YI, TimeStem.GENG)
    test("乙庚合 (YI-GENG)", rel == BranchRelation.LIU_HE)
    
    rel = tc.analyze_stem_relation(TimeStem.BING, TimeStem.XIN)
    test("丙辛合 (BING-XIN)", rel == BranchRelation.LIU_HE)
    
    rel = tc.analyze_stem_relation(TimeStem.DING, TimeStem.REN)
    test("丁壬合 (DING-REN)", rel == BranchRelation.LIU_HE)
    
    rel = tc.analyze_stem_relation(TimeStem.WU, TimeStem.GUI)
    test("戊癸合 (WU-GUI)", rel == BranchRelation.LIU_HE)
    
    # Xiang Chong (相冲) - Six Conflicts
    rel = tc.analyze_stem_relation(TimeStem.JIA, TimeStem.GENG)
    test("甲庚冲 (JIA-GENG)", rel == BranchRelation.LIU_CHONG)
    
    rel = tc.analyze_stem_relation(TimeStem.YI, TimeStem.XIN)
    test("乙辛冲 (YI-XIN)", rel == BranchRelation.LIU_CHONG)
    
    rel = tc.analyze_stem_relation(TimeStem.BING, TimeStem.REN)
    test("丙壬冲 (BING-REN)", rel == BranchRelation.LIU_CHONG)
    
    rel = tc.analyze_stem_relation(TimeStem.DING, TimeStem.GUI)
    test("丁癸冲 (DING-GUI)", rel == BranchRelation.LIU_CHONG)
    
    # No relation
    rel = tc.analyze_stem_relation(TimeStem.JIA, TimeStem.YI)
    test("甲乙无合冲关系", rel is None)
    
    print("\n[4] Earthly Branch Relation Tests")
    print("-" * 40)
    
    # Liu He (六合)
    rels = tc.analyze_branch_relation(TimeBranch.ZI, TimeBranch.CHOU)
    test("子丑合 (ZI-CHOU)", BranchRelation.LIU_HE in rels)
    
    rels = tc.analyze_branch_relation(TimeBranch.YIN, TimeBranch.HAI)
    test("寅亥合 (YIN-HAI)", BranchRelation.LIU_HE in rels)
    
    rels = tc.analyze_branch_relation(TimeBranch.MAO, TimeBranch.XU)
    test("卯戌合 (MAO-XU)", BranchRelation.LIU_HE in rels)
    
    rels = tc.analyze_branch_relation(TimeBranch.CHEN, TimeBranch.YOU)
    test("辰酉合 (CHEN-YOU)", BranchRelation.LIU_HE in rels)
    
    rels = tc.analyze_branch_relation(TimeBranch.SI, TimeBranch.SHEN)
    test("巳申合 (SI-SHEN)", BranchRelation.LIU_HE in rels)
    
    rels = tc.analyze_branch_relation(TimeBranch.WU, TimeBranch.WEI)
    test("午未合 (WU-WEI)", BranchRelation.LIU_HE in rels)
    
    # Liu Chong (六冲)
    rels = tc.analyze_branch_relation(TimeBranch.ZI, TimeBranch.WU)
    test("子午冲 (ZI-WU)", BranchRelation.LIU_CHONG in rels)
    
    rels = tc.analyze_branch_relation(TimeBranch.CHOU, TimeBranch.WEI)
    test("丑未冲 (CHOU-WEI)", BranchRelation.LIU_CHONG in rels)
    
    rels = tc.analyze_branch_relation(TimeBranch.YIN, TimeBranch.SHEN)
    test("寅申冲 (YIN-SHEN)", BranchRelation.LIU_CHONG in rels)
    
    rels = tc.analyze_branch_relation(TimeBranch.MAO, TimeBranch.YOU)
    test("卯酉冲 (MAO-YOU)", BranchRelation.LIU_CHONG in rels)
    
    rels = tc.analyze_branch_relation(TimeBranch.CHEN, TimeBranch.XU)
    test("辰戌冲 (CHEN-XU)", BranchRelation.LIU_CHONG in rels)
    
    rels = tc.analyze_branch_relation(TimeBranch.SI, TimeBranch.HAI)
    test("巳亥冲 (SI-HAI)", BranchRelation.LIU_CHONG in rels)
    
    print("\n[5] Hidden Stems Tests")
    print("-" * 40)
    
    # 子藏癸 (index 8 = REN)
    stems = tc.get_hidden_stems(TimeBranch.ZI)
    test("子藏癸 (ZI hidden REN)", len(stems) == 1 and stems[0] == TimeStem.REN)
    
    # 寅藏甲丙戊
    stems = tc.get_hidden_stems(TimeBranch.YIN)
    test("寅藏甲丙戊 (YIN hidden JIA, BING, WU)", 
         len(stems) == 3 and 
         TimeStem.JIA in stems and TimeStem.BING in stems and TimeStem.WU in stems)
    
    # 亥藏壬甲
    stems = tc.get_hidden_stems(TimeBranch.HAI)
    test("亥藏壬甲 (HAI hidden REN, JIA)", 
         len(stems) == 2 and 
         TimeStem.REN in stems and TimeStem.JIA in stems)
    
    code = tc.create_code(0, 0)  # 甲子
    test("StemBranchCode.hidden_stems property", 
         len(code.hidden_stems) == 1 and code.hidden_stems[0] == TimeStem.REN)
    
    print("\n[6] San He (三合局) Trigram Tests")
    print("-" * 40)
    
    # 申子辰 - 水局
    code_sh = tc.create_code(6, 8)   # 庚申
    code_zi = tc.create_code(8, 0)   # 壬子
    code_ch = tc.create_code(4, 4)   # 戊辰
    
    is_tri, energy = tc.is_same_trigram(code_sh, code_zi, code_ch)
    test("申子辰合水局 (SHEN-ZI-CHEN)", is_tri and energy == "water")
    
    # Partial trigram check (2 branches)
    is_tri, energy = tc.is_same_trigram(code_sh, code_zi)
    test("申子同属水局 (partial)", is_tri and energy == "water")
    
    # 亥卯未 - 木局
    code_h = tc.create_code(8, 11)   # 壬亥
    code_m = tc.create_code(1, 3)    # 乙卯
    code_w = tc.create_code(5, 7)    # 己未
    
    is_tri, energy = tc.is_same_trigram(code_h, code_m, code_w)
    test("亥卯未合木局 (HAI-MAO-WEI)", is_tri and energy == "wood")
    
    # 寅午戌 - 火局
    code_y = tc.create_code(0, 2)    # 甲寅
    code_w = tc.create_code(3, 6)    # 丁午
    code_x = tc.create_code(4, 10)   # 戊戌
    
    is_tri, energy = tc.is_same_trigram(code_y, code_w, code_x)
    test("寅午戌合火局 (YIN-WU-XU)", is_tri and energy == "fire")
    
    # 巳酉丑 - 金局
    code_s = tc.create_code(2, 5)    # 丙巳
    code_y = tc.create_code(7, 9)    # 辛酉
    code_c = tc.create_code(5, 1)    # 己丑
    
    is_tri, energy = tc.is_same_trigram(code_s, code_y, code_c)
    test("巳酉丑合金局 (SI-YOU-CHOU)", is_tri and energy == "metal")
    
    print("\n[7] Cycle Distance Tests")
    print("-" * 40)
    
    test("get_cycle_distance(0, 59) wraps to 1", tc.get_cycle_distance(0, 59) == 1)
    test("get_cycle_distance(0, 30) is 30", tc.get_cycle_distance(0, 30) == 30)
    test("get_cycle_distance(10, 20) is 10", tc.get_cycle_distance(10, 20) == 10)
    test("get_cycle_distance(30, 0) is 30", tc.get_cycle_distance(30, 0) == 30)
    
    print("\n[8] Energy Type Tests")
    print("-" * 40)
    
    test("get_branch_energy(ZI) = water", tc.get_branch_energy(TimeBranch.ZI) == "water")
    test("get_branch_energy(YIN) = wood", tc.get_branch_energy(TimeBranch.YIN) == "wood")
    test("get_branch_energy(SI) = fire", tc.get_branch_energy(TimeBranch.SI) == "fire")
    test("get_branch_energy(SHEN) = metal", tc.get_branch_energy(TimeBranch.SHEN) == "metal")
    test("get_branch_energy(CHEN) = earth", tc.get_branch_energy(TimeBranch.CHEN) == "earth")
    
    test("get_stem_energy(JIA) = wood", tc.get_stem_energy(TimeStem.JIA) == "wood")
    test("get_stem_energy(BING) = fire", tc.get_stem_energy(TimeStem.BING) == "fire")
    test("get_stem_energy(WU) = earth", tc.get_stem_energy(TimeStem.WU) == "earth")
    test("get_stem_energy(GENG) = metal", tc.get_stem_energy(TimeStem.GENG) == "metal")
    test("get_stem_energy(REN) = water", tc.get_stem_energy(TimeStem.REN) == "water")
    
    print("\n[9] Polarity Tests")
    print("-" * 40)
    
    test("JIA is yang", tc.is_stem_yang(TimeStem.JIA) == True)
    test("YI is yin", tc.is_stem_yang(TimeStem.YI) == False)
    test("ZI is yang", tc.is_branch_yang(TimeBranch.ZI) == True)
    test("CHOU is yin", tc.is_branch_yang(TimeBranch.CHOU) == False)
    
    test("code.polarity for 甲子", tc.create_code(0, 0).polarity == "yang")
    test("code.polarity for 乙丑", tc.create_code(1, 1).polarity == "yin")
    
    print("\n[10] San He Branch Retrieval")
    print("-" * 40)
    
    branches = tc.get_san_he_branches("water")
    test("水局三合: 申子辰", 
         len(branches) == 3 and 
         TimeBranch.SHEN in branches and 
         TimeBranch.ZI in branches and 
         TimeBranch.CHEN in branches)
    
    branches = tc.get_san_he_branches("wood")
    test("木局三合: 亥卯未", 
         len(branches) == 3 and 
         TimeBranch.HAI in branches and 
         TimeBranch.MAO in branches and 
         TimeBranch.WEI in branches)
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    _run_tests()
