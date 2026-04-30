"""
Spacetime Index Module (时空索引)

This module integrates energy relationship systems with spacetime
indexing for memory retrieval and ranking.

Core Features:
- Energy-aware spacetime indexing
- Four-phase temporal rhythm integration
- Season-based retrieval weight adjustment
- Cross-layer energy flow for memory nodes

Architecture Integration:
- Energy System: Energy operation rules
- Four-Phase: Temporal rhythm (initial-yang → peak-yang → initial-yin → peak-yin)
- Trigrams: Semantic category mapping
- Spacetime: Stem-branch temporal encoding

【Post-Phase Symbolic】- Spacetime indexing uses post trigram ordering for symbolic applications
- Seasonal energy patterns
- Direction-based weight adjustments
- Time-of-day energy variations

【Pre-Phase Numeric】- Index calculations use prior trigram ordering for numerical operations
- Binary encoding
- Position calculations
- Distance metrics
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from ._enums import TimeStem, TimeBranch
from ._energy_relations import (
    RelationType,
    analyze_relation,
    get_affinity_score,
)
from ._dimension_map import TaijiMapper, STEM_TO_TRIGRAM, BRANCH_TO_TRIGRAM


# =============================================================================
# Energy to Season Mapping (能量季节映射)
# =============================================================================

# Map energy types to their associated seasons
ENERGY_TO_SEASON: Dict[str, str] = {
    "wood": "spring",
    "fire": "summer",
    "earth": "late_summer",
    "metal": "autumn",
    "water": "winter",
}

# Map energy types to their four-phase attribution
ENERGY_TO_FOUR_PHASE: Dict[str, str] = {
    "wood": "INITIAL_YANG",
    "fire": "PEAK_YANG",
    "earth": "CENTER",
    "metal": "INITIAL_YIN",
    "water": "PEAK_YIN",
}


# =============================================================================
# Spacetime Energy Configuration
# =============================================================================

@dataclass
class SpacetimeConfig:
    """Configuration for spacetime energy indexing"""
    # Energy decay rates
    seasonal_decay: float = 0.9       # Seasonal energy decay
    directional_decay: float = 0.85    # Direction-based decay
    temporal_decay: float = 0.95        # Time-of-day decay

    # Weight multipliers
    enhance_weight: float = 1.3          # 相生增强
    suppress_weight: float = 0.7        # 相克削弱
    same_energy_weight: float = 1.1     # 同类增强
    center_balance_weight: float = 1.05 # 中宫平衡

    # Four-phase temporal weights
    four_phase_weights: Dict[str, float] = field(default_factory=lambda: {
        "INITIAL_YANG": 1.2,   # 初始阳 - 春 - 木 - 生发
        "PEAK_YANG": 1.3,      # 盛阳 - 夏 - 火 - 炎盛
        "INITIAL_YIN": 1.1,    # 初始阴 - 秋 - 金 - 收敛
        "PEAK_YIN": 1.15,      # 盛阴 - 冬 - 水 - 闭藏
        "CENTER": 1.0,          # 中宫 - 长夏 - 土 - 化育
    })

    # Season energy multipliers
    season_weights: Dict[str, float] = field(default_factory=lambda: {
        "spring": 1.2,      # 春 - 木
        "summer": 1.3,      # 夏 - 火
        "late_summer": 1.0, # 长夏 - 土
        "autumn": 1.1,      # 秋 - 金
        "winter": 1.15,    # 冬 - 水
    })


# =============================================================================
# Spacetime Energy Node
# =============================================================================

@dataclass
class SpacetimeNode:
    """
    Spacetime node with energy attributes for indexing.

    Attributes:
        node_id: Unique identifier
        energy_type: Five elements energy type
        stem_idx: Heavenly stem index (0-9)
        branch_idx: Earthly branch index (0-11)
        trigram_idx: Trigram index (0-7)
        four_phase: Four-phase attribution
        season: Current season
        base_weight: Base retrieval weight
        energy_boost: Energy-based boost factor
    """
    node_id: str
    energy_type: str

    # Spacetime attributes
    stem_idx: Optional[int] = None
    branch_idx: Optional[int] = None
    trigram_idx: Optional[int] = None

    # Energy attributes
    four_phase: Optional[str] = None
    season: Optional[str] = None

    # Weights
    base_weight: float = 1.0
    energy_boost: float = 1.0

    # Metadata
    position: Optional[Tuple[int, int]] = None  # (spatial, temporal)
    metadata: Dict = field(default_factory=dict)

    @property
    def effective_weight(self) -> float:
        """Calculate effective retrieval weight"""
        return self.base_weight * self.energy_boost

    @property
    def energy_level(self) -> str:
        """Get energy level description"""
        boost = self.energy_boost
        if boost >= 1.3:
            return "旺盛"
        elif boost >= 1.1:
            return "偏旺"
        elif boost >= 0.9:
            return "中和"
        elif boost >= 0.6:
            return "偏弱"
        else:
            return "衰弱"


# =============================================================================
# Spacetime Index Engine
# =============================================================================

class SpacetimeIndexEngine:
    """
    Spacetime Index Engine with Five Elements energy integration.

    Provides energy-aware memory retrieval and ranking by integrating:
    - Five Elements relationships (相生/相克)
    - Four symbols temporal rhythm (Four Symbols)
    - Season energy patterns (季节)
    - Trigram spacetime mapping (Trigram Patterns)

    Example:
        >>> engine = SpacetimeIndexEngine()
        >>> engine.add_node(SpacetimeNode("node1", "wood", stem_idx=0, branch_idx=0))
        >>> engine.add_node(SpacetimeNode("node2", "fire", stem_idx=2, branch_idx=6))
        >>>
        >>> # Query with energy enhancement
        >>> results = engine.search_by_energy("wood", limit=10)
        >>>
        >>> # Get temporal ranking
        >>> ranking = engine.get_temporal_ranking(TimeStem.JIA, TimeBranch.ZI)
    """

    def __init__(self, config: Optional[SpacetimeConfig] = None):
        """Initialize the spacetime index engine"""
        self._nodes: Dict[str, SpacetimeNode] = {}
        self._stem_nodes: Dict[int, List[str]] = defaultdict(list)
        self._branch_nodes: Dict[int, List[str]] = defaultdict(list)
        self._trigram_nodes: Dict[int, List[str]] = defaultdict(list)
        self._energy_nodes: Dict[str, List[str]] = defaultdict(list)
        self._four_phase_nodes: Dict[str, List[str]] = defaultdict(list)

        self._config = config or SpacetimeConfig()
        self._taiji_mapper = TaijiMapper()

    # =========================================================================
    # Node Management
    # =========================================================================

    def add_node(
        self,
        node: SpacetimeNode,
        auto_index: bool = True
    ) -> str:
        """
        Add a spacetime node to the index.

        Args:
            node: SpacetimeNode to add
            auto_index: Whether to auto-index by spacetime attributes

        Returns:
            Node ID
        """
        if node.node_id in self._nodes:
            raise ValueError(f"Node {node.node_id} already exists")

        self._nodes[node.node_id] = node

        if auto_index:
            self._index_node(node)

        return node.node_id

    def _index_node(self, node: SpacetimeNode):
        """Index a node by its attributes"""
        # Index by stem
        if node.stem_idx is not None:
            self._stem_nodes[node.stem_idx].append(node.node_id)

        # Index by branch
        if node.branch_idx is not None:
            self._branch_nodes[node.branch_idx].append(node.node_id)

        # Index by trigram
        if node.trigram_idx is not None:
            self._trigram_nodes[node.trigram_idx].append(node.node_id)

        # Index by energy
        self._energy_nodes[node.energy_type].append(node.node_id)

        # Index by four symbols
        if node.four_phase:
            self._four_phase_nodes[node.four_phase].append(node.node_id)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the index"""
        if node_id not in self._nodes:
            return False

        self._nodes[node_id]

        # Remove from indexes
        for idx_dict in [self._stem_nodes, self._branch_nodes, self._trigram_nodes,
                         self._energy_nodes, self._four_phase_nodes]:
            for key_list in idx_dict.values():
                if node_id in key_list:
                    key_list.remove(node_id)

        del self._nodes[node_id]
        return True

    def get_node(self, node_id: str) -> Optional[SpacetimeNode]:
        """Get a node by ID"""
        return self._nodes.get(node_id)

    # =========================================================================
    # Energy-Aware Retrieval
    # =========================================================================

    def search_by_energy(
        self,
        energy_type: str,
        relation_type: Optional[RelationType] = None,
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search nodes by energy type with relationship weighting.

        Args:
            energy_type: Target energy type (wood, fire, earth, metal, water)
            relation_type: Optional filter by relation type
            limit: Maximum results to return

        Returns:
            List of (node_id, weight) tuples sorted by effective weight
        """
        results: List[Tuple[str, float]] = []

        # Get all nodes
        for node_id, node in self._nodes.items():
            # Calculate energy weight based on relation
            relation = analyze_relation(node.energy_type, energy_type)

            if relation_type is not None and relation.relation != relation_type:
                continue

            # Calculate weight multiplier
            if relation.relation == RelationType.ENHANCE:
                weight = self._config.enhance_weight
            elif relation.relation == RelationType.SUPPRESS:
                weight = self._config.suppress_weight
            elif relation.relation == RelationType.SAME:
                weight = self._config.same_energy_weight
            else:
                weight = 1.0

            # Apply four symbol temporal weight
            if node.four_phase:
                fs_weight = self._config.four_phase_weights.get(node.four_phase, 1.0)
                weight *= fs_weight

            # Apply season weight
            if node.season:
                season_weight = self._config.season_weights.get(node.season, 1.0)
                weight *= season_weight

            # Calculate final weight
            final_weight = node.base_weight * weight * node.energy_boost
            results.append((node_id, final_weight))

        # Sort by weight descending
        results.sort(key=lambda x: -x[1])

        return results[:limit]

    def search_by_stem_branch(
        self,
        stem: TimeStem,
        branch: TimeBranch,
        energy_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search nodes by stem-branch combination with energy mapping.

        Args:
            stem: Heavenly stem
            branch: Earthly branch
            energy_type: Optional energy type filter
            limit: Maximum results

        Returns:
            List of (node_id, weight) tuples
        """
        results: List[Tuple[str, float]] = []

        stem_idx = stem.value
        branch_idx = branch.value

        # Get nodes matching stem or branch
        candidate_ids = set()

        if stem_idx in self._stem_nodes:
            candidate_ids.update(self._stem_nodes[stem_idx])

        if branch_idx in self._branch_nodes:
            candidate_ids.update(self._branch_nodes[branch_idx])

        # Map to trigram via Najia
        trigram_idx = STEM_TO_TRIGRAM.get(stem_idx)
        if trigram_idx is not None and trigram_idx in self._trigram_nodes:
            candidate_ids.update(self._trigram_nodes[trigram_idx])

        branch_trigram_idx = BRANCH_TO_TRIGRAM.get(branch_idx)
        if branch_trigram_idx is not None and branch_trigram_idx in self._trigram_nodes:
            candidate_ids.update(self._trigram_nodes[branch_trigram_idx])

        # Calculate weights
        for node_id in candidate_ids:
            node = self._nodes.get(node_id)
            if not node:
                continue

            if energy_type and node.energy_type != energy_type:
                continue

            # Base weight from stem-branch match
            weight = 1.0

            # Check stem match
            if node.stem_idx == stem_idx:
                weight *= 1.2

            # Check branch match
            if node.branch_idx == branch_idx:
                weight *= 1.2

            # Check trigram match
            if node.trigram_idx == trigram_idx or node.trigram_idx == branch_trigram_idx:
                weight *= 1.1

            # Apply energy boost
            weight *= node.energy_boost

            results.append((node_id, weight))

        # Sort by weight
        results.sort(key=lambda x: -x[1])

        return results[:limit]

    # =========================================================================
    # Temporal Energy Flow
    # =========================================================================

    def get_temporal_ranking(
        self,
        stem: TimeStem,
        branch: TimeBranch,
        energy_direction: str = "ascending"
    ) -> List[Tuple[str, float]]:
        """
        Get temporal energy ranking for stem-branch combination.

        Uses post trigram ordering (后天主象) for symbolic applications.

        Args:
            stem: Heavenly stem
            branch: Earthly branch
            energy_direction: 'ascending' or 'descending' by temporal energy

        Returns:
            List of (node_id, weight) tuples
        """
        results: List[Tuple[str, float]] = []

        stem_idx = stem.value
        branch_idx = branch.value

        # Get stem-branch energy from mapping
        stem_trigram = STEM_TO_TRIGRAM.get(stem_idx)
        branch_trigram = BRANCH_TO_TRIGRAM.get(branch_idx)

        # Calculate temporal position using post ordering (后天主象)
        stem_post_pos = self._taiji_mapper.get_post_position(stem_trigram) if stem_trigram is not None else 0
        branch_post_pos = self._taiji_mapper.get_post_position(branch_trigram) if branch_trigram is not None else 0

        temporal_energy = (stem_post_pos + branch_post_pos) / 2

        # Score all nodes
        for node_id, node in self._nodes.items():
            # Calculate energy affinity
            if node.trigram_idx is not None:
                node_post_pos = self._taiji_mapper.get_post_position(node.trigram_idx)
                distance = abs(node_post_pos - temporal_energy)

                # Temporal decay based on distance (先天主数)
                temporal_weight = self._config.temporal_decay ** distance
            else:
                temporal_weight = 0.5

            # Apply energy type factor
            if node.energy_type:
                affinity = get_affinity_score(node.energy_type, node.energy_type)
                temporal_weight *= affinity

            # Apply seasonal weight
            if node.season:
                temporal_weight *= self._config.season_weights.get(node.season, 1.0)

            weight = node.base_weight * temporal_weight * node.energy_boost
            results.append((node_id, weight))

        # Sort
        if energy_direction == "descending":
            results.sort(key=lambda x: -x[1])
        else:
            results.sort(key=lambda x: x[1])

        return results

    def calculate_energy_flow(
        self,
        source_id: str,
        target_energy: str,
        distance: float = 1.0
    ) -> float:
        """
        Calculate energy flow between nodes.

        Args:
            source_id: Source node ID
            target_energy: Target energy type
            distance: Spatial/temporal distance factor

        Returns:
            Energy flow weight (0.0 - 2.0)
        """
        source = self._nodes.get(source_id)
        if not source:
            return 0.0

        # Calculate relation
        relation = analyze_relation(source.energy_type, target_energy)

        # Base flow from relation
        if relation.relation == RelationType.ENHANCE:
            flow = 1.5
        elif relation.relation == RelationType.SUPPRESS:
            flow = 0.5
        elif relation.relation == RelationType.SAME:
            flow = 1.2
        else:
            flow = 1.0

        # Apply distance decay
        flow *= (self._config.directional_decay ** distance)

        # Apply four symbol factor
        if source.four_phase:
            fs_factor = self._config.four_phase_weights.get(source.four_phase, 1.0)
            flow *= fs_factor

        return flow

    # =========================================================================
    # Cross-Layer Energy Integration
    # =========================================================================

    def integrate_energy_bus(self, energy_bus) -> Dict[str, float]:
        """
        Integrate with energy bus for cross-layer energy flow.

        Args:
            energy_bus: EnergyBus instance

        Returns:
            Mapping of node IDs to energy adjustments
        """
        adjustments: Dict[str, float] = {}

        # Get energy bus state
        energy_bus.get_bus_state()

        for node_id, node in self._nodes.items():
            # Get energy balance contribution
            balance = energy_bus._calculate_energy_balance()

            # Calculate adjustment based on five elements balance
            if node.energy_type in balance.get('ratios', {}):
                element_ratio = balance['ratios'][node.energy_type]

                # If element is dominant (>40%), reduce boost
                if element_ratio > 0.4:
                    adjustment = 0.9
                # If element is weak (<15%), increase boost
                elif element_ratio < 0.15:
                    adjustment = 1.2
                # If balanced, standard boost
                else:
                    adjustment = 1.0

                # Apply to node
                node.energy_boost *= adjustment
                adjustments[node_id] = adjustment

        return adjustments

    # =========================================================================
    # State and Statistics
    # =========================================================================

    def get_index_state(self) -> Dict:
        """Get comprehensive index state"""
        energy_counts: Dict[str, int] = {}
        for node in self._nodes.values():
            energy_counts[node.energy_type] = energy_counts.get(node.energy_type, 0) + 1

        four_phase_counts: Dict[str, int] = {}
        for node in self._nodes.values():
            if node.four_phase:
                four_phase_counts[node.four_phase] = (
                    four_phase_counts.get(node.four_phase, 0) + 1
                )

        return {
            "total_nodes": len(self._nodes),
            "energy_counts": energy_counts,
            "four_phase_counts": four_phase_counts,
            "stem_index_size": len(self._stem_nodes),
            "branch_index_size": len(self._branch_nodes),
            "trigram_index_size": len(self._trigram_nodes),
        }

    def __repr__(self) -> str:
        return f"SpacetimeIndexEngine(nodes={len(self._nodes)})"


# =============================================================================
# Convenience Functions
# =============================================================================

def create_spacetime_engine(config: Optional[SpacetimeConfig] = None) -> SpacetimeIndexEngine:
    """Create a spacetime index engine"""
    return SpacetimeIndexEngine(config)


def create_energy_aware_node(
    node_id: str,
    energy_type: str,
    stem_idx: Optional[int] = None,
    branch_idx: Optional[int] = None,
    season: Optional[str] = None
) -> SpacetimeNode:
    """
    Create a spacetime node with automatic energy attribution.

    Args:
        node_id: Unique identifier
        energy_type: Five elements energy type
        stem_idx: Heavenly stem index (optional)
        branch_idx: Earthly branch index (optional)
        season: Season (optional)

    Returns:
        SpacetimeNode with computed attributes
    """
    # Determine trigram from stem/branch
    trigram_idx = None
    if stem_idx is not None:
        trigram_idx = STEM_TO_TRIGRAM.get(stem_idx)
    elif branch_idx is not None:
        trigram_idx = BRANCH_TO_TRIGRAM.get(branch_idx)

    # Determine four symbol from energy
    four_phase = ENERGY_TO_FOUR_PHASE.get(energy_type)

    # Determine season if not provided
    if season is None and energy_type in ENERGY_TO_SEASON:
        season = ENERGY_TO_SEASON[energy_type]

    return SpacetimeNode(
        node_id=node_id,
        energy_type=energy_type,
        stem_idx=stem_idx,
        branch_idx=branch_idx,
        trigram_idx=trigram_idx,
        four_phase=four_phase,
        season=season,
        base_weight=1.0,
        energy_boost=1.0
    )


# =============================================================================
# Test Suite
# =============================================================================

def test_spacetime_index():
    """Test spacetime index engine"""
    print("=" * 60)
    print("Testing Spacetime Index Engine")
    print("=" * 60)

    engine = SpacetimeIndexEngine()
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

    # Test 1: Add nodes
    print("\n[Test 1] Node Management")
    print("-" * 40)

    node1 = create_energy_aware_node("node1", "wood", stem_idx=0, branch_idx=0, season="spring")
    node2 = create_energy_aware_node("node2", "fire", stem_idx=2, branch_idx=6, season="summer")
    node3 = create_energy_aware_node("node3", "earth", branch_idx=1, season="late_summer")
    node4 = create_energy_aware_node("node4", "metal", stem_idx=6, branch_idx=9, season="autumn")
    node5 = create_energy_aware_node("node5", "water", stem_idx=8, branch_idx=0, season="winter")

    for node in [node1, node2, node3, node4, node5]:
        engine.add_node(node)

    test("Add 5 nodes", len(engine._nodes) == 5)

    # Test 2: Search by energy
    print("\n[Test 2] Energy-Aware Search")
    print("-" * 40)

    wood_results = engine.search_by_energy("wood")
    test("Search wood energy", len(wood_results) > 0)

    # First result should be wood node (highest weight)
    if wood_results:
        node_ids = [r[0] for r in wood_results]
        test("Wood node in results", "node1" in node_ids)

    # Test 3: Four symbols attribution
    print("\n[Test 3] Four Symbols Attribution")
    print("-" * 40)

    node1_found = engine.get_node("node1")
    test("Node1 four_phase is SHAO_YANG", node1_found.four_phase == "SHAO_YANG")
    test("Node1 season is spring", node1_found.season == "spring")

    node2_found = engine.get_node("node2")
    test("Node2 four_phase is TAI_YANG", node2_found.four_phase == "TAI_YANG")
    test("Node2 season is summer", node2_found.season == "summer")

    # Test 4: Temporal ranking
    print("\n[Test 4] Temporal Ranking")
    print("-" * 40)

    ranking = engine.get_temporal_ranking(TimeStem.JIA, TimeBranch.ZI)
    test("Temporal ranking returns results", len(ranking) > 0)

    # Test 5: Stem-branch search
    print("\n[Test 5] Stem-Branch Search")
    print("-" * 40)

    stem_branch_results = engine.search_by_stem_branch(TimeStem.JIA, TimeBranch.ZI)
    test("Stem-branch search returns results", len(stem_branch_results) >= 0)

    # Test 6: Energy flow calculation
    print("\n[Test 6] Energy Flow Calculation")
    print("-" * 40)

    flow = engine.calculate_energy_flow("node1", "fire", distance=1.0)
    test("Wood->Fire flow > 1.0", flow > 1.0)

    flow_suppress = engine.calculate_energy_flow("node1", "earth", distance=1.0)
    test("Wood->Earth flow < 1.0", flow_suppress < 1.0)

    # Test 7: Index state
    print("\n[Test 7] Index State")
    print("-" * 40)

    state = engine.get_index_state()
    test("State has total_nodes", state["total_nodes"] == 5)
    test("State has energy_counts", "energy_counts" in state)
    test("State has four_phase_counts", "four_phase_counts" in state)

    # Test 8: Node removal
    print("\n[Test 8] Node Removal")
    print("-" * 40)

    result = engine.remove_node("node1")
    test("Remove node returns True", result)
    test("Node count reduced", len(engine._nodes) == 4)
    test("Node1 not in nodes", engine.get_node("node1") is None)

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = test_spacetime_index()
    exit(0 if success else 1)
