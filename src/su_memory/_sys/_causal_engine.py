"""
WuXing Causal Engine - Integration of Five Elements Relations with Causal Inference

This module integrates the Five Elements (Wu Xing) enhance/suppress relationships
into the causal inference engine for memory node association strength adjustment.

Key Features:
- Link weight adjustment based on energy relations
- Cross-layer integration (semantic, temporal, energy)
- Memory retrieval with energy-aware scoring
- Balance constraint enforcement
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import time

from ._energy_relations import (
    RelationType,
    ENERGY_ENHANCE,
    ENERGY_SUPPRESS,
    analyze_relation,
    calculate_link_weight,
    get_affinity_score,
    analyze_balance,
    EnergyRelation,
)


@dataclass
class EnergyMemoryNode:
    """Memory node with energy attributes for causal inference"""
    node_id: str
    content: str
    energy_type: str  # Five elements: wood, fire, earth, metal, water
    category: Optional[str] = None  # Semantic category (optional)
    time_stem: Optional[int] = None  # Heavenly stem index (optional)
    time_branch: Optional[int] = None  # Earthly branch index (optional)
    intensity: float = 1.0
    timestamp: float = field(default_factory=time.time)
    neighbors: Dict[str, float] = field(default_factory=dict)


class WuXingCausalEngine:
    """
    Causal Engine with Five Elements Energy Relations

    This engine extends traditional causal inference with:
    1. Energy-based link weight calculation
    2. Enhance (相生) relationships boost association strength
    3. Suppress (相克) relationships reduce association strength
    4. Cross-layer integration with semantic and temporal dimensions
    """

    def __init__(self):
        # Node storage
        self.nodes: Dict[str, EnergyMemoryNode] = {}

        # Causal graph: parent -> [children]
        self.graph: Dict[str, List[str]] = defaultdict(list)
        self.reverse_graph: Dict[str, List[str]] = defaultdict(list)

        # Energy propagation history
        self.propagation_history: List[Dict] = []

        # Node energy cache
        self._energy_cache: Dict[str, str] = {}

        # Cross-layer mappings
        self.category_energy_map: Dict[str, str] = {
            "creative": "metal",
            "lake": "metal",
            "light": "fire",
            "thunder": "wood",
            "wind": "wood",
            "abyss": "water",
            "mountain": "earth",
            "receptive": "earth",
        }

        # Temporal energy mapping (Earthly branches)
        self.branch_energy_map: Dict[str, str] = {
            "branch_1": "water", "branch_2": "earth", "branch_3": "wood", "branch_4": "wood",
            "branch_5": "earth", "branch_6": "fire", "branch_7": "fire", "branch_8": "earth",
            "branch_9": "metal", "branch_10": "metal", "branch_11": "earth", "branch_12": "water"
        }

    def add_node(
        self,
        node_id: str,
        content: str,
        energy_type: str = None,
        category: str = None,
        time_stem: int = None,
        time_branch: int = None,
        intensity: float = 1.0
    ) -> bool:
        """
        Add a memory node with energy attributes.

        Args:
            node_id: Unique node identifier
            content: Node content
            energy_type: Five elements type (wood/fire/earth/metal/water)
            category: Semantic category (optional)
            time_stem: Heavenly stem index (0-9, optional)
            time_branch: Earthly branch index (0-11, optional)
            intensity: Node intensity (default: 1.0)

        Returns:
            True if added successfully
        """
        if node_id in self.nodes:
            return False

        # Infer energy type if not provided
        if energy_type is None:
            if category:
                energy_type = self.category_energy_map.get(category, "earth")
            else:
                energy_type = "earth"

        node = EnergyMemoryNode(
            node_id=node_id,
            content=content,
            energy_type=energy_type,
            category=category,
            time_stem=time_stem,
            time_branch=time_branch,
            intensity=intensity
        )

        self.nodes[node_id] = node
        self._energy_cache[node_id] = energy_type

        return True

    def _get_node_energy(self, node_id: str) -> str:
        """Get node's energy type with caching"""
        if node_id in self._energy_cache:
            return self._energy_cache[node_id]
        if node_id in self.nodes:
            return self.nodes[node_id].energy_type
        return "earth"  # Default

    def link(
        self,
        parent_id: str,
        child_id: str,
        base_weight: float = 1.0,
        use_energy: bool = True
    ) -> Tuple[bool, float]:
        """
        Create a causal link between two nodes.

        The link weight is adjusted based on Five Elements relationships:
        - Enhance (相生): weight * 1.2 (增强)
        - Suppress (相克): weight * 0.8 (削弱)
        - Same type: weight * 1.1 (同类增强)
        - Neutral: weight * 1.0 (不变)

        Args:
            parent_id: Parent node ID
            child_id: Child node ID
            base_weight: Base link weight (default: 1.0)
            use_energy: Whether to apply energy relation adjustment

        Returns:
            Tuple of (success, actual_weight)
        """
        if parent_id not in self.nodes or child_id not in self.nodes:
            return False, 0.0

        parent_energy = self._get_node_energy(parent_id)
        child_energy = self._get_node_energy(child_id)

        # Calculate link weight based on energy relations
        if use_energy:
            actual_weight = calculate_link_weight(parent_energy, child_energy, base_weight)
        else:
            actual_weight = base_weight

        # Update graph
        if child_id not in self.graph[parent_id]:
            self.graph[parent_id].append(child_id)

        if parent_id not in self.reverse_graph[child_id]:
            self.reverse_graph[child_id].append(parent_id)

        # Update node neighbor weights
        self.nodes[parent_id].neighbors[child_id] = actual_weight

        return True, actual_weight

    def link_with_energy_relation(
        self,
        source_id: str,
        target_id: str,
        direction: str = "forward"
    ) -> Tuple[bool, EnergyRelation]:
        """
        Link two nodes based on their energy relation.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            direction: "forward" (source->target) or "backward" (target->source)

        Returns:
            Tuple of (success, EnergyRelation)
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return False, None

        source_energy = self._get_node_energy(source_id)
        target_energy = self._get_node_energy(target_id)

        # Analyze energy relation
        relation = analyze_relation(source_energy, target_energy)

        # Create link based on direction
        if direction == "forward":
            success, _ = self.link(source_id, target_id, use_energy=True)
        else:
            success, _ = self.link(target_id, source_id, use_energy=True)

        return success, relation

    def propagate(
        self,
        source_id: str,
        delta: float = 0.1,
        use_energy_balance: bool = True
    ) -> Dict[str, float]:
        """
        Propagate energy along causal chains.

        Energy propagation follows Five Elements rules:
        - Enhance relationship: propagate * 1.1
        - Suppress relationship: propagate * 0.3
        - Other: propagate * 1.0

        Args:
            source_id: Source node ID
            delta: Energy delta to propagate
            use_energy_balance: Whether to apply energy balance constraints

        Returns:
            Dict mapping node_id to new intensity
        """
        if source_id not in self.nodes:
            return {}

        result: Dict[str, float] = {}
        queue: List[str] = [source_id]
        visited: Set[str] = {source_id}
        energy_counts: Dict[str, float] = defaultdict(float)

        self._get_node_energy(source_id)

        while queue:
            current = queue.pop(0)
            current_energy = self._get_node_energy(current)
            self.nodes[current]

            for next_id in self.graph.get(current, []):
                if next_id in visited:
                    continue

                visited.add(next_id)
                next_energy = self._get_node_energy(next_id)

                # Calculate propagation factor based on energy relation
                if ENERGY_ENHANCE.get(current_energy) == next_energy:
                    prop_factor = 1.1  # Enhance boost
                elif ENERGY_SUPPRESS.get(current_energy) == next_energy:
                    prop_factor = 0.3  # Suppress reduction
                else:
                    prop_factor = 1.0  # Normal

                propagated = delta * prop_factor
                self.nodes[next_id].intensity += propagated
                result[next_id] = round(self.nodes[next_id].intensity, 3)

                energy_counts[next_energy] += propagated
                queue.append(next_id)

        # Record propagation history
        self.propagation_history.append({
            "source": source_id,
            "delta": delta,
            "affected": list(result.keys()),
            "energy_dist": dict(energy_counts),
        })

        # Apply energy balance constraint
        if use_energy_balance and energy_counts:
            self._apply_energy_balance(energy_counts)

        return result

    def _apply_energy_balance(self, energy_counts: Dict[str, float]) -> List[str]:
        """
        Apply energy balance constraint.

        When a certain energy type exceeds 60% of total,
        constrain nodes of that energy type.

        Args:
            energy_counts: Dict mapping energy type to propagated energy

        Returns:
            List of constrained node IDs
        """
        total = sum(energy_counts.values())
        if total == 0:
            return []

        max_type = max(energy_counts, key=energy_counts.get)
        max_ratio = energy_counts[max_type] / total

        if max_ratio > 0.6:
            constrained = []
            suppressed_type = ENERGY_SUPPRESS.get(max_type)  # What this type suppresses

            for node_id, node in self.nodes.items():
                if node.energy_type == suppressed_type:
                    node.intensity *= 0.9
                    constrained.append(node_id)

            return constrained

        return []

    def get_relation(self, node1_id: str, node2_id: str) -> Optional[EnergyRelation]:
        """
        Get the energy relation between two nodes.

        Args:
            node1_id: First node ID
            node2_id: Second node ID

        Returns:
            EnergyRelation if both nodes exist, None otherwise
        """
        if node1_id not in self.nodes or node2_id not in self.nodes:
            return None

        energy1 = self._get_node_energy(node1_id)
        energy2 = self._get_node_energy(node2_id)

        return analyze_relation(energy1, energy2)

    def get_neighbors_by_relation(
        self,
        node_id: str,
        relation_type: RelationType = None
    ) -> List[Tuple[str, EnergyRelation]]:
        """
        Get neighbors of a node filtered by relation type.

        Args:
            node_id: Node ID
            relation_type: Filter by this relation type (optional)

        Returns:
            List of (neighbor_id, EnergyRelation) tuples
        """
        if node_id not in self.nodes:
            return []

        node_energy = self._get_node_energy(node_id)
        results = []

        for neighbor_id in self.graph.get(node_id, []):
            neighbor_energy = self._get_node_energy(neighbor_id)
            relation = analyze_relation(node_energy, neighbor_energy)

            if relation_type is None or relation.relation == relation_type:
                results.append((neighbor_id, relation))

        return results

    def get_enhancing_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbors that this node enhances (相生)"""
        results = self.get_neighbors_by_relation(node_id, RelationType.ENHANCE)
        return [nid for nid, _ in results]

    def get_suppressing_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbors that this node suppresses (相克)"""
        results = self.get_neighbors_by_relation(node_id, RelationType.SUPPRESS)
        return [nid for nid, _ in results]

    def analyze_memory_graph(self) -> Dict[str, Any]:
        """
        Analyze the entire memory graph from Five Elements perspective.

        Returns:
            Analysis result with energy distribution, balance, etc.
        """
        if not self.nodes:
            return {"status": "empty", "energy_distribution": {}}

        # Energy distribution
        energy_dist: Dict[str, int] = defaultdict(int)
        for node in self.nodes.values():
            energy_dist[node.energy_type] += 1

        # Analyze balance
        balance_result = analyze_balance(energy_dist)

        # Count relations
        enhance_count = 0
        suppress_count = 0
        neutral_count = 0

        for parent_id, children in self.graph.items():
            parent_energy = self._get_node_energy(parent_id)
            for child_id in children:
                child_energy = self._get_node_energy(child_id)
                relation = analyze_relation(parent_energy, child_energy)

                if relation.relation == RelationType.ENHANCE:
                    enhance_count += 1
                elif relation.relation == RelationType.SUPPRESS:
                    suppress_count += 1
                else:
                    neutral_count += 1

        return {
            "status": "ok",
            "node_count": len(self.nodes),
            "edge_count": sum(len(v) for v in self.graph.values()),
            "energy_distribution": dict(energy_dist),
            "balance": balance_result,
            "relation_stats": {
                "enhance": enhance_count,
                "suppress": suppress_count,
                "neutral": neutral_count,
            }
        }

    def query_with_energy_boost(
        self,
        query_node_id: str,
        candidates: List[str],
        base_scores: Dict[str, float] = None
    ) -> List[Dict]:
        """
        Query candidates with energy relation boosting.

        Args:
            query_node_id: Query node ID
            candidates: List of candidate node IDs
            base_scores: Dict of candidate_id -> base similarity score

        Returns:
            List of candidates with boosted scores, sorted by score
        """
        if query_node_id not in self.nodes:
            return []

        query_energy = self._get_node_energy(query_node_id)
        base_scores = base_scores or {}

        results = []
        for cand_id in candidates:
            if cand_id not in self.nodes:
                continue

            cand_energy = self._get_node_energy(cand_id)
            base = base_scores.get(cand_id, 0.5)  # Default 0.5 if not provided

            # Calculate affinity score
            affinity = get_affinity_score(query_energy, cand_energy)

            # Get energy relation details
            relation = analyze_relation(query_energy, cand_energy)

            # Boosted score
            boosted_score = base * affinity

            results.append({
                "node_id": cand_id,
                "content": self.nodes[cand_id].content,
                "energy_type": cand_energy,
                "base_score": base,
                "affinity": affinity,
                "boosted_score": boosted_score,
                "relation": relation.relation.value,
                "relation_desc": relation.description
            })

        # Sort by boosted score
        results.sort(key=lambda x: x["boosted_score"], reverse=True)

        return results


# ============================================================
# Unit Tests
# ============================================================

def test_wuxing_causal_engine():
    """Test WuXing Causal Engine"""
    print("=" * 60)
    print("Testing WuXing Causal Engine")
    print("=" * 60)

    engine = WuXingCausalEngine()

    # Test 1: Add nodes with different energy types
    print("\n[Test 1] Add Nodes")
    engine.add_node("node_wood", "Wood energy node", energy_type="wood")
    engine.add_node("node_fire", "Fire energy node", energy_type="fire")
    engine.add_node("node_earth", "Earth energy node", energy_type="earth")
    engine.add_node("node_metal", "Metal energy node", energy_type="metal")
    engine.add_node("node_water", "Water energy node", energy_type="water")
    print("  Added 5 nodes with different energy types")

    # Test 2: Link with energy relation (enhance)
    print("\n[Test 2] Link with Enhance Relation (wood->fire)")
    success, weight = engine.link("node_wood", "node_fire", base_weight=1.0)
    print(f"  Wood -> Fire: success={success}, weight={weight}")
    assert success, "Link should succeed"
    assert abs(weight - 1.2) < 0.01, f"Weight should be 1.2 (enhance), got {weight}"

    # Test 3: Link with energy relation (suppress)
    print("\n[Test 3] Link with Suppress Relation (wood->earth)")
    success, weight = engine.link("node_wood", "node_earth", base_weight=1.0)
    print(f"  Wood -> Earth: success={success}, weight={weight}")
    assert success, "Link should succeed"
    assert abs(weight - 0.8) < 0.01, f"Weight should be 0.8 (suppress), got {weight}"

    # Test 4: Get relation
    print("\n[Test 4] Get Energy Relation")
    relation = engine.get_relation("node_wood", "node_fire")
    print(f"  Wood <-> Fire: {relation.relation.value}, strength={relation.strength}")
    assert relation.relation == RelationType.ENHANCE

    # Test 5: Propagate with energy balance
    print("\n[Test 5] Energy Propagation")
    engine.nodes["node_wood"].intensity = 2.0
    result = engine.propagate("node_wood", delta=0.1)
    print(f"  Propagated from wood: affected={len(result)} nodes")
    if "node_fire" in result:
        print(f"     Fire intensity: {result['node_fire']} (enhanced)")
    if "node_earth" in result:
        print(f"     Earth intensity: {result['node_earth']} (suppressed)")

    # Test 6: Get neighbors by relation
    print("\n[Test 6] Get Neighbors by Relation Type")
    engine.link("node_fire", "node_earth", base_weight=1.0)  # Fire enhances Earth
    engine.link("node_water", "node_wood", base_weight=1.0)  # Water enhances Wood

    enhancing = engine.get_enhancing_neighbors("node_wood")
    print(f"  Wood's enhancing neighbors: {enhancing}")

    suppressing = engine.get_suppressing_neighbors("node_wood")
    print(f"  Wood's suppressing neighbors: {suppressing}")

    # Test 7: Query with energy boost
    print("\n[Test 7] Query with Energy Boost")
    candidates = ["node_wood", "node_fire", "node_earth", "node_metal", "node_water"]
    base_scores = {c: 0.7 for c in candidates}
    results = engine.query_with_energy_boost("node_wood", candidates, base_scores)

    print("  Query: node_wood (wood energy)")
    for r in results[:3]:
        print(f"    {r['node_id']} ({r['energy_type']}): base={r['base_score']:.2f}, "
              f"affinity={r['affinity']:.2f}, boosted={r['boosted_score']:.2f}")

    # Verify fire is first (enhance)
    assert results[0]["node_id"] == "node_fire", "Fire should be first (enhance)"

    # Test 8: Analyze memory graph
    print("\n[Test 8] Analyze Memory Graph")
    analysis = engine.analyze_memory_graph()
    print(f"  Nodes: {analysis['node_count']}")
    print(f"  Edges: {analysis['edge_count']}")
    print(f"  Energy dist: {analysis['energy_distribution']}")
    print(f"  Relation stats: {analysis['relation_stats']}")

    print("\n" + "=" * 60)
    print("All WuXing Causal Engine tests passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_wuxing_causal_engine()
