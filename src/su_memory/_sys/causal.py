"""
Causal Chain Tracking Module
Target: 95%+ node coverage

Five-Layer Architecture:
- Layer 1: Direct Causality (direct reference / temporal precedence)
- Layer 2: Semantic Causality (creative/receptive/light/abyss/thunder/wind/mountain/lake)
- Layer 3: Energy Flow Causality (enhance/suppress energy flow)
- Layer 4: Temporal Causality (adjacent branch enhance / opposed branch weaken)
- Layer 5: Pattern Transform Causality (inverse/mirror/rotation multi-dimensional reasoning)
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import time


# ============================================================
# Semantic-Energy Constants Table
# ============================================================

CATEGORY_ENERGY_MAP = {
    "creative": "metal", "lake": "metal", "light": "fire", "thunder": "wood",
    "wind": "wood", "abyss": "water", "mountain": "earth", "receptive": "earth",
}

CATEGORY_CAUSALITY = {
    # Category generation sequence (energy transfer direction)
    "creative": {"generates": ["light"], "contradicts": ["wind"]},
    "lake": {"generates": ["abyss"], "contradicts": ["thunder"]},
    "light": {"generates": ["thunder", "wind"], "contradicts": ["creative"]},
    "thunder": {"generates": ["receptive"], "contradicts": ["lake"]},
    "wind": {"generates": ["creative"], "contradicts": ["light"]},
    "abyss": {"generates": ["lake"], "contradicts": ["mountain"]},
    "mountain": {"generates": ["abyss"], "contradicts": ["receptive"]},
    "receptive": {"generates": ["creative"], "contradicts": ["mountain"]},
}

ENERGY_ENHANCE = {
    "wood": "fire", "fire": "earth", "earth": "metal", "metal": "water", "water": "wood",
}

ENERGY_SUPPRESS = {
    "wood": "earth", "earth": "water", "water": "fire", "fire": "metal", "metal": "wood",
}

BRANCH_TEMPORAL = {
    "branch_1": ["branch_12", "branch_2"], "branch_2": ["branch_1", "branch_3"],
    "branch_3": ["branch_2", "branch_4"], "branch_4": ["branch_3", "branch_5"],
    "branch_5": ["branch_4", "branch_6"], "branch_6": ["branch_5", "branch_7"],
    "branch_7": ["branch_6", "branch_8"], "branch_8": ["branch_7", "branch_9"],
    "branch_9": ["branch_8", "branch_10"], "branch_10": ["branch_9", "branch_11"],
    "branch_11": ["branch_10", "branch_12"], "branch_12": ["branch_11", "branch_1"],
}

BRANCH_OPPOSE = {
    "branch_1": "branch_7", "branch_2": "branch_8", "branch_3": "branch_9", "branch_4": "branch_10",
    "branch_5": "branch_11", "branch_6": "branch_12", "branch_7": "branch_1", "branch_8": "branch_2",
    "branch_9": "branch_3", "branch_10": "branch_4", "branch_11": "branch_5", "branch_12": "branch_6",
}

ENERGY_BRANCH = {
    "wood": ["metal", "earth"],
    "fire": ["water", "metal"],
    "earth": ["wood", "water"],
    "metal": ["fire", "wood"],
    "water": ["earth", "fire"],
}


class CausalChain:
    """
    Multi-Layer Causal Chain Tracker
    Target: 95%+ node coverage
    """

    def __init__(self):
        # Layer 1: Direct causal graph
        self.graph: Dict[str, List[str]] = defaultdict(list)
        self.reverse_graph: Dict[str, List[str]] = defaultdict(list)

        # Node energy
        self.energy: Dict[str, float] = {}

        # Layer 2: Semantic attributes
        self.category_map: Dict[str, str] = {}

        # Layer 3: Energy attributes
        self.energy_map: Dict[str, str] = {}

        # Layer 4: Temporal associations
        self.time_map: Dict[str, str] = {}
        self.temporal_links: Dict[str, List[str]] = defaultdict(list)

        # Layer 5: Pattern relationships (inverse/mirror/rotation)
        self.pattern_pairs: Dict[str, Tuple[str, str, str]] = {}

        # Energy propagation history (for balance constraints)
        self.propagation_history: List[Dict] = []

    def add(self, memory_id: str, category: str = None, energy_type: str = None) -> None:
        """Add memory node with semantic-energy attributes"""
        if memory_id not in self.energy:
            self.energy[memory_id] = 1.0
        if category:
            self.category_map[memory_id] = category
        if energy_type:
            self.energy_map[memory_id] = energy_type

    def link(self, parent: str, child: str) -> bool:
        """Layer 1: Create direct causal association"""
        if parent not in self.energy or child not in self.energy:
            return False
        if child not in self.graph[parent]:
            self.graph[parent].append(child)
            self.reverse_graph[child].append(parent)
        return True

    def link_with_category(self, parent: str, child: str,
                       parent_category: str = None, child_category: str = None) -> bool:
        """Layer 2: Create causal association based on semantic category"""
        pc = parent_category or self.category_map.get(parent)
        cc = child_category or self.category_map.get(child)
        if not pc or not cc:
            return self.link(parent, child)

        causality = CATEGORY_CAUSALITY.get(pc, {})

        if cc in causality.get("generates", []):
            # Enhance -> strong link, energy +0.15
            self.energy[parent] = self.energy.get(parent, 1.0) + 0.15
            result = self.link(parent, child)
            if result:
                self.pattern_pairs[(parent, child)] = (pc, cc, "enhance")
            return result

        if cc in causality.get("contradicts", []):
            # Suppress -> weak link, no active propagation
            self.pattern_pairs[(parent, child)] = (pc, cc, "suppress")
            return False

        # Same type or unrelated -> medium link
        if pc == cc:
            self.energy[parent] = self.energy.get(parent, 1.0) + 0.05
        return self.link(parent, child)

    def link_with_energy(self, parent: str, child: str,
                        parent_energy: str = None, child_energy: str = None) -> bool:
        """Layer 3: Create causal association based on energy flow"""
        pe = parent_energy or self.energy_map.get(parent)
        ce = child_energy or self.energy_map.get(child)
        if not pe or not ce:
            return self.link(parent, child)

        if ENERGY_ENHANCE.get(pe) == ce:
            # Parent energy flows to child -> energy +0.1
            self.energy[parent] = self.energy.get(parent, 1.0) + 0.1
            return self.link(parent, child)

        if ENERGY_SUPPRESS.get(pe) == ce:
            # Suppression -> weak link, energy -0.05
            self.energy[parent] = max(0.1, self.energy.get(parent, 1.0) - 0.05)
            self.pattern_pairs[(parent, child)] = (pe, ce, "suppress")
            return False

        return self.link(parent, child)

    def link_temporal(self, memory_id: str, time_branch: str) -> None:
        """Layer 4: Associate memory to temporal branch"""
        if memory_id not in self.energy:
            self.add(memory_id)

        self.time_map[memory_id] = time_branch

        for neighbor in BRANCH_TEMPORAL.get(time_branch, []):
            if neighbor != time_branch:
                self.temporal_links[memory_id].append(neighbor)

    def link_with_timecode(self, parent: str, child: str,
                         parent_tb: str = None, child_tb: str = None) -> bool:
        """Layer 4: Create temporal causal association based on time code"""
        ptb = parent_tb or self.time_map.get(parent)
        ctb = child_tb or self.time_map.get(child)
        if not ptb or not ctb:
            return self.link(parent, child)

        # Adjacent branches -> strong association
        if ptb == ctb or ctb in BRANCH_TEMPORAL.get(ptb, []):
            return self.link(parent, child)

        # Opposed branches -> weak association
        if BRANCH_OPPOSE.get(ptb) == ctb:
            self.energy[parent] = max(0.1, self.energy.get(parent, 1.0) - 0.05)
            return False

        return self.link(parent, child)

    def propagate(self, source: str, delta: float = 0.1) -> Dict[str, float]:
        """Energy propagation: propagate energy along causal chain with energy balance"""
        result: Dict[str, float] = {}
        queue: List[str] = [source]
        visited: set = {source}
        energy_counts: Dict[str, float] = defaultdict(float)

        while queue:
            current = queue.pop(0)
            current_energy_type = self.energy_map.get(current)
            self.energy.get(current, 1.0)

            for nxt in self.graph.get(current, []):
                if nxt not in visited:
                    visited.add(nxt)

                    next_energy_type = self.energy_map.get(nxt)

                    if current_energy_type and next_energy_type:
                        if ENERGY_ENHANCE.get(current_energy_type) == next_energy_type:
                            propagated_energy = delta * 1.1
                        elif ENERGY_SUPPRESS.get(current_energy_type) == next_energy_type:
                            propagated_energy = delta * 0.3
                        else:
                            propagated_energy = delta
                    else:
                        propagated_energy = delta

                    self.energy[nxt] = self.energy.get(nxt, 1.0) + propagated_energy
                    result[nxt] = round(self.energy[nxt], 3)

                    if next_energy_type:
                        energy_counts[next_energy_type] += propagated_energy

                    queue.append(nxt)

        # Record propagation history
        self.propagation_history.append({
            "source": source,
            "delta": delta,
            "affected": list(result.keys()),
            "energy_dist": dict(energy_counts),
        })

        # Apply energy balance constraint
        self._apply_energy_balance(energy_counts)

        return result

    def _apply_energy_balance(self, energy_counts: Dict[str, float]) -> List[str]:
        """Energy balance constraint: triggered when a certain energy type is too strong"""
        if not energy_counts:
            return []

        max_energy_type = max(energy_counts, key=energy_counts.get)
        max_strength = energy_counts[max_energy_type]
        total = sum(energy_counts.values())

        # If a certain energy type exceeds 60% ratio, treat as "too strong"
        if max_strength / max(total, 1) > 0.6:
            constrained = []
            for et in ENERGY_BRANCH.get(max_energy_type, []):
                for mem_id, mem_et in list(self.energy_map.items()):
                    if mem_et == et:
                        self.energy[mem_id] *= 0.9
                        constrained.append(mem_id)
            return constrained
        return []

    def coverage(self, all_ids: List[str]) -> float:
        """
        Multi-layer causal coverage rate

        Node "covered" conditions (any one):
        1. Has direct parent-child relationship
        2. Has semantic association (enhance/same type)
        3. Has energy association
        4. Has temporal association
        5. Participates in pattern relationships (inverse/mirror/rotation)
        """
        if not all_ids:
            return 0.0

        covered = set()

        for mid in all_ids:
            # Layer 1: Direct association
            if self.graph.get(mid) or mid in self.reverse_graph:
                covered.add(mid)
                continue

            # Layer 2: Semantic association
            mid_category = self.category_map.get(mid)
            if mid_category:
                for other_id, other_category in self.category_map.items():
                    if other_id != mid and other_category:
                        causality = CATEGORY_CAUSALITY.get(mid_category, {})
                        if other_category in causality.get("generates", []):
                            covered.add(mid)
                            break

            # Layer 3: Energy association
            mid_energy_type = self.energy_map.get(mid)
            if mid_energy_type:
                for other_id, other_energy_type in self.energy_map.items():
                    if other_id != mid and other_energy_type:
                        if ENERGY_ENHANCE.get(mid_energy_type) == other_energy_type:
                            covered.add(mid)
                            break

            # Layer 4: Temporal association
            mid_tb = self.time_map.get(mid)
            if mid_tb:
                neighbors = BRANCH_TEMPORAL.get(mid_tb, [])
                if any(self.time_map.get(oid) == nb for oid in all_ids for nb in neighbors if oid != mid):
                    covered.add(mid)

            # Layer 5: Pattern relationships
            if (mid,) in self.pattern_pairs or any(mid in pair for pairs in self.pattern_pairs.values() for pair in pairs[:2]):
                covered.add(mid)

        return round(len(covered) / len(all_ids) * 100, 1)

    def detect_conflicts(self, beliefs: List[Dict]) -> List[Dict]:
        """Detect belief conflicts based on energy suppression and semantic contradiction"""
        conflicts: List[Dict] = []

        for i in range(len(beliefs)):
            for j in range(i + 1, len(beliefs)):
                a = beliefs[i]
                b = beliefs[j]

                a_id = a.get("id", f"belief_{i}")
                b_id = b.get("id", f"belief_{j}")
                a_content = a.get("content", "")
                b_content = b.get("content", "")
                a_energy_type = a.get("energy_type") or self.energy_map.get(a_id)
                b_energy_type = b.get("energy_type") or self.energy_map.get(b_id)
                a_category = a.get("category") or self.category_map.get(a_id)
                b_category = b.get("category") or self.category_map.get(b_id)

                severity = 0.5
                conflict_type = "textual"

                # Energy suppression -> high severity
                if a_energy_type and b_energy_type and ENERGY_SUPPRESS.get(a_energy_type) == b_energy_type:
                    severity = 0.9
                    conflict_type = "energy_suppress"

                # Semantic contradiction -> medium severity
                elif a_category and b_category:
                    a_contradicts = CATEGORY_CAUSALITY.get(a_category, {}).get("contradicts", [])
                    if b_category in a_contradicts:
                        severity = 0.7
                        conflict_type = "semantic_suppress"

                # Text contradiction detection (fallback)
                elif self._contradicts(a_content, b_content):
                    severity = 0.6
                    conflict_type = "textual"

                if severity > 0.5:
                    conflicts.append({
                        "memory_a": a_id,
                        "memory_b": b_id,
                        "severity": severity,
                        "type": conflict_type,
                    })

        return sorted(conflicts, key=lambda x: -x["severity"])

    def get_causal_path(self, source: str, target: str) -> List[str]:
        """BFS find causal chain path"""
        if source == target:
            return [source]
        if source not in self.energy or target not in self.energy:
            return []

        queue: List[Tuple[str, List[str]]] = [(source, [source])]
        visited: set = {source}

        while queue:
            current, path = queue.pop(0)

            for nxt in self.graph.get(current, []):
                if nxt == target:
                    return path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))

        return []

    def apply_energy_balance(self) -> List[str]:
        """Actively trigger energy balance constraint (for external call)"""
        if not self.propagation_history:
            return []

        recent = self.propagation_history[-1]
        energy_dist = recent.get("energy_dist", {})

        if energy_dist:
            return self._apply_energy_balance(energy_dist)
        return []

    def get_aging(self, memories: List[Dict]) -> List[Dict]:
        """Knowledge aging detection"""
        aging = []
        now = time.time()

        for m in memories:
            days = (now - m.get("timestamp", now)) / 86400
            if days > 14:
                aging.append({
                    "memory_id": m.get("id"),
                    "days": round(days),
                    "severity": "warning" if days < 30 else "critical",
                })

        return aging

    @staticmethod
    def _contradicts(text_a: str, text_b: str) -> bool:
        """Text contradiction detection (fallback)"""
        pos = ["yes", "have", "correct", "know", "should", "can"]
        neg = ["no", "none", "wrong", "unknown", "should not", "cannot"]

        a_pos = sum(1 for p in pos if p in text_a)
        b_pos = sum(1 for p in pos if p in text_b)
        a_neg = sum(1 for n in neg if n in text_a)
        b_neg = sum(1 for n in neg if n in text_b)

        return (a_pos > 0 and b_neg > 0) or (a_neg > 0 and b_pos > 0)


# ============================================================
# Retrieval-Level Causal Inference Engine
# ============================================================

class CausalInference:
    """
    Retrieval-Level Causal Inference Engine

    No need to pre-build graph, directly compute causal relevance based on
    query and candidate semantic-energy attributes.
    Used for causal dimension in fusion.py and multi-hop retrieval.
    """

    def __init__(self):
        self._energy_enhance_reverse = {v: k for k, v in ENERGY_ENHANCE.items()}

    def infer_relation(self, query_category: str, query_energy: str,
                       cand_category: str, cand_energy: str) -> Dict:
        if query_category == cand_category:
            return {"relation": "same", "score": 1.0, "path": ["same_category"],
                    "explanation": f"{query_category} and {cand_category} are same category"}
        causality = CATEGORY_CAUSALITY.get(query_category, {})
        if cand_category in causality.get("generates", []):
            return {"relation": "generates", "score": 0.8, "path": ["category_enhance"],
                    "explanation": f"{query_category} generates {cand_category} (semantic)"}
        if cand_category in causality.get("contradicts", []):
            if query_energy and cand_energy and ENERGY_ENHANCE.get(query_energy) == cand_energy:
                return {"relation": "generates", "score": 0.7, "path": ["semantic_suppress", "energy_enhance"],
                        "explanation": f"{query_category} suppresses {cand_category} (semantic), but {query_energy} generates {cand_energy} (energy)"}
            return {"relation": "contradicts", "score": 0.3, "path": ["semantic_suppress"],
                    "explanation": f"{query_category} suppresses {cand_category} (semantic)"}
        if query_energy and cand_energy:
            if ENERGY_ENHANCE.get(query_energy) == cand_energy:
                return {"relation": "generates", "score": 0.7, "path": ["energy_enhance"],
                        "explanation": f"{query_energy} generates {cand_energy} (energy)"}
            if ENERGY_ENHANCE.get(cand_energy) == query_energy:
                return {"relation": "generates", "score": 0.6, "path": ["energy_reverse"],
                        "explanation": f"{cand_energy} generates {query_energy} (reverse)"}
            if ENERGY_SUPPRESS.get(query_energy) == cand_energy:
                return {"relation": "contradicts", "score": 0.2, "path": ["energy_suppress"],
                        "explanation": f"{query_energy} suppresses {cand_energy} (energy)"}
            if ENERGY_SUPPRESS.get(cand_energy) == query_energy:
                return {"relation": "contradicts", "score": 0.2, "path": ["energy_suppress_reverse"],
                        "explanation": f"{cand_energy} suppresses {query_energy} (reverse)"}
        return {"relation": "neutral", "score": 0.0, "path": [],
                "explanation": f"{query_category}{query_energy} and {cand_category}{cand_energy} have no direct causal relation"}

    def multi_hop_inference(self, query_category: str, query_energy: str,
                           memories: List[Dict], max_hops: int = 3) -> List[Dict]:
        hop_decay = 0.7
        mem_attrs = []
        for m in memories:
            cat = m.get("category_name") or m.get("payload", {}).get("category_name", "")
            eng = m.get("energy_type") or m.get("payload", {}).get("energy_type", "")
            if not eng and cat:
                eng = CATEGORY_ENERGY_MAP.get(cat, "")
            mem_attrs.append({"category": cat, "energy": eng})
        best_scores = {}
        first_hop_results = []
        for i, (m, attr) in enumerate(zip(memories, mem_attrs)):
            if not attr["category"]:
                continue
            rel = self.infer_relation(query_category, query_energy, attr["category"], attr["energy"])
            score = rel["score"]
            if score > 0:
                first_hop_results.append((i, score, rel))
                best_scores[i] = {"hop_score": score, "hop_count": 1,
                    "hop_path": [f"query->{m.get('id', i)}({rel['relation']})"]}
        if max_hops < 2:
            return self._build_results(memories, best_scores)
        first_hop_results.sort(key=lambda x: x[1], reverse=True)
        bridges = first_hop_results[:5]
        for bridge_idx, bridge_score, bridge_rel in bridges:
            bridge_attr = mem_attrs[bridge_idx]
            for j, attr in enumerate(mem_attrs):
                if j == bridge_idx or not attr["category"]:
                    continue
                rel2 = self.infer_relation(bridge_attr["category"], bridge_attr["energy"],
                                           attr["category"], attr["energy"])
                if rel2["score"] > 0:
                    hop2_score = bridge_score * hop_decay * rel2["score"]
                    if j not in best_scores or hop2_score > best_scores[j]["hop_score"]:
                        bridge_id = memories[bridge_idx].get("id", bridge_idx)
                        target_id = memories[j].get("id", j)
                        best_scores[j] = {"hop_score": hop2_score, "hop_count": 2,
                            "hop_path": [f"query->{bridge_id}({bridge_rel['relation']})",
                                         f"{bridge_id}->{target_id}({rel2['relation']})"]}
        if max_hops < 3:
            return self._build_results(memories, best_scores)
        hop2_bridges = [(idx, info) for idx, info in best_scores.items() if info["hop_count"] == 2]
        hop2_bridges.sort(key=lambda x: x[1]["hop_score"], reverse=True)
        for bridge_idx, bridge_info in hop2_bridges[:3]:
            bridge_attr = mem_attrs[bridge_idx]
            for j, attr in enumerate(mem_attrs):
                if j == bridge_idx or not attr["category"]:
                    continue
                rel3 = self.infer_relation(bridge_attr["category"], bridge_attr["energy"],
                                           attr["category"], attr["energy"])
                if rel3["score"] > 0:
                    hop3_score = bridge_info["hop_score"] * hop_decay * rel3["score"]
                    if j not in best_scores or hop3_score > best_scores[j]["hop_score"]:
                        bridge_id = memories[bridge_idx].get("id", bridge_idx)
                        target_id = memories[j].get("id", j)
                        best_scores[j] = {"hop_score": hop3_score, "hop_count": 3,
                            "hop_path": bridge_info["hop_path"] + [f"{bridge_id}->{target_id}({rel3['relation']})"]}
        return self._build_results(memories, best_scores)

    def build_reasoning_chain(self, memories: List[Dict]) -> Dict:
        nodes = []
        edges = []
        covered = set()
        mem_attrs = []
        for m in memories:
            cat = m.get("category_name") or m.get("payload", {}).get("category_name", "")
            eng = m.get("energy_type") or m.get("payload", {}).get("energy_type", "")
            if not eng and cat:
                eng = CATEGORY_ENERGY_MAP.get(cat, "")
            mid = m.get("id", f"mem_{len(mem_attrs)}")
            nodes.append({"id": mid, "category": cat, "energy": eng})
            mem_attrs.append({"id": mid, "category": cat, "energy": eng})
        for i in range(len(mem_attrs)):
            for j in range(len(mem_attrs)):
                if i == j:
                    continue
                a, b = mem_attrs[i], mem_attrs[j]
                if not a["category"] or not b["category"]:
                    continue
                rel = self.infer_relation(a["category"], a["energy"], b["category"], b["energy"])
                if rel["score"] > 0 and rel["relation"] != "neutral":
                    edges.append({"from": a["id"], "to": b["id"],
                                  "relation": rel["relation"], "score": rel["score"]})
                    covered.add(i)
                    covered.add(j)
        adj = defaultdict(list)
        for e in edges:
            if e["relation"] in ("generates", "same"):
                adj[e["from"]].append((e["to"], e["score"]))
        longest_chain = []
        for start_node in nodes:
            chain = self._dfs_longest(adj, start_node["id"], set())
            if len(chain) > len(longest_chain):
                longest_chain = chain
        coverage = len(covered) / max(len(memories), 1) * 100
        return {"nodes": nodes, "edges": edges, "chains": longest_chain, "coverage": round(coverage, 1)}

    def _build_results(self, memories, best_scores):
        results = []
        for i, m in enumerate(memories):
            entry = dict(m)
            if i in best_scores:
                entry["hop_score"] = round(best_scores[i]["hop_score"], 4)
                entry["hop_count"] = best_scores[i]["hop_count"]
                entry["hop_path"] = best_scores[i]["hop_path"]
            else:
                entry["hop_score"] = 0.0
                entry["hop_count"] = 0
                entry["hop_path"] = []
            results.append(entry)
        results.sort(key=lambda x: x["hop_score"], reverse=True)
        return results

    @staticmethod
    def _dfs_longest(adj, node, visited):
        visited.add(node)
        best = [node]
        for nxt, _ in adj.get(node, []):
            if nxt not in visited:
                chain = [node] + CausalInference._dfs_longest(adj, nxt, visited.copy())
                if len(chain) > len(best):
                    best = chain
        return best
