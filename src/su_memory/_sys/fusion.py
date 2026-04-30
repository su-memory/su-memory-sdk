"""
Multi-dimensional Fusion Retrieval Interface

Exposed: MultiViewRetriever
Internal: Implemented in su_core._sys

5-Dimensional Fusion:
1. semantic (0.40): Semantic vector similarity
2. category_soft (0.15): Category soft matching (probability distribution similarity)
3. energy_match (0.15): Energy type matching (enhance/suppress relationships)
4. holographic (0.15): Holographic structure matching (primary/mutual/reverse/complement)
5. causal (0.15): Causality chain correlation
"""

from typing import List, Dict, Any
from .encoders import EncoderCore, EncodingInfo, _cosine_similarity_dict
from ._c2 import CategoryType, ENERGY_ENHANCE, ENERGY_SUPPRESS
from .causal import CATEGORY_CAUSALITY


class MultiViewRetriever:
    """
    Multi-dimensional Fusion Retriever

    Provides 5-dimensional fusion retrieval capability:
    1. Semantic vector similarity (core dimension)
    2. Category soft matching (probability distribution similarity)
    3. Energy type matching (enhance/suppress relationships)
    4. Holographic structure matching (primary/mutual/reverse/complement)
    5. Causality chain correlation

    External hiding: specific algorithms, weight configuration
    """

    def __init__(self):
        self.pattern_system = EncoderCore()
        # 5-dimensional fusion weight configuration
        self._weights = {
            "semantic": 0.40,
            "category_soft": 0.15,
            "energy_match": 0.15,
            "holographic": 0.15,
            "causal": 0.15,
        }

    def retrieve(
        self,
        query_content: str,
        query_pattern: EncodingInfo,
        candidates: List[Dict[str, Any]],
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Multi-dimensional retrieval main entry point

        Args:
            query_content: Query text
            query_pattern: Pattern encoding info (with category_probs/energy_scores)
            candidates: Candidate memory list
            top_k: Number of results to return

        Returns:
            Sorted memory list with fusion scores and fusion details
        """
        if not candidates:
            return []

        # Extract query category name (for causal dimension)
        query_category = self._get_category_name(query_pattern)

        for i, cand in enumerate(candidates):
            pattern_idx = cand.get("hexagram_index", 0)

            # ---- Dimension 1: semantic (0.40) ----
            semantic_score = cand.get("vector_score", 0.5)

            # ---- Dimension 2: category_soft (0.15) ----
            category_score = self._compute_category_match(query_pattern, cand, pattern_idx)

            # ---- Dimension 3: energy_match (0.15) ----
            energy_score = self._compute_energy_match(query_pattern, cand, pattern_idx)

            # ---- Dimension 4: holographic (0.15) ----
            holo_score = self._compute_holographic(query_pattern, cand, pattern_idx)

            # ---- Dimension 5: causal (0.15) ----
            causal_score = self._compute_causal(query_category, cand)

            # Combined score = 5-dimensional weighted sum
            total_score = (
                self._weights["semantic"] * semantic_score +
                self._weights["category_soft"] * category_score +
                self._weights["energy_match"] * energy_score +
                self._weights["holographic"] * holo_score +
                self._weights["causal"] * causal_score
            )

            cand["holographic_score"] = round(total_score, 4)
            cand["fusion_detail"] = {
                "semantic": round(semantic_score, 4),
                "category_soft": round(category_score, 4),
                "energy_match": round(energy_score, 4),
                "holographic": round(holo_score, 4),
                "causal": round(causal_score, 4),
            }

        # Sort by combined score
        candidates.sort(key=lambda x: x.get("holographic_score", 0), reverse=True)

        return candidates[:top_k]

    def _get_category_name(self, info: EncodingInfo) -> str:
        """Extract dominant category name from EncodingInfo"""
        if info.category_probs:
            return max(info.category_probs, key=info.category_probs.get)
        # fallback: infer category from pattern name
        from .encoders import CATEGORY_NAMES, HEXAGRAM_TRIGRAMS_BELOW
        below_idx = HEXAGRAM_TRIGRAMS_BELOW[info.index]
        if 0 <= below_idx < len(CATEGORY_NAMES):
            return CATEGORY_NAMES[below_idx]
        return "lake"

    def _compute_category_match(self, query_info: EncodingInfo, cand: Dict, pattern_idx: int) -> float:
        """Compute category soft matching score"""
        q_probs = query_info.category_probs
        c_probs = cand.get("category_probs")

        if q_probs and c_probs:
            return max(0.0, _cosine_similarity_dict(q_probs, c_probs))

        # fallback: 0/1 primary pattern matching
        return 1.0 if pattern_idx == query_info.index else 0.0

    def _compute_energy_match(self, query_info: EncodingInfo, cand: Dict, pattern_idx: int) -> float:
        """Compute energy type matching score"""
        q_scores = query_info.energy_scores
        c_scores = cand.get("energy_scores")

        base_score = 0.0
        if q_scores and c_scores:
            base_score = max(0.0, _cosine_similarity_dict(q_scores, c_scores))
        else:
            # fallback: same energy type -> 1.0
            cand_info = EncodingInfo.from_index(pattern_idx)
            base_score = 1.0 if cand_info.energy == query_info.energy else 0.0

        # Enhance/suppress bonus
        q_energy = self._dominant_energy(query_info)
        c_energy = self._dominant_energy_from_cand(cand, pattern_idx)

        if q_energy and c_energy:
            if ENERGY_ENHANCE.get(q_energy) == c_energy or ENERGY_ENHANCE.get(c_energy) == q_energy:
                base_score = min(1.0, base_score + 0.2)
            elif ENERGY_SUPPRESS.get(q_energy) == c_energy or ENERGY_SUPPRESS.get(c_energy) == q_energy:
                base_score = max(0.0, base_score - 0.1)

        return base_score

    def _dominant_energy(self, info: EncodingInfo):
        """Get dominant energy type from EncodingInfo"""
        if info.energy_scores:
            dominant_name = max(info.energy_scores, key=info.energy_scores.get)
            return self._name_to_energy(dominant_name)
        return self._name_to_energy(info.energy)

    def _dominant_energy_from_cand(self, cand: Dict, pattern_idx: int):
        """Get dominant energy type from candidate"""
        c_scores = cand.get("energy_scores")
        if c_scores:
            dominant_name = max(c_scores, key=c_scores.get)
            return self._name_to_energy(dominant_name)
        payload = cand.get("payload", {})
        energy_name = payload.get("energy") or EncodingInfo.from_index(pattern_idx).energy
        return self._name_to_energy(energy_name)

    @staticmethod
    def _name_to_energy(name: str):
        """Energy name -> CategoryType enum"""
        mapping = {
            "metal": CategoryType.METAL, "wood": CategoryType.WOOD,
            "water": CategoryType.WATER, "fire": CategoryType.FIRE, "earth": CategoryType.EARTH,
            # Backward compatibility
            "metal": CategoryType.METAL, "wood": CategoryType.WOOD,
            "water": CategoryType.WATER, "fire": CategoryType.FIRE, "earth": CategoryType.EARTH
        }
        return mapping.get(name)

    def _compute_holographic(self, query_info: EncodingInfo, cand: Dict, pattern_idx: int) -> float:
        """Compute holographic structure matching score"""
        # Build candidate EncodingInfo (if category_probs exists)
        candidate_infos = {}
        c_probs = cand.get("category_probs")
        c_escores = cand.get("energy_scores")
        if c_probs or c_escores:
            ci = EncodingInfo.from_index(pattern_idx)
            ci.category_probs = c_probs
            ci.energy_scores = c_escores
            candidate_infos[pattern_idx] = ci

        scores = self.pattern_system.retrieve_holographic(
            query_index=query_info.index,
            candidate_indices=[pattern_idx],
            top_k=1,
            query_info=query_info,
            candidate_infos=candidate_infos if candidate_infos else None,
        )

        if scores:
            return scores[0][1]
        return 0.0

    def _compute_causal(self, query_category: str, cand: Dict) -> float:
        """Compute causality chain correlation"""
        payload = cand.get("payload", {})
        cand_pattern_name = payload.get("hexagram_name", "")

        # Infer category from pattern name (take lower trigram)
        cand_category = self._pattern_to_category(cand_pattern_name, cand)

        if not query_category or not cand_category:
            return 0.0

        # Same category -> 1.0
        if query_category == cand_category:
            return 1.0

        # Check CATEGORY_CAUSALITY
        causality = CATEGORY_CAUSALITY.get(query_category, {})
        if cand_category in causality.get("generates", []):
            return 0.8
        if cand_category in causality.get("contradicts", []):
            return 0.3

        return 0.0

    def _pattern_to_category(self, pattern_name: str, cand: Dict) -> str:
        """Infer category from 64-pattern name (lower trigram)"""
        # First try to get argmax from candidate's category_probs
        c_probs = cand.get("category_probs")
        if c_probs:
            return max(c_probs, key=c_probs.get)

        # Try to get energy from payload then map to category
        payload = cand.get("payload", {})
        energy_name = payload.get("energy", "")
        energy_to_default_category = {
            "metal": "creative", "wood": "thunder", "water": "abyss",
            "fire": "light", "earth": "receptive",
            # Chinese aliases for backward compatibility
            "金": "creative", "木": "thunder", "水": "abyss", "火": "light", "土": "receptive",
        }
        if energy_name in energy_to_default_category:
            return energy_to_default_category[energy_name]

        # fallback: from pattern_index
        pattern_idx = cand.get("hexagram_index", 0)
        from .encoders import CATEGORY_NAMES, HEXAGRAM_TRIGRAMS_BELOW
        below_idx = HEXAGRAM_TRIGRAMS_BELOW[pattern_idx]
        if 0 <= below_idx < len(CATEGORY_NAMES):
            return CATEGORY_NAMES[below_idx]
        return "lake"
