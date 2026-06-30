"""Entity-bridge multi-hop recall via the algebra CausalDAG.

This module implements the *entity co-occurrence bridge graph* that was
validated on real HotpotQA (Full@2 +0.040 over the embedding baseline). It
builds an undirected graph where two memories are linked if they share a
named entity, then propagates from a seed memory to recall bridged memories
that share a bridge entity but have low direct query similarity — the
hallmark of multi-hop evidence.

It is the production counterpart of the A/B experiment's treatment path,
exposed for ``SuMemoryLitePro.query_multihop``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..algebra.causal_graph import CausalDAG

__all__ = ["EntityBridgeRecaller", "extract_entities"]


# Common sentence-initial / generic capitalized words to filter out so the
# co-occurrence graph is not dominated by uninformative tokens.
_STOP_ENTITIES = {
    "the", "this", "that", "these", "those", "his", "her", "their",
    "she", "him", "was", "were", "has", "had", "been", "from", "into",
    "after", "also", "american", "united", "first", "second", "their",
    "its", "it", "he", "we", "they", "them", "who", "which", "what",
    "when", "where", "while", "during", "before", "since", "than",
}


def extract_entities(text: str) -> set[str]:
    """Extract capitalized n-gram entities (1-3 tokens) from text.

    A lightweight, model-free extractor: any run of capitalized words forms a
    candidate entity. Short stopwords and generic words are filtered. This is
    sufficient to capture the bridge entity connecting two multi-hop evidence
    memories (e.g. a person named in both).
    """
    ents: set[str] = set()
    for m in re.finditer(r"[A-Z][a-zA-Z]+(?:[\s\-][A-Z][a-zA-Z]+){0,3}", text):
        ent = m.group().strip()
        if len(ent) >= 3 and ent.lower() not in _STOP_ENTITIES:
            ents.add(ent)
    return ents


@dataclass
class EntityBridgeRecaller:
    """Builds and queries an entity co-occurrence bridge graph.

    The graph is constructed lazily from a corpus of (id, content) pairs and
    cached. ``recall(seed_ids, top_k)`` propagates from seed memories via the
    algebra ``CausalDAG`` to return bridged memory ids ranked by entity
    specificity (rare-entity co-occurrence with the seeds).
    """

    ids: list[str]
    contents: list[str]

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.contents):
            raise ValueError("ids and contents must align")
        self._ent_sets: list[set[str]] | None = None
        self._dag: CausalDAG | None = None
        self._spec: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Graph construction (lazy)
    # ------------------------------------------------------------------
    def _ensure_built(self) -> None:
        if self._dag is not None:
            return
        n = len(self.ids)
        idx = {mid: i for i, mid in enumerate(self.ids)}
        ent_sets = [extract_entities(c) for c in self.contents]

        # document frequency per entity (for idf specificity)
        df: dict[str, int] = {}
        for ents in ent_sets:
            for e in ents:
                df[e] = df.get(e, 0) + 1

        dag = CausalDAG()
        for i in range(n):
            dag.add_node(i)
        # undirected co-occurrence: link paragraphs sharing >=1 entity
        # (build via inverted index for efficiency on large corpora)
        inv: dict[str, list[int]] = {}
        for i, ents in enumerate(ent_sets):
            for e in ents:
                inv.setdefault(e, []).append(i)
        for e, docs in inv.items():
            for a in range(len(docs)):
                for b in range(a + 1, len(docs)):
                    dag.add_edge(docs[a], docs[b], weight=1.0)
                    dag.add_edge(docs[b], docs[a], weight=1.0)

        self._ent_sets = ent_sets
        self._dag = dag
        self._df = df
        self._idx = idx

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------
    def recall(
        self,
        seed_ids: list[str],
        top_k: int = 5,
        exclude: set[str] | None = None,
    ) -> list[str]:
        """Propagate from seeds and return bridged memory ids.

        Bridged memories are those reachable from any seed via shared entities,
        ranked by aggregate entity specificity (sum of idf of shared entities
        across all seeds). Seeds and ``exclude`` ids are not returned.

        Parameters
        ----------
        seed_ids : list
            Memory ids to propagate from (typically the top vector hits).
        top_k : int
            Number of bridged memories to return.
        exclude : set, optional
            Additional ids to exclude from results.

        Returns
        -------
        list[str]
            Bridged memory ids, most-specific first.
        """
        self._ensure_built()
        assert self._dag is not None and self._ent_sets is not None
        exclude = exclude or set()
        seed_set = set(seed_ids) | exclude

        # aggregate received effect + specificity across all seeds
        scores: dict[int, float] = {}
        for sid in seed_ids:
            i = self._idx.get(sid)
            if i is None:
                continue
            eff = self._dag.propagate(i, delta=1.0)
            seed_ents = self._ent_sets[i]
            n = len(self.ids)
            for j, _ in eff.items():
                if j == i or self.ids[j] in seed_set:
                    continue
                shared = seed_ents & self._ent_sets[j]
                if not shared:
                    continue
                spec = sum(math_log_idf(self._df[e], n) for e in shared)
                scores[j] = scores.get(j, 0.0) + spec

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [self.ids[j] for j, _ in ranked]

    def specificity_scores(self, seed_ids: list[str]) -> dict[str, float]:
        """Return {memory_id: specificity} for all bridged memories.

        Useful for fusion with vector scores in the caller.
        """
        self._ensure_built()
        assert self._dag is not None and self._ent_sets is not None
        out: dict[str, float] = {}
        n = len(self.ids)
        for sid in seed_ids:
            i = self._idx.get(sid)
            if i is None:
                continue
            eff = self._dag.propagate(i, delta=1.0)
            seed_ents = self._ent_sets[i]
            for j, _ in eff.items():
                if j == i:
                    continue
                shared = seed_ents & self._ent_sets[j]
                if shared:
                    s = sum(math_log_idf(self._df[e], n) for e in shared)
                    mid = self.ids[j]
                    out[mid] = out.get(mid, 0.0) + s
        return out


def math_log_idf(df: int, n: int) -> float:
    """idf weight: log((n+1)/(df+1)) + 1."""
    import math
    return math.log((n + 1) / (df + 1)) + 1
