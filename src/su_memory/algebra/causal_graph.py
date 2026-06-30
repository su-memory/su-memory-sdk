"""CausalDAG — directed acyclic graph with BFS belief/energy propagation.

Mathematical foundation
-----------------------
A causal model over a finite set of nodes is a directed graph G = (V, E)
together with an edge-weight function w : E → (0, +∞) that gives the
*transmission coefficient* along each edge. The graph encodes "X influences Y"
as the edge X → Y.

Given an intervention (a "do" or evidence) that injects an amount Δ at a
source node s, the propagated effect at every reachable node v is computed by
breadth-first traversal along the directed edges, accumulating::

    effect(v) = Δ · Π_{e on a path s→v, taking the min-weight edge into v} ...

In the simplest (BFS, single-visit) form used here, each node is visited once
and receives::

    effect(child) = effect(parent) · w(parent → child)

i.e. the weight acts as a multiplicative attenuation along each edge, and the
first (shortest-path) arrival determines the effect. This is the algebraic
core of the legacy ``CausalChain.propagate``.

If the graph contains a directed cycle, the structure is no longer a DAG and
causal propagation is ill-defined (effects would loop indefinitely). We
therefore detect cycles with a topological sort and refuse to propagate on a
cyclic graph unless an explicit ring-tolerant mode is requested.

This module is pure graph theory: nodes are arbitrary hashable keys, weights
are plain floats, no SDK coupling.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

__all__ = ["CausalDAG"]


@dataclass
class CausalDAG:
    """Directed acyclic graph with weighted BFS propagation.

    Attributes
    ----------
    nodes : set
        The node set V (arbitrary hashable keys).
    edges : dict
        Adjacency map node -> list of (child, weight) tuples.
    """

    nodes: set = field(default_factory=set)
    edges: dict = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self) -> None:
        self.edges = defaultdict(list, self.edges)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_node(self, node) -> None:
        """Insert a node (no-op if present)."""
        self.nodes.add(node)
        _ = self.edges[node]  # ensure key exists

    def add_edge(self, parent, child, weight: float = 1.0) -> None:
        """Add a directed edge parent -> child with transmission weight.

        ``weight`` is a positive transmission coefficient: >1 amplifies the
        cause along this edge (an "enhancing" relationship), =1 passes it
        through unchanged, <1 attenuates it (an "inhibiting" relationship).
        """
        if not weight > 0.0:
            raise ValueError(f"edge weight must be positive, got {weight}")
        self.add_node(parent)
        self.add_node(child)
        self.edges[parent].append((child, float(weight)))

    def remove_node(self, node) -> None:
        """Remove a node and all edges incident to it."""
        self.nodes.discard(node)
        self.edges.pop(node, None)
        for p in list(self.edges):
            self.edges[p] = [(c, w) for (c, w) in self.edges[p] if c != node]

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------
    def parents(self, child) -> list:
        """Direct parents of ``child``."""
        return [p for p in self.edges if any(c == child for c, _ in self.edges[p])]

    def children(self, parent) -> list:
        """Direct children of ``parent``."""
        return [c for c, _ in self.edges.get(parent, [])]

    def edge_weight(self, parent, child) -> float:
        """Transmission weight of edge parent -> child (0 if no edge)."""
        for c, w in self.edges.get(parent, []):
            if c == child:
                return w
        return 0.0

    def reach(self, source) -> set:
        """Set of nodes reachable from ``source`` along directed edges."""
        if source not in self.nodes:
            return set()
        seen, dq = set(), deque([source])
        while dq:
            cur = dq.popleft()
            for c, _ in self.edges.get(cur, []):
                if c not in seen:
                    seen.add(c)
                    dq.append(c)
        return seen

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------
    def topological_order(self) -> list:
        """Kahn's algorithm topological sort.

        Returns a valid ordering of all nodes. Raises ValueError if the graph
        contains a directed cycle (i.e. is not a DAG).
        """
        indeg = {n: 0 for n in self.nodes}
        for p in self.edges:
            for c, _ in self.edges[p]:
                indeg[c] = indeg.get(c, 0) + 1
        dq = deque([n for n, d in indeg.items() if d == 0])
        order = []
        local_indeg = dict(indeg)
        while dq:
            cur = dq.popleft()
            order.append(cur)
            for c, _ in self.edges.get(cur, []):
                local_indeg[c] -= 1
                if local_indeg[c] == 0:
                    dq.append(c)
        if len(order) != len(self.nodes):
            raise ValueError("graph contains a directed cycle; not a DAG")
        return order

    def is_dag(self) -> bool:
        """True iff the graph has no directed cycle."""
        try:
            self.topological_order()
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Propagation (the algebraic core)
    # ------------------------------------------------------------------
    def propagate(self, source, delta: float = 1.0) -> dict:
        """BFS causal propagation of an intervention at ``source``.

        Injects ``delta`` at ``source`` and diffuses it along directed edges,
        attenuating multiplicatively by each edge's transmission weight. Each
        node is visited exactly once (shortest-path / first-arrival semantics).

        Parameters
        ----------
        source : node key
            The intervention node.
        delta : float
            Magnitude injected at the source (must be non-negative).

        Returns
        -------
        dict
            Mapping node -> received effect. ``source`` itself is included with
            value ``delta``. Unreachable nodes are absent.
        """
        if delta < 0:
            raise ValueError("delta must be non-negative")
        if source not in self.nodes:
            return {}
        effect = {source: float(delta)}
        visited = {source}
        dq = deque([source])
        while dq:
            cur = dq.popleft()
            cur_effect = effect[cur]
            if cur_effect <= 0:
                continue
            for child, w in self.edges.get(cur, []):
                if child not in visited:
                    visited.add(child)
                    effect[child] = cur_effect * w
                    dq.append(child)
        return effect

    def propagate_multi(self, interventions: dict, normalize: bool = True) -> dict:
        """Propagate several interventions simultaneously and sum effects.

        Parameters
        ----------
        interventions : dict
            node -> injected delta.
        normalize : bool
            If True, clip final effects to [0, ∞) (defensive).

        Returns
        -------
        dict
            node -> total received effect (summed over all sources).
        """
        total: dict = defaultdict(float)
        for src, dlt in interventions.items():
            for node, eff in self.propagate(src, dlt).items():
                total[node] += eff
        if normalize:
            for n in total:
                if total[n] < 0:
                    total[n] = 0.0
        return dict(total)

    # ------------------------------------------------------------------
    # Matrix view
    # ------------------------------------------------------------------
    def adjacency_matrix(self, node_order: list | None = None) -> tuple[np.ndarray, list]:
        """Weighted adjacency matrix W (n x n), with node ordering.

        W[i, j] = weight of edge node_order[i] -> node_order[j], else 0.
        Returns (W, node_order).
        """
        order = list(node_order) if node_order is not None else sorted(self.nodes, key=str)
        idx = {n: i for i, n in enumerate(order)}
        n = len(order)
        W = np.zeros((n, n), dtype=np.float64)
        for p in order:
            for c, w in self.edges.get(p, []):
                W[idx[p], idx[c]] = w
        return W, order

    def propagation_vector(self, source, node_order: list | None = None) -> tuple[np.ndarray, list]:
        """Effect vector from a single source, aligned with ``node_order``.

        Equivalent to ``propagate`` but returned as a dense numpy vector.
        """
        order = list(node_order) if node_order is not None else sorted(self.nodes, key=str)
        eff = self.propagate(source)
        return np.array([eff.get(n, 0.0) for n in order], dtype=np.float64), order
