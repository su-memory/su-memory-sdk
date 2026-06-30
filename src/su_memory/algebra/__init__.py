"""su_memory.algebra — pure mathematical object layer.

This package contains the linear-algebra / probability-graph primitives that
underlie su-memory's structured multi-dimensional coordinate system.

Design contract:
- NO I/O, NO side effects, NO dependency on the rest of the SDK.
- Every public object is constructible from plain numpy arrays and must be
  unit-testable in isolation.
- Symbolic labels (legacy trigram names, energy names, ...) are carried as
  optional string tags only; all computation works on the underlying vectors /
  matrices / graph topology.

Modules
-------
basis        : TrigramSpace  — the 8-element vector space GF(2)^3.
tensor       : HexagramTensor — the 64-element tensor product GF(2)^3 ⊗ GF(2)^3.
projector    : DimensionProjector — learnable projection R^{k x d} (PCA-init).
affinity     : AffinityMatrix — 5x5 energy coupling matrix (generating/overcoming).
temporal     : TemporalRing — Z_60 cyclic-group coordinate (60-cycle ring).
causal_graph : CausalDAG — directed acyclic graph with BFS belief propagation.
belief_net   : BeliefNetwork — Beta-Bernoulli conjugate + loopy belief propagation.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - trivial
    __version__ = version("su-memory")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
