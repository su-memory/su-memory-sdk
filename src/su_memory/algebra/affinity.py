"""AffinityMatrix — 5x5 directed coupling between five energy sub-spaces.

Mathematical foundation
-----------------------
The five "energies" (legacy: wood/fire/earth/metal/water) form a 5-element set
equipped with two distinct 5-cycle permutations:

- the *generating* cycle  σ : i ↦ σ(i)   (wood→fire→earth→metal→water→wood)
- the *overcoming* cycle  τ : i ↦ τ(i)   (wood→earth→water→fire→metal→wood)

σ and τ are both elements of the symmetric group S_5. Their interaction
generates the dihedral structure of the pentagon. We encode the *coupling
strength* between any ordered pair (i, j) as a single 5×5 real matrix A where:

    A[i, j] > 1  : i enhances j            (j = σ(i))
    A[i, j] < 1  : i diminishes j          (j = τ(i), default -20%)
    A[i, j] = 1  : neutral / self

The default weights (enhance ≈ 1.2, overcome ≈ 0.4, etc.) reproduce the legacy
boost factors. The matrix is exposed as a plain numpy array so it can be
spectrally analysed (eigenvalues, stationary distribution) and re-weighted
without touching any symbolic code.

This module is pure linear algebra. Legacy symbolic names are carried only as
an optional label list for interop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["AffinityMatrix", "DEFAULT_LABELS"]

# Default legacy-free labels (kept for interop / debugging only).
DEFAULT_LABELS: tuple[str, ...] = ("E0", "E1", "E2", "E3", "E4")


@dataclass
class AffinityMatrix:
    """5×5 directed coupling matrix with two canonical 5-cycles.

    The two cycles are stored as permutation arrays (length-5 numpy arrays of
    indices). ``generate[i]`` is the index that ``i`` enhances; ``overcome[i]``
    is the index that ``i`` diminishes.

    Attributes
    ----------
    generate : np.ndarray
        Length-5 int array; the generating 5-cycle σ.
    overcome : np.ndarray
        Length-5 int array; the overcoming 5-cycle τ.
    weights : dict
        Symbolic weight table. Keys: "enhance", "overcome", "neutral".
        Defaults reproduce the legacy boost factors.
    labels : tuple
        Optional 5 labels for interop.
    """

    # The two canonical 5-cycles over indices 0..4.
    #   σ : 0→1→2→3→4→0   (legacy: wood→fire→earth→metal→water)
    #   τ : 0→2→4→1→3→0  i.e. τ[i] in [2,3,4,0,1]  (wood→earth→water→fire→metal)
    generate: np.ndarray = field(
        default_factory=lambda: np.array([1, 2, 3, 4, 0], dtype=np.intp)
    )
    overcome: np.ndarray = field(
        default_factory=lambda: np.array([2, 3, 4, 0, 1], dtype=np.intp)
    )
    weights: dict = field(
        default_factory=lambda: {
            "enhance": 1.2,   # i generates j  -> +20% coupling
            "overcome": 0.8,  # i diminishes j -> -20% coupling (aligned with legacy SUPPRESS)
            "neutral": 1.0,
        }
    )
    labels: tuple = DEFAULT_LABELS

    def __post_init__(self) -> None:
        g = np.asarray(self.generate, dtype=np.intp)
        o = np.asarray(self.overcome, dtype=np.intp)
        if g.shape != (5,) or o.shape != (5,):
            raise ValueError("cycles must be length-5 arrays")
        if sorted(g.tolist()) != [0, 1, 2, 3, 4]:
            raise ValueError("generate must be a permutation of 0..4")
        if sorted(o.tolist()) != [0, 1, 2, 3, 4]:
            raise ValueError("overcome must be a permutation of 0..4")
        # Both must be genuine 5-cycles (one orbit).
        if len(self._orbit(g, 0)) != 5:
            raise ValueError("generate must be a single 5-cycle")
        if len(self._orbit(o, 0)) != 5:
            raise ValueError("overcome must be a single 5-cycle")
        if len(self.labels) != 5:
            raise ValueError("labels must have 5 entries")
        self.generate = g
        self.overcome = o

    @staticmethod
    def _orbit(perm: np.ndarray, start: int) -> list[int]:
        seen, cur = [], start
        for _ in range(len(perm) + 1):
            seen.append(int(cur))
            cur = int(perm[cur])
            if cur == start:
                break
        return seen

    # ------------------------------------------------------------------
    # Matrix construction
    # ------------------------------------------------------------------
    @property
    def matrix(self) -> np.ndarray:
        """The 5×5 coupling matrix A.

        A[i, j] is the weight with which energy i influences energy j:
        - A[i, generate[i]] = weights["enhance"]
        - A[i, overcome[i]] = weights["overcome"]
        - A[i, i] = weights["neutral"]
        - otherwise A[i, j] = weights["neutral"]
        """
        A = np.full((5, 5), self.weights["neutral"], dtype=np.float64)
        for i in range(5):
            A[i, int(self.generate[i])] = self.weights["enhance"]
            A[i, int(self.overcome[i])] = self.weights["overcome"]
            A[i, i] = self.weights["neutral"]
        return A

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def coupling(self, i: int, j: int) -> float:
        """A[i, j]: directed coupling weight from energy i to j."""
        return float(self.matrix[i, j])

    def boost_factor(self, i: int, j: int) -> float:
        """Alias for :meth:`coupling` (interop with legacy naming)."""
        return self.coupling(i, j)

    def generate_chain(self, start: int, steps: int = 5) -> list[int]:
        """Walk the generating cycle σ for ``steps`` from ``start``."""
        seq = [start]
        cur = start
        for _ in range(steps - 1):
            cur = int(self.generate[cur])
            seq.append(cur)
        return seq

    def overcome_chain(self, start: int, steps: int = 5) -> list[int]:
        """Walk the overcoming cycle τ for ``steps`` from ``start``."""
        seq = [start]
        cur = start
        for _ in range(steps - 1):
            cur = int(self.overcome[cur])
            seq.append(cur)
        return seq

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------
    def stationary_distribution(self) -> np.ndarray:
        """Principal left eigenvector of the row-normalised coupling graph.

        Computes the stationary distribution π of the Markov chain induced by
        row-normalising A. This is the long-run "balance" distribution over the
        five energies under repeated coupling — the algebraic meaning of the
        legacy ``analyze_balance``.
        """
        A = self.matrix
        # Row-normalise to a stochastic matrix.
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P = A / row_sums
        # Power iteration for the stationary (left) distribution.
        pi = np.full(5, 1.0 / 5.0)
        for _ in range(1000):
            new = pi @ P
            if np.allclose(new, pi, atol=1e-12):
                pi = new
                break
            pi = new
        pi = pi / pi.sum()
        return pi

    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of A (complex), sorted by descending magnitude."""
        w = np.linalg.eigvals(self.matrix)
        return w[np.argsort(-np.abs(w))]

    # ------------------------------------------------------------------
    # Balance metric
    # ------------------------------------------------------------------
    def balance_deviation(self, distribution: np.ndarray) -> float:
        """L2 distance of ``distribution`` from the stationary distribution.

        A distribution close to the stationary one is "balanced"; deviation
        measures imbalance. Returns a non-negative float.
        """
        d = np.asarray(distribution, dtype=np.float64).ravel()
        if d.shape != (5,):
            raise ValueError("distribution must be length-5")
        if d.sum() <= 0:
            raise ValueError("distribution must be non-negative and non-zero")
        d = d / d.sum()
        stat = self.stationary_distribution()
        return float(np.linalg.norm(d - stat))
