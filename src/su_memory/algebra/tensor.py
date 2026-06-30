"""HexagramTensor — the 64-element tensor product GF(2)^3 ⊗ GF(2)^3.

Mathematical foundation
-----------------------
A hexagram is a 6-bit word, canonically split into a lower trigram (lines
1-3) and an upper trigram (lines 4-6). Algebraically this is the tensor
product of two GF(2)^3 elements::

    hexagram  <=>  v_lower ⊗ v_upper   in  GF(2)^3 ⊗ GF(2)^3  ≅  GF(2)^6

There are 2^6 = 64 such elements. We order them by the King Wen sequence-free
linear ordering ``(lower_index, upper_index) -> lower_index * 8 + upper_index``
so that the 64 elements form the rows of an (64, 6) matrix over GF(2). Every
classical hexagram transformation is then a *linear or affine map* on GF(2)^6:

- **错卦 (cuo / complement)**  : bit-flip all 6 lines  ≡  affine map v ↦ v ⊕ 1.
  This is the unique non-trivial central involution of the 6-cube; it maps
  index ``i`` to ``63 - i``.
- **综卦 (zong / reverse)**    : swap lower ⊗ upper      ≡  the swap operator
  Σ : V ⊗ V → V ⊗ V, v ⊗ w ↦ w ⊗ v. On the 6-bit word it reverses the line
  order; on the (lower, upper) index pair it exchanges the two factors.
- **互卦 (hu / interleaving)** : take lines 2-3 + 4-5 to form a new pair  ≡
  a linear projection that re-tensors the interior lines.

All three are exact over GF(2) and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .basis import FU_XI_BASIS, TrigramSpace

__all__ = ["HexagramTensor"]


def _kw_pair_to_linear_index(lower: int, upper: int) -> int:
    """Map a (lower_trigram_index, upper_trigram_index) pair to 0..63.

    We use ``lower * 8 + upper`` so that the 64 rows are the cartesian product
    of two Fu Xi orderings. This is a linear (non-King-Wen) indexing chosen for
    algebraic clarity; the SDK's outer layer is responsible for any King Wen
    reordering when surfacing results to users.
    """
    if not (0 <= lower < 8 and 0 <= upper < 8):
        raise ValueError(f"trigram indices must be 0..7, got {lower},{upper}")
    return lower * 8 + upper


@dataclass
class HexagramTensor:
    """The 64-element space GF(2)^3 ⊗ GF(2)^3 with its three classical maps.

    Attributes
    ----------
    space : TrigramSpace
        The underlying single-factor space GF(2)^3.
    rows : np.ndarray
        (64, 6) int8 matrix; row ``i`` is the 6-bit word of hexagram ``i``
        with the *lower* trigram in columns 0-2 and the *upper* in 3-5.
    """

    space: TrigramSpace = field(default_factory=TrigramSpace)

    def __post_init__(self) -> None:
        b = self.space.basis  # (8,3)
        # Kronecker-style cartesian product: row (l*8+u) = [b[l], b[u]].
        rows = np.empty((64, 6), dtype=np.int8)
        for l in range(8):
            for u in range(8):
                rows[l * 8 + u] = np.concatenate([b[l], b[u]])
        self.rows = rows

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        """64."""
        return 64

    @property
    def dim(self) -> int:
        """6 (ambient dimension of GF(2)^6)."""
        return 6

    def vector(self, index: int) -> np.ndarray:
        """Return the 6-bit word of hexagram ``index`` (0..63)."""
        if not 0 <= index < 64:
            raise IndexError(f"hexagram index must be 0..63, got {index}")
        return self.rows[index].copy()

    def index_of(self, vector) -> int:
        """Inverse of :meth:`vector`."""
        v = np.asarray(vector, dtype=np.int8).ravel()
        if v.shape != (6,) or not np.all((v == 0) | (v == 1)):
            raise ValueError("vector must be a 6-vector over {0,1}")
        for i in range(64):
            if np.array_equal(v, self.rows[i]):
                return i
        raise ValueError("not a hexagram vector")

    def pair(self, index: int) -> tuple[int, int]:
        """Decompose hexagram ``index`` into (lower_trigram, upper_trigram)."""
        if not 0 <= index < 64:
            raise IndexError(f"hexagram index must be 0..63, got {index}")
        return index // 8, index % 8

    def from_pair(self, lower: int, upper: int) -> int:
        """Compose a hexagram from two trigram indices."""
        return _kw_pair_to_linear_index(lower, upper)

    # ------------------------------------------------------------------
    # The three classical linear / affine maps over GF(2)^6
    # ------------------------------------------------------------------
    def complement(self, index: int) -> int:
        """错卦 (cuogua): bit-flip all 6 lines.

        Affine map ``v ↦ v ⊕ 1`` over GF(2). On indices: ``i ↦ 63 - i``.
        This is an involutive isometry (distance-preserving) of the 6-cube.
        """
        if not 0 <= index < 64:
            raise IndexError(f"hexagram index must be 0..63, got {index}")
        return 63 - index

    def swap_factors(self, index: int) -> int:
        """综卦 (zonggua): exchange the lower and upper trigrams.

        Linear map: the swap operator Σ on V ⊗ V. On the (lower, upper) pair
        it exchanges the two factors. NOT distance-preserving in general
        (it is an involution but not an isometry of the cube metric).
        """
        lower, upper = self.pair(index)
        return self.from_pair(upper, lower)

    def interleave(self, index: int) -> int:
        """互卦 (hugua): re-tensor the interior lines.

        Take lines 2,3 (top of lower trigram) and lines 4,5 (bottom of upper
        trigram) to form a *new* lower and upper trigram, with the boundary
        line reused. Implemented as an exact linear projection over GF(2):

            new_lower = [line2, line3, line4]
            new_upper = [line3, line4, line5]

        where the 6-bit word is [l1,l2,l3, u4,u5,u6] (1-indexed lines).
        """
        v = self.vector(index)  # [l0,l1,l2, u3,u4,u5]  (0-indexed)
        new_lower = np.array([v[1], v[2], v[3]], dtype=np.int8)
        new_upper = np.array([v[2], v[3], v[4]], dtype=np.int8)
        lo = self.space.index_of(new_lower)
        up = self.space.index_of(new_upper)
        return self.from_pair(lo, up)

    # ------------------------------------------------------------------
    # Metric structure
    # ------------------------------------------------------------------
    @property
    def adjacency_cube(self) -> np.ndarray:
        """(64, 64) Hamming-distance adjacency of the 6-cube."""
        diff = self.rows[:, None, :] ^ self.rows[None, :, :]
        return diff.sum(axis=2).astype(np.int8)

    def hamming(self, a: int, b: int) -> int:
        """Hamming distance between hexagrams ``a`` and ``b`` on the 6-cube."""
        return int(self.adjacency_cube[a, b])

    # ------------------------------------------------------------------
    # Operator-matrix view (for introspection / verification)
    # ------------------------------------------------------------------
    @property
    def swap_matrix(self) -> np.ndarray:
        """(64, 64) permutation matrix realising 综卦 (factor swap).

        ``P[i, j] = 1`` iff ``swap_factors(j) == i``.
        """
        P = np.zeros((64, 64), dtype=np.int8)
        for j in range(64):
            P[self.swap_factors(j), j] = 1
        return P

    @property
    def complement_matrix(self) -> np.ndarray:
        """(64, 64) permutation matrix realising 错卦 (bit flip).

        ``C[i, j] = 1`` iff ``complement(j) == i``.
        """
        C = np.zeros((64, 64), dtype=np.int8)
        for j in range(64):
            C[self.complement(j), j] = 1
        return C
