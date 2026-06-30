"""TrigramSpace — the 8-element vector space GF(2)^3.

Mathematical foundation
-----------------------
The eight trigrams are the eight non-zero/zero vectors of the binary vector
space GF(2)^3 — equivalently the projective points of PG(2,2) plus the zero
vector. We index them by the Fu Xi (伏羲) ordering, which corresponds to the
standard big-endian binary read of the three coordinates::

    index | vector (a0,a1,a2) | legacy label
    ------+-------------------+--------------
        0 | (1, 1, 1)         | QIAN  (creative)
        1 | (0, 1, 1)         | DUI   (lake)
        2 | (1, 0, 1)         | LI    (light)
        3 | (0, 0, 1)         | ZHEN  (thunder)
        4 | (1, 1, 0)         | XUN   (wind)
        5 | (0, 1, 0)         | KAN   (abyss)
        6 | (1, 0, 0)         | GEN   (mountain)
        7 | (0, 0, 0)         | KUN   (receptive)

In GF(2)^3:
- addition is component-wise XOR,
- the zero vector is KUN (0,0,0),
- the additive inverse of any vector is itself,
- every non-zero vector has order 2.

This module is pure linear algebra over GF(2). The legacy symbolic names are
carried only as an optional label map for interop with the SDK's outer layers;
they never participate in computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Sequence

import numpy as np

__all__ = [
    "TrigramSpace",
    "FU_XI_BASIS",
    "LEGACY_LABELS",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The 8 basis vectors in Fu Xi order, as an (8,3) int8 array over GF(2).
FU_XI_BASIS: np.ndarray = np.array(
    [
        [1, 1, 1],  # 0 QIAN
        [0, 1, 1],  # 1 DUI
        [1, 0, 1],  # 2 LI
        [0, 0, 1],  # 3 ZHEN
        [1, 1, 0],  # 4 XUN
        [0, 1, 0],  # 5 KAN
        [1, 0, 0],  # 6 GEN
        [0, 0, 0],  # 7 KUN
    ],
    dtype=np.int8,
)

# Legacy symbolic names kept purely for interop / debugging. Order matches
# FU_XI_BASIS rows. These are NOT used by any computation in this layer.
LEGACY_LABELS: tuple[str, ...] = (
    "QIAN",
    "DUI",
    "LI",
    "ZHEN",
    "XUN",
    "KAN",
    "GEN",
    "KUN",
)


@dataclass(frozen=True)
class TrigramSpace:
    """The vector space GF(2)^3 with its standard Fu Xi basis.

    All operations are exact over GF(2) (no floating point involved), so the
    results are bit-for-bit reproducible across machines.

    Attributes
    ----------
    basis : np.ndarray
        (8, 3) int8 basis matrix; row ``i`` is the vector of trigram ``i``.
    dim : int
        Always 3 (ambient dimension of GF(2)^3).
    size : int
        Always 8 (number of elements).
    """

    basis: np.ndarray = field(default_factory=lambda: FU_XI_BASIS.copy())

    def __post_init__(self) -> None:
        b = np.asarray(self.basis, dtype=np.int8)
        if b.shape != (8, 3):
            raise ValueError(f"basis must be (8,3), got {b.shape}")
        # Verify it really spans GF(2)^3: all 8 distinct vectors present.
        rows = {tuple(int(x) for x in row) for row in b}
        if len(rows) != 8 or not all(
            all(v in (0, 1) for v in r) for r in rows
        ):
            raise ValueError("basis rows must be the 8 distinct GF(2)^3 vectors")
        object.__setattr__(self, "basis", b)

    # ------------------------------------------------------------------
    # Basic geometry
    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        """Ambient dimension, always 3."""
        return 3

    @property
    def size(self) -> int:
        """Number of elements, always 8."""
        return 8

    def vector(self, index: int) -> np.ndarray:
        """Return the basis vector of trigram ``index`` (Fu Xi order)."""
        if not 0 <= index < 8:
            raise IndexError(f"trigram index must be 0..7, got {index}")
        return self.basis[index].copy()

    def index_of(self, vector: Sequence[int] | np.ndarray) -> int:
        """Inverse of :meth:`vector`: vector -> Fu Xi index."""
        v = np.asarray(vector, dtype=np.int8).ravel()
        if v.shape != (3,) or not np.all((v == 0) | (v == 1)):
            raise ValueError("vector must be a 3-vector over {0,1}")
        for i in range(8):
            if np.array_equal(v, self.basis[i]):
                return i
        raise ValueError(f"{tuple(v.tolist())} is not a Fu Xi basis vector")

    # ------------------------------------------------------------------
    # GF(2) arithmetic
    # ------------------------------------------------------------------
    def add(
        self,
        a: Sequence[int] | np.ndarray,
        b: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """Vector addition over GF(2) (component-wise XOR).

        Because every element is its own inverse, ``add`` is also subtraction.
        """
        va = np.asarray(a, dtype=np.int8).ravel()
        vb = np.asarray(b, dtype=np.int8).ravel()
        if va.shape != (3,) or vb.shape != (3,):
            raise ValueError("operands must be 3-vectors")
        return (va ^ vb).astype(np.int8)

    def complement(self, vector: Sequence[int] | np.ndarray) -> np.ndarray:
        """Bitwise complement a.k.a. affine flip (1-v) over GF(2).

        Geometrically this is the trigram 错卦 (cuogua): swap every line.
        """
        v = np.asarray(vector, dtype=np.int8).ravel()
        if v.shape != (3,):
            raise ValueError("vector must be a 3-vector")
        return (1 - v).astype(np.int8)

    def hamming(self, a: Sequence[int] | np.ndarray, b: Sequence[int] | np.ndarray) -> int:
        """Hamming distance = popcount(a XOR b) over GF(2).

        This is the canonical graph metric on the 3-cube {0,1}^3.
        """
        return int(np.count_nonzero(self.add(a, b)))

    # ------------------------------------------------------------------
    # Aggregate views
    # ------------------------------------------------------------------
    @cached_property
    def gram(self) -> np.ndarray:
        r"""Gram matrix G = B B^T mod 2 (8x8).

        Entry (i,j) = inner product of trigram i and trigram j over GF(2),
        i.e. parity of the bitwise AND. Diagonal = parity of the weight.
        """
        prod = (self.basis.astype(np.int16) @ self.basis.T.astype(np.int16)) % 2
        return prod.astype(np.int8)

    @cached_property
    def adjacency_cube(self) -> np.ndarray:
        """(8,8) Hamming-distance adjacency of the 3-cube.

        Entry (i,j) is the Hamming distance between trigrams i and j.
        """
        diff = (
            self.basis[:, None, :] ^ self.basis[None, :, :]
        )  # (8,8,3)
        return diff.sum(axis=2).astype(np.int8)

    def label(self, index: int) -> str:
        """Legacy symbolic name (interop only; not used by math)."""
        return LEGACY_LABELS[index]
