"""TemporalRing — the cyclic group Z_60 (the 60-cycle temporal torus).

Mathematical foundation
-----------------------
The 60-cycle (legacy: 天干地支 / sexagenary cycle) is the cyclic group Z_60.
Each position t ∈ {0, ..., 59} decomposes, by the Chinese Remainder Theorem,
into two residues on two smaller cycles:

    t mod 10  : the 10-cycle  (legacy: ten stems)
    t mod 12  : the 12-cycle  (legacy: twelve branches)

Because gcd(10, 12) = 2 ≠ 1, the map Z_60 → Z_10 × Z_12 is *not* a ring
isomorphism — it is a 2-to-1 surjection whose image is the fibre product
{ (a, b) ∈ Z_10 × Z_12 : a ≡ b (mod 2) }. The 60 valid pairs are exactly the
parity-matched ones; this is why there are 60 (not 120) valid codes.

Geometrically, Z_60 is a 1-dimensional torus S^1 sampled at 60 points; the
two residue maps are the two natural quotient projections onto shorter
tori. Time arithmetic is addition mod 60; "advance k steps" is t ↦ t + k
mod 60. Phase relationships (合 / 冲 / 刑) are integer distance classes on
these cycles — e.g. branch 冲 (opposition) is distance 6 on the 12-cycle.

This module implements Z_60 as a pure cyclic-group coordinate, with the two
residue projections and the canonical distance / phase metrics. Legacy stem
and branch *names* are optional labels carried by the outer SDK layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TemporalRing"]


@dataclass(frozen=True)
class TemporalRing:
    """The cyclic group Z_60 with its two CRT residue projections.

    Attributes
    ----------
    order : int
        Group order, always 60.
    stem_period : int
        Period of the stem residue map, always 10.
    branch_period : int
        Period of the branch residue map, always 12.
    """

    order: int = 60
    stem_period: int = 10
    branch_period: int = 12

    def __post_init__(self) -> None:
        if self.order != 60:
            raise ValueError("TemporalRing order is fixed at 60")
        if self.stem_period != 10 or self.branch_period != 12:
            raise ValueError("stem/branch periods are 10 and 12")
        # The two periods must generate the full 60-cycle: lcm(10,12)=60.
        if np.lcm(self.stem_period, self.branch_period) != self.order:
            raise ValueError("periods must have lcm 60")

    # ------------------------------------------------------------------
    # Coordinate access
    # ------------------------------------------------------------------
    def stem(self, t: int) -> int:
        """Stem residue t mod 10 (legacy: one of ten stems)."""
        self._check(t)
        return t % self.stem_period

    def branch(self, t: int) -> int:
        """Branch residue t mod 12 (legacy: one of twelve branches)."""
        self._check(t)
        return t % self.branch_period

    def from_residues(self, stem: int, branch: int) -> int:
        """Inverse of the residue map: find t in 0..59 with given residues.

        Uses CRT on the fibre product. Raises ValueError if the parity
        constraint (stem ≡ branch mod 2) is violated, since such a pair is
        not a valid 60-cycle element.
        """
        if not (0 <= stem < self.stem_period and 0 <= branch < self.branch_period):
            raise ValueError("residues out of range")
        if stem % 2 != branch % 2:
            raise ValueError(
                f"invalid pair ({stem},{branch}): stem and branch must share "
                f"parity (CRT fibre-product constraint over Z_60)"
            )
        for t in range(self.order):
            if t % self.stem_period == stem and t % self.branch_period == branch:
                return t
        # Unreachable given the parity check, but keep for safety.
        raise ValueError(f"no valid t for ({stem},{branch})")

    def _check(self, t: int) -> None:
        if not isinstance(t, (int, np.integer)):
            raise TypeError(f"t must be int, got {type(t)}")
        if not 0 <= t < self.order:
            raise ValueError(f"t must be in 0..{self.order - 1}, got {t}")

    # ------------------------------------------------------------------
    # Group arithmetic
    # ------------------------------------------------------------------
    def add(self, a: int, b: int) -> int:
        """Group addition a + b mod 60 (advance b steps from a)."""
        return (a + b) % self.order

    def distance(self, a: int, b: int) -> int:
        """Cyclic (toroidal) distance on Z_60: min forward/backward steps.

        Returns an int in 0..30.
        """
        self._check(a)
        self._check(b)
        d = abs(a - b) % self.order
        return min(d, self.order - d)

    # ------------------------------------------------------------------
    # Phase relationships (distance classes on the two sub-cycles)
    # ------------------------------------------------------------------
    def stem_distance(self, a: int, b: int) -> int:
        """Cyclic distance of the stem residues (on Z_10)."""
        return self._cyclic_dist(self.stem(a), self.stem(b), self.stem_period)

    def branch_distance(self, a: int, b: int) -> int:
        """Cyclic distance of the branch residues (on Z_12)."""
        return self._cyclic_dist(self.branch(a), self.branch(b), self.branch_period)

    @staticmethod
    def _cyclic_dist(x: int, y: int, n: int) -> int:
        d = abs(x - y) % n
        return min(d, n - d)

    def is_opposition(self, a: int, b: int) -> bool:
        """Branch 冲 (opposition): branch distance == 6 (half of 12).

        Two positions whose branches are diametrically opposite on the
        12-cycle. This is the algebraic root of the legacy "六冲".
        """
        return self.branch_distance(a, b) == self.branch_period // 2

    def is_combination(self, a: int, b: int) -> bool:
        """Stem 合 (combination): stem distance == 5 (half of 10).

        Two positions whose stems are diametrically opposite on the 10-cycle.
        Algebraic root of the legacy "五合".
        """
        return self.stem_distance(a, b) == self.stem_period // 2

    # ------------------------------------------------------------------
    # Aggregate structure
    # ------------------------------------------------------------------
    def residue_table(self) -> np.ndarray:
        """(60, 2) array of [stem, branch] residues for t = 0..59."""
        return np.array(
            [[self.stem(t), self.branch(t)] for t in range(self.order)],
            dtype=np.int8,
        )

    def valid_pairs(self) -> list[tuple[int, int]]:
        """The 60 parity-matched (stem, branch) pairs (= Z_60 elements)."""
        return [(self.stem(t), self.branch(t)) for t in range(self.order)]

    def polarity(self, t: int) -> int:
        """Parity of t (0 = yin, 1 = yang).

        Because stem ≡ branch ≡ t (mod 2), this single bit labels the CRT
        parity class. The 60-cycle splits into two interleaved cosets of 30.
        """
        self._check(t)
        return int(t % 2)
