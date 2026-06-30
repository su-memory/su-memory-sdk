"""Unit tests for su_memory.algebra.tensor — HexagramTensor GF(2)^3 ⊗ GF(2)^3."""
import numpy as np
import pytest

from su_memory.algebra.tensor import HexagramTensor
from su_memory.algebra.basis import TrigramSpace


@pytest.fixture
def ht():
    return HexagramTensor()


class TestStructure:
    def test_shape(self, ht):
        assert ht.rows.shape == (64, 6)
        assert ht.size == 64
        assert ht.dim == 6

    def test_all_sixtyfour_distinct(self, ht):
        rows = {tuple(int(x) for x in r) for r in ht.rows}
        assert len(rows) == 64

    def test_all_binary(self, ht):
        assert np.all((ht.rows == 0) | (ht.rows == 1))

    def test_pair_decomposition(self, ht):
        # index = lower*8 + upper, lower in cols 0-2, upper in cols 3-5.
        for idx in (0, 9, 27, 63):
            lo, up = ht.pair(idx)
            assert idx == lo * 8 + up
            v = ht.vector(idx)
            assert v[:3].tolist() == TrigramSpace().vector(lo).tolist()
            assert v[3:].tolist() == TrigramSpace().vector(up).tolist()


class TestComplementMap:
    def test_is_involution(self, ht):
        for i in range(64):
            assert ht.complement(ht.complement(i)) == i

    def test_index_relation(self, ht):
        # complement(i) = 63 - i (bit-flip of the 6-bit word).
        for i in range(0, 64, 7):
            assert ht.complement(i) == 63 - i

    def test_preserves_hamming(self, ht):
        # The complement is an isometry of the cube.
        for i in range(64):
            for j in range(64):
                if i < j:
                    assert ht.hamming(i, j) == ht.hamming(
                        ht.complement(i), ht.complement(j)
                    )


class TestSwapFactorsMap:
    def test_is_involution(self, ht):
        for i in range(64):
            assert ht.swap_factors(ht.swap_factors(i)) == i

    def test_fixed_points_are_diagonal(self, ht):
        # Fixed points are hexagrams with lower == upper trigram: exactly 8.
        fixed = [i for i in range(64) if ht.swap_factors(i) == i]
        assert len(fixed) == 8
        for i in fixed:
            lo, up = ht.pair(i)
            assert lo == up

    def test_swaps_factors(self, ht):
        lo, up = 3, 5
        idx = ht.from_pair(lo, up)
        assert ht.pair(ht.swap_factors(idx)) == (up, lo)


class TestInterleaveMap:
    def test_self_interleave_for_uniform(self, ht):
        # The all-1 hexagram (QIAN ⊗ QIAN) and all-0 (KUN ⊗ KUN) are fixed.
        assert ht.interleave(0) == 0  # all-ones
        assert ht.interleave(63) == 63  # all-zeros

    def test_uses_interior_lines(self, ht):
        # Verify interleave really takes lines 2,3,4,5 (0-indexed 1..4).
        for idx in (1, 7, 12, 40, 55):
            v = ht.vector(idx)
            from su_memory.algebra.basis import TrigramSpace

            ts = TrigramSpace()
            new_lo = ts.index_of(np.array([v[1], v[2], v[3]]))
            new_up = ts.index_of(np.array([v[2], v[3], v[4]]))
            assert ht.interleave(idx) == ht.from_pair(new_lo, new_up)


class TestMetric:
    def test_adjacency_shape(self, ht):
        assert ht.adjacency_cube.shape == (64, 64)

    def test_adjacency_symmetric(self, ht):
        assert np.array_equal(ht.adjacency_cube, ht.adjacency_cube.T)

    def test_hamming_max_six(self, ht):
        assert ht.adjacency_cube.max() == 6


class TestOperatorMatrices:
    def test_swap_matrix_is_involution(self, ht):
        P = ht.swap_matrix
        assert np.array_equal(P @ P % 2, np.eye(64, dtype=np.int8))

    def test_complement_matrix_is_involution(self, ht):
        C = ht.complement_matrix
        assert np.array_equal(C @ C % 2, np.eye(64, dtype=np.int8))

    def test_matrices_are_permutations(self, ht):
        # Each row/col of a permutation matrix has exactly one 1.
        for M in (ht.swap_matrix, ht.complement_matrix):
            assert np.all(M.sum(axis=0) == 1)
            assert np.all(M.sum(axis=1) == 1)
