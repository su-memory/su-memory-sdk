"""Unit tests for su_memory.algebra.basis — TrigramSpace GF(2)^3.

Pure linear-algebra assertions, no SDK coupling. Every claim is reproducible.
"""
import numpy as np
import pytest

from su_memory.algebra.basis import FU_XI_BASIS, LEGACY_LABELS, TrigramSpace


@pytest.fixture
def ts():
    return TrigramSpace()


class TestSpaceStructure:
    def test_shape_and_size(self, ts):
        assert ts.basis.shape == (8, 3)
        assert ts.size == 8
        assert ts.dim == 3

    def test_basis_is_complete_gf2(self, ts):
        # The 8 rows must be exactly the 8 distinct GF(2)^3 vectors.
        rows = {tuple(int(x) for x in r) for r in ts.basis}
        assert rows == {
            (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
            (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
        }

    def test_all_entries_binary(self, ts):
        assert np.all((ts.basis == 0) | (ts.basis == 1))


class TestVectorIndexRoundtrip:
    def test_roundtrip(self, ts):
        for i in range(8):
            assert ts.index_of(ts.vector(i)) == i

    def test_index_of_rejects_non_basis(self, ts):
        # (1,1,1) exists; but flipping to a non-basis-like value errors only if
        # outside {0,1}. Here we check the rejection of non-binary input.
        with pytest.raises(ValueError):
            ts.index_of([2, 0, 0])

    def test_index_out_of_range(self, ts):
        with pytest.raises(IndexError):
            ts.vector(8)
        with pytest.raises(IndexError):
            ts.vector(-1)


class TestGFArithmetic:
    def test_add_is_xor(self, ts):
        # XOR addition: QIAN(111) + DUI(011) = 100 = GEN
        assert ts.add([1, 1, 1], [0, 1, 1]).tolist() == [1, 0, 0]
        assert ts.index_of(ts.add([1, 1, 1], [0, 1, 1])) == 6  # GEN

    def test_add_self_is_zero(self, ts):
        # In GF(2) every element is its own inverse: v + v = 0.
        for i in range(8):
            assert ts.add(ts.vector(i), ts.vector(i)).tolist() == [0, 0, 0]

    def test_complement_is_bitflip(self, ts):
        # complement flips every bit; QIAN(111) -> KUN(000).
        assert ts.complement([1, 1, 1]).tolist() == [0, 0, 0]
        assert ts.complement([0, 1, 0]).tolist() == [1, 0, 1]

    def test_complement_is_involution(self, ts):
        for i in range(8):
            v = ts.vector(i)
            assert ts.index_of(ts.complement(ts.complement(v))) == i

    def test_hamming_is_cube_metric(self, ts):
        # Hamming(QIAN=111, KUN=000) = 3 (opposite corners of the 3-cube).
        assert ts.hamming([1, 1, 1], [0, 0, 0]) == 3
        assert ts.hamming([1, 1, 1], [1, 1, 1]) == 0
        assert ts.hamming([1, 0, 1], [0, 0, 1]) == 1


class TestAggregateViews:
    def test_adjacency_cube_shape(self, ts):
        assert ts.adjacency_cube.shape == (8, 8)

    def test_adjacency_diagonal_zero(self, ts):
        assert np.all(np.diag(ts.adjacency_cube) == 0)

    def test_adjacency_symmetric(self, ts):
        assert np.array_equal(ts.adjacency_cube, ts.adjacency_cube.T)

    def test_adjacency_max_is_three(self, ts):
        # On the 3-cube the max Hamming distance is 3.
        assert ts.adjacency_cube.max() == 3

    def test_gram_shape_and_binary(self, ts):
        assert ts.gram.shape == (8, 8)
        assert np.all((ts.gram == 0) | (ts.gram == 1))


class TestInvalidConstruction:
    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError):
            TrigramSpace(basis=np.zeros((7, 3), dtype=np.int8))

    def test_duplicate_rows_rejected(self):
        bad = np.ones((8, 3), dtype=np.int8)  # all rows identical
        with pytest.raises(ValueError):
            TrigramSpace(basis=bad)


class TestLabels:
    def test_label_count(self, ts):
        assert len(LEGACY_LABELS) == 8
        for i in range(8):
            assert isinstance(ts.label(i), str)
