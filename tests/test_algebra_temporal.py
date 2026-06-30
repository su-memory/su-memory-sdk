"""Unit tests for su_memory.algebra.temporal — TemporalRing Z_60."""
import numpy as np
import pytest

from su_memory.algebra.temporal import TemporalRing


@pytest.fixture
def tr():
    return TemporalRing()


class TestStructure:
    def test_constants(self, tr):
        assert tr.order == 60
        assert tr.stem_period == 10
        assert tr.branch_period == 12

    def test_lcm_is_order(self, tr):
        assert int(np.lcm(tr.stem_period, tr.branch_period)) == tr.order


class TestResidues:
    def test_stem_residue(self, tr):
        assert tr.stem(0) == 0
        assert tr.stem(12) == 2  # 12 mod 10
        assert tr.stem(59) == 9

    def test_branch_residue(self, tr):
        assert tr.branch(0) == 0
        assert tr.branch(12) == 0  # 12 mod 12
        assert tr.branch(59) == 11

    def test_roundtrip_from_residues(self, tr):
        for t in range(60):
            assert tr.from_residues(tr.stem(t), tr.branch(t)) == t

    def test_parity_constraint(self, tr):
        # gcd(10,12)=2: only parity-matched (stem,branch) pairs are valid.
        with pytest.raises(ValueError):
            tr.from_residues(0, 1)  # even stem, odd branch

    def test_sixty_valid_pairs(self, tr):
        pairs = tr.valid_pairs()
        assert len(set(pairs)) == 60
        assert all(s % 2 == b % 2 for s, b in pairs)


class TestGroupArithmetic:
    def test_add_mod_60(self, tr):
        assert tr.add(0, 0) == 0
        assert tr.add(58, 5) == 3
        assert tr.add(59, 1) == 0

    def test_distance_toroidal(self, tr):
        assert tr.distance(0, 0) == 0
        assert tr.distance(0, 30) == 30
        assert tr.distance(0, 59) == 1  # wrap-around
        assert tr.distance(0, 31) == 29

    def test_distance_max_thirty(self, tr):
        for a in range(60):
            for b in range(60):
                assert tr.distance(a, b) <= 30


class TestPhaseRelations:
    def test_stem_distance(self, tr):
        assert tr.stem_distance(0, 5) == 5
        assert tr.stem_distance(0, 9) == 1

    def test_branch_distance(self, tr):
        assert tr.branch_distance(0, 6) == 6
        assert tr.branch_distance(0, 1) == 1

    def test_opposition_half_branch(self, tr):
        # 冲: branch distance == 6 (half of 12).
        assert tr.is_opposition(0, 6)
        assert not tr.is_opposition(0, 1)

    def test_combination_half_stem(self, tr):
        # 合: stem distance == 5 (half of 10).
        assert tr.is_combination(0, 5)
        assert not tr.is_combination(0, 1)


class TestPolarity:
    def test_parity_classes(self, tr):
        yang = [t for t in range(60) if tr.polarity(t) == 1]
        yin = [t for t in range(60) if tr.polarity(t) == 0]
        assert len(yang) == 30
        assert len(yin) == 30


class TestResidueTable:
    def test_shape(self, tr):
        rt = tr.residue_table()
        assert rt.shape == (60, 2)

    def test_unique_rows(self, tr):
        rt = tr.residue_table()
        assert len({tuple(r) for r in rt.tolist()}) == 60


class TestValidation:
    def test_bad_t(self, tr):
        with pytest.raises(ValueError):
            tr.stem(60)
        with pytest.raises(ValueError):
            tr.branch(-1)

    def test_non_int_t(self, tr):
        with pytest.raises(TypeError):
            tr.stem(1.5)
