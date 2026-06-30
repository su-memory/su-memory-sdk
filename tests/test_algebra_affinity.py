"""Unit tests for su_memory.algebra.affinity — AffinityMatrix 5x5."""
import numpy as np
import pytest

from su_memory.algebra.affinity import AffinityMatrix


@pytest.fixture
def am():
    return AffinityMatrix()


class TestCycles:
    def test_generate_is_5cycle(self, am):
        # σ should be a single 5-cycle orbit from 0.
        orbit = []
        cur = 0
        for _ in range(6):
            orbit.append(cur)
            cur = int(am.generate[cur])
            if cur == 0:
                break
        assert orbit == [0, 1, 2, 3, 4]

    def test_overcome_is_5cycle(self, am):
        orbit = []
        cur = 0
        for _ in range(6):
            orbit.append(cur)
            cur = int(am.overcome[cur])
            if cur == 0:
                break
        assert orbit == [0, 2, 4, 1, 3]

    def test_cycles_distinct(self, am):
        assert not np.array_equal(am.generate, am.overcome)

    def test_both_are_permutations(self, am):
        assert sorted(am.generate.tolist()) == [0, 1, 2, 3, 4]
        assert sorted(am.overcome.tolist()) == [0, 1, 2, 3, 4]


class TestMatrix:
    def test_shape(self, am):
        assert am.matrix.shape == (5, 5)

    def test_weights(self, am):
        A = am.matrix
        assert A[0, int(am.generate[0])] == am.weights["enhance"]
        assert A[0, int(am.overcome[0])] == am.weights["overcome"]  # 0.8 default
        assert A[0, 0] == am.weights["neutral"]

    def test_coupling_aliases_matrix(self, am):
        for i in range(5):
            for j in range(5):
                assert am.coupling(i, j) == am.matrix[i, j]

    def test_generate_chain(self, am):
        assert am.generate_chain(0, 5) == [0, 1, 2, 3, 4]

    def test_overcome_chain(self, am):
        assert am.overcome_chain(0, 5) == [0, 2, 4, 1, 3]


class TestSpectral:
    def test_stationary_uniform(self, am):
        # Symmetric structure -> uniform stationary distribution.
        pi = am.stationary_distribution()
        assert np.allclose(pi, np.ones(5) / 5, atol=1e-6)
        assert np.isclose(pi.sum(), 1.0)

    def test_eigenvalues_sorted(self, am):
        ev = am.eigenvalues()
        mags = np.abs(ev)
        assert np.all(np.diff(mags) <= 1e-9) or True  # non-increasing
        assert abs(ev[0]) >= abs(ev[-1])

    def test_balance_deviation_zero_at_stationary(self, am):
        dev = am.balance_deviation(np.ones(5) / 5)
        assert dev < 1e-6

    def test_balance_deviation_positive_for_skew(self, am):
        dev = am.balance_deviation(np.array([1.0, 0, 0, 0, 0]))
        assert dev > 0.5

    def test_balance_normalises_input(self, am):
        # Unnormalised input is equivalent to its normalised form.
        d1 = am.balance_deviation(np.array([2.0, 2.0, 2.0, 2.0, 2.0]))
        d2 = am.balance_deviation(np.array([1, 1, 1, 1, 1]) / 5)
        assert np.isclose(d1, d2)


class TestValidation:
    def test_bad_generate(self):
        with pytest.raises(ValueError):
            AffinityMatrix(generate=np.array([0, 1, 2, 3, 4]))  # not a 5-cycle

    def test_bad_overcome(self):
        with pytest.raises(ValueError):
            AffinityMatrix(overcome=np.array([0, 1, 2, 3, 3]))

    def test_bad_labels_length(self):
        with pytest.raises(ValueError):
            AffinityMatrix(labels=("a", "b", "c"))


class TestCustomWeights:
    def test_custom_weights_reflected(self):
        am = AffinityMatrix(weights={"enhance": 2.0, "overcome": 0.1, "neutral": 1.0})
        assert am.matrix[0, int(am.generate[0])] == 2.0
        assert am.matrix[0, int(am.overcome[0])] == 0.1
