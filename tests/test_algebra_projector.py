"""Unit tests for su_memory.algebra.projector — DimensionProjector R^{k x d}."""
import numpy as np
import pytest

from su_memory.algebra.projector import DimensionProjector, _softmax_rows


@pytest.fixture
def clustered_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((300, 64)) * 0.05
    # Three clusters along canonical axes 0, 1, 2.
    X[:100] += np.eye(64)[0] * 4
    X[100:200] += np.eye(64)[1] * 4
    X[200:] += np.eye(64)[2] * 4
    return X


class TestSoftmax:
    def test_simplex_rows(self):
        z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        p = _softmax_rows(z)
        assert p.shape == (2, 3)
        assert np.allclose(p.sum(axis=1), 1.0)
        assert np.all(p >= 0)

    def test_simplex_vector(self):
        z = np.array([1.0, 2.0, 3.0])
        p = _softmax_rows(z)
        assert p.shape == (3,)
        assert np.isclose(p.sum(), 1.0)


class TestFit:
    def test_shape_after_fit(self, clustered_data):
        dp = DimensionProjector(k=8, d=64)
        dp.fit(clustered_data)
        assert dp.R.shape == (8, 64)

    def test_rows_unit_norm(self, clustered_data):
        dp = DimensionProjector(k=8, d=64)
        dp.fit(clustered_data)
        assert np.allclose(dp.row_norms(), 1.0, atol=1e-5)

    def test_rank_at_most_k(self, clustered_data):
        dp = DimensionProjector(k=8, d=64)
        dp.fit(clustered_data)
        assert dp.rank() <= 8

    def test_small_sample_padding(self):
        # Fewer samples than k -> padded with canonical basis vectors.
        dp = DimensionProjector(k=8, d=16)
        dp.fit(np.random.default_rng(0).standard_normal((3, 16)))
        assert dp.R.shape == (8, 16)
        assert np.allclose(dp.row_norms(), 1.0, atol=1e-5)

    def test_separates_clusters(self, clustered_data):
        # PCA should make the three axis-clusters land in different categories.
        dp = DimensionProjector(k=8, d=64)
        dp.fit(clustered_data)
        P = dp.project_batch(clustered_data)
        modes = [int(np.bincount(P[i * 100:(i + 1) * 100].argmax(axis=1), minlength=8).argmax())
                 for i in range(3)]
        assert len(set(modes)) == 3  # three distinct dominant categories


class TestProject:
    def test_project_simplex(self, clustered_data):
        dp = DimensionProjector(k=8, d=64).fit(clustered_data)
        p = dp.project(clustered_data[0])
        assert p.shape == (8,)
        assert np.isclose(p.sum(), 1.0)
        assert np.all(p >= 0)

    def test_project_batch_shape(self, clustered_data):
        dp = DimensionProjector(k=8, d=64).fit(clustered_data)
        P = dp.project_batch(clustered_data)
        assert P.shape == (300, 8)
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_temperature_sharpens(self, clustered_data):
        dp_hot = DimensionProjector(k=8, d=64, temperature=0.1).fit(clustered_data)
        dp_cold = DimensionProjector(k=8, d=64, temperature=10.0).fit(clustered_data)
        # Same R (PCA deterministic), different temperature.
        dp_hot.R = dp_cold.R.copy()
        p_hot = dp_hot.project(clustered_data[0])
        p_cold = dp_cold.project(clustered_data[0])
        # Hot temperature => sharper (higher max).
        assert p_hot.max() >= p_cold.max()

    def test_logits_before_fit_errors(self):
        dp = DimensionProjector(k=4, d=8)
        with pytest.raises(RuntimeError):
            dp.logits(np.zeros(8))


class TestUpdate:
    def test_incremental_update(self, clustered_data):
        dp = DimensionProjector(k=8, d=64).fit(clustered_data)
        R_before = dp.R.copy()
        dp.update(clustered_data[200:], lr=0.3)
        # R changed but remained unit-norm.
        assert not np.allclose(dp.R, R_before)
        assert np.allclose(dp.row_norms(), 1.0, atol=1e-5)

    def test_update_without_fit_does_fit(self, clustered_data):
        dp = DimensionProjector(k=8, d=64)
        dp.update(clustered_data[:50], lr=0.5)
        assert dp.R is not None
        assert dp.R.shape == (8, 64)


class TestValidation:
    def test_bad_k(self):
        with pytest.raises(ValueError):
            DimensionProjector(k=0, d=8)

    def test_bad_temperature(self):
        with pytest.raises(ValueError):
            DimensionProjector(k=4, d=8, temperature=0)

    def test_bad_R_shape(self):
        with pytest.raises(ValueError):
            DimensionProjector(k=4, d=8, R=np.zeros((3, 8)))

    def test_bad_lr(self, clustered_data):
        dp = DimensionProjector(k=4, d=64).fit(clustered_data)
        with pytest.raises(ValueError):
            dp.update(clustered_data[:10], lr=0)
