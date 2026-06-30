"""DimensionProjector — learnable linear projection R^{k x d}.

Mathematical foundation
-----------------------
The legacy encoder mapped a d-dimensional semantic embedding ``x`` to an
8-category distribution by the *ad hoc* recipe "take the first 16 dims, fold
two dims per category, mean, softmax". That map is:

- lossy (throws away all but 16 dims of, say, a 1024-dim bge-m3 vector),
- non-learnable (fixed coefficients regardless of data),
- not even a proper linear projection (the mean-fold mixes coordinates
  without regard for variance).

The correct linear-algebraic object is a **projection matrix** ``R`` of shape
(k, d) that produces logits ``z = R x`` followed by a softmax to obtain a
distribution over k trigram categories (k = 8), or over k = 5 energy
sub-spaces. We initialise ``R`` by Principal Component Analysis on a sample of
real embeddings, so the projection captures the directions of maximal variance
in the *actual* data manifold rather than an arbitrary fixed slicing.

This module is pure numpy: no I/O, no model loading, no SDK coupling. The
outer encoder layer is responsible for feeding it real embeddings and for
caching ``R`` to disk if desired.

Key properties enforced / verifiable:
- ``R`` has shape (k, d), dtype float32.
- ``R`` is L2 row-normalised after init (each row a unit direction), so the
  logits are comparable across categories.
- ``project(x)`` returns a simplex (non-negative, sums to 1).
- ``project_batch(X)`` is the vectorised form.
- ``update(X)`` performs an incremental rank-1-ish re-fit (optional).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["DimensionProjector"]


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax. Accepts (n, k) or (k,)."""
    z = np.asarray(logits, dtype=np.float64)
    single = z.ndim == 1
    if single:
        z = z[None, :]
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out = e / e.sum(axis=1, keepdims=True)
    return out[0] if single else out


@dataclass
class DimensionProjector:
    """Linear projection R^{k x d} with softmax readout.

    Parameters
    ----------
    k : int
        Output dimension (number of categories). 8 for trigram categories,
        5 for energy sub-spaces.
    d : int
        Input embedding dimension (e.g. 1024 for bge-m3, 384 for MiniLM).
    R : np.ndarray, optional
        Pre-existing (k, d) matrix. If omitted, initialised to identity-ish
        PCA basis lazily on first :meth:`fit`.
    temperature : float
        Softmax temperature; lower => sharper distribution.
    """

    k: int
    d: int
    R: np.ndarray | None = None
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.k <= 0 or self.d <= 0:
            raise ValueError("k and d must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.R is not None:
            R = np.asarray(self.R, dtype=np.float32)
            if R.shape != (self.k, self.d):
                raise ValueError(f"R must be ({self.k},{self.d}), got {R.shape}")
            self.R = R
        # else: lazily fit

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, n_iter: int = 0) -> "DimensionProjector":
        """Initialise R via PCA on a sample of embeddings.

        Computes the top-``k`` principal directions of ``X`` (SVD), L2
        row-normalises them, and stores as ``R``. This is the data-driven
        replacement for the legacy fixed 16-dim fold.

        Parameters
        ----------
        X : np.ndarray
            (n, d) matrix of real embeddings.
        n_iter : int
            Optional refinement iterations (kept for API symmetry; the PCA
            init is exact and usually needs no refinement).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(f"X must be (n, {self.d}), got {X.shape}")
        n = X.shape[0]

        # Centre the data.
        mean = X.mean(axis=0, keepdims=True)
        Xc = X - mean

        if n < self.k:
            # Not enough samples to span k principal directions; pad with
            # canonical basis vectors so R is well-defined.
            base = self._pca_or_pad(Xc, self.k)
        else:
            base = self._pca_or_pad(Xc, self.k)

        # L2 row-normalise so logit magnitudes are comparable.
        norms = np.linalg.norm(base, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.R = (base / norms).astype(np.float32)
        return self

    def _pca_or_pad(self, Xc: np.ndarray, k: int) -> np.ndarray:
        """Return the top-k principal directions, padding with basis vectors."""
        # Economy SVD: Xc = U S V^T; principal directions are rows of V^T.
        try:
            # full_matrices=False -> U (n,r), S (r,), Vt (r,d)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            got = Vt[:k]  # (min(k,r), d)
        except np.linalg.LinAlgError:
            got = np.zeros((0, self.d))

        if got.shape[0] < k:
            # Pad with canonical basis vectors (e_i) for the missing rows.
            pad = np.zeros((k - got.shape[0], self.d), dtype=np.float64)
            for i in range(pad.shape[0]):
                pad[i, i % self.d] = 1.0
            return np.vstack([got, pad])
        return got

    def update(self, X: np.ndarray, lr: float = 0.05) -> "DimensionProjector":
        """Incremental re-fit: blend current R toward the PCA of new X.

        Uses a convex combination ``R <- (1-lr) R + lr R_new`` so the projector
        adapts online without catastrophic forgetting. ``R_new`` is the PCA
        of the fresh batch (centred, normalised). Requires prior :meth:`fit`.
        """
        if self.R is None:
            return self.fit(X)
        if not 0.0 < lr <= 1.0:
            raise ValueError("lr must be in (0, 1]")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(f"X must be (n, {self.d})")

        tmp = DimensionProjector(self.k, self.d)
        tmp.fit(X)
        blended = ((1 - lr) * self.R + lr * tmp.R).astype(np.float32)
        norms = np.linalg.norm(blended, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.R = blended / norms
        return self

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def logits(self, x: np.ndarray) -> np.ndarray:
        """Logits z = R x / temperature. Accepts (d,) or (n, d)."""
        if self.R is None:
            raise RuntimeError("projector not fitted; call fit() first")
        X = np.asarray(x, dtype=np.float32)
        single = X.ndim == 1
        if single:
            X = X[None, :]
        if X.shape[1] != self.d:
            raise ValueError(f"input last dim must be {self.d}, got {X.shape}")
        z = (X @ self.R.T) / self.temperature
        return z[0] if single else z

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project x to a probability simplex over k categories."""
        return _softmax_rows(self.logits(x))

    def project_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorised project; returns (n, k)."""
        return _softmax_rows(self.logits(X))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def row_norms(self) -> np.ndarray:
        """L2 norm of each row of R (should be ~1 after fit)."""
        if self.R is None:
            raise RuntimeError("projector not fitted")
        return np.linalg.norm(self.R, axis=1)

    def rank(self) -> int:
        """Numerical rank of R (<= min(k, d))."""
        if self.R is None:
            raise RuntimeError("projector not fitted")
        return int(np.linalg.matrix_rank(self.R))
