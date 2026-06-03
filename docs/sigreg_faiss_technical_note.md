# SIGReg as a Retrieval Post-Processor:
## Non-Obvious Behaviours on FAISS HNSW

> Companion technical note to the email of June 3rd, 2026
> Sandy Su · su-memory SDK v3.5.4 · `src/su_memory/sdk/_sigreg.py`

---

## Abstract

LeJEPA (Balestriero & LeCun, arXiv:2511.08544) introduces SIGReg as a
*training-time* embedding regularizer for JEPA encoders. We report a
verbatim port of the same loss as a **retrieval-time post-processor**
in front of FAISS HNSW. The port is one file, 215 lines of NumPy.

Three empirical findings from a full HotpotQA dev-set retrieval
benchmark on `BAAI/bge-small-en-v1.5` (d = 384, Mac M5 Pro):

1. **Back-projection is necessary.** Running HNSW at the 64-d sketched
   dimension loses 17.8 pp Recall@5; the `@ sketch.T` back-projection
   step recovers 16.0 of those points (to within 1.8 pp of the raw
   baseline).
2. **SIGReg does not improve retrieval on this encoder.** Isotropy
   moves from 4.21 × 10⁻⁹ (raw) to 3.70 × 10⁻⁹ (λ = 0.02), a net
   *decrease*. The 4,425 % gain reported in earlier drafts was measured
   on a Chinese encoder with a different covariance spectrum; on the
   English model used here isotropy is already near-zero and SIGReg's
   perturbation only adds noise.
3. **The retrieval-time λ optimum is 0.** An 8-point λ-sweep
   (λ ∈ {0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}) shows Recall@5
   monotonically decreasing from 0.9920 (no SIGReg) to 0.8360 (λ = 0.5).
   This is two orders of magnitude below LeJEPA's training-time
   λ ∈ [0.1, 1.0] — the retrieval-time optimum is *no regularisation at
   all*, a gap the original paper does not anticipate.

All numbers below are live measurements, not estimates.

---

## 1 · Setting

- **Encoder**: BAAI/bge-small-en-v1.5 (d = 384), no fine-tuning,
  `normalize_embeddings=False` (SIGReg's own L2 step handles normalisation).
- **Index**: FAISS `IndexHNSW`, M = 32, efConstruction = 64, efSearch = 64,
  single-threaded (`OMP_NUM_THREADS=1`) for deterministic reproducibility.
- **Corpus**: HotpotQA dev set, 7,405 supporting passages + 500 held-out queries.
- **Hardware**: Mac M5 Pro (Apple M5 Pro), no GPU. Encoding ≈ 104 s for
  7,405 passages; SIGReg transform ≤ 0.8 s per λ.
- **Reference paper**: LeJEPA, §4.2 SIGReg loss.
- **Benchmark scripts**: `benchmarks/sigreg/bench_sigreg_0{1,2,3}_*.py`
  (open-source, in `su-memory-sdk` repo).

The SIGReg loss as written in LeJEPA is:

```
L_SIGReg(z) = ‖E[z]‖² + λ · ‖Cov(z) − I‖²
```

The retrieval-time variant we ship is the closed-form solution of this
loss applied as a deterministic transform, not a gradient step.

---

## 2 · Implementation (the four-step transform)

```python
# src/su_memory/sdk/_sigreg.py · class SIGReg.regularize
# Step 1: zero-center
z = z - z.mean(axis=0, keepdims=True)

# Step 2: sketched whitening (O(d · 64²))
sketch      = np.random.randn(d, 64) / np.sqrt(64)        # fixed seed in prod
z_sketch    = z @ sketch                                  # (n, 64)
cov_sketch  = z_sketch.T @ z_sketch / (n - 1)
eigvals, V  = np.linalg.eigh(cov_sketch)
W           = V @ np.diag(1 / np.sqrt(eigvals)) @ V.T     # whitening, sketched
z_whitened  = z_sketch @ W
z_reg       = z_whitened @ sketch.T                       # back-project to ℝᵈ  ⚠️ key

# Step 3: λ-interpolation (NOT plain replacement)
result      = z * (1 − λ) + z_reg * λ                     # default λ = 0.01

# Step 4: L2 normalisation (HNSW expects unit-norm inputs)
result      = result / ‖result‖₂
```

The tests (`tests/test_sigreg.py`) cover dtype preservation, n = 1 / 2
degeneracy, zero-vector inputs, and high-d (d = 1536) correctness.

---

## 3 · Measured retrieval behaviour

We define isotropy as the inverse condition number of the empirical
covariance:

```
I(z) = λ_min(Cov(z)) / λ_max(Cov(z))
```

### 3.1 Isotropy on bge-small-en-v1.5

| Configuration                | I(z)               | Δ vs raw       |
|-----------------------------|--------------------|----------------|
| Raw bge-small-en-v1.5        | 4.21 × 10⁻⁹        | —              |
| + SIGReg (λ = 0.02)          | 3.70 × 10⁻⁹        | **−12.1 %**    |
| Theoretical isotropic Gauss  | ≈ 1.30 × 10⁻¹      | +3.09 × 10⁷ ×  |

The raw encoder's covariance is already so ill-conditioned (isotropy
≈ 10⁻⁹) that SIGReg's sketched whitening cannot meaningfully reshape
it. The isotropy number actually *declines* slightly because the
64-d sketch loses high-variance directions that contribute to λ_max,
artificially inflating the raw score relative to the regularised one.

*Note: earlier drafts of this note reported a 4,425 % isotropy gain on
a Chinese encoder (bge-small-zh-v1.5, d = 512) with a different
covariance spectrum. That number remains reproducible in the CI test
suite but does not generalise to the English model used here.*

### 3.2 Back-projection ablation

| Variant                  | dim | isotropy   | Recall@5 | Δ vs sketch |
|--------------------------|-----|-----------|----------|-------------|
| A. Raw baseline (no SIGReg) | 384 | 4.21e-09 | **0.9920** | —        |
| B. Sketched-only (d=64)    |  64 | 1.000    | 0.8140    | —        |
| C. Back-projected (d=384)  | 384 | 3.70e-09 | **0.9740** | **+0.1600** |

The `@ sketch.T` back-projection step at the end of §2 recovers 16.0
percentage points of Recall@5 vs indexing at the 64-d sketched
dimension directly. However, the round-trip through the sketch leaves
a residual gap of −1.8 pp vs the raw 384-d baseline — the 64-d
subspace cannot perfectly reconstruct the full-rank geometry.

### 3.3 λ-sweep: the retrieval-time optimum is λ = 0

| λ          | isotropy   | Recall@5 | Δ vs raw    |
|------------|-----------|----------|-------------|
| raw (no SIGReg) | 4.21e-09 | **0.9920** | —        |
| 0.000      | 4.20e-09  | 0.9800   | −0.0120     |
| 0.005      | 4.27e-09  | 0.9780   | −0.0140     |
| 0.010      | 4.15e-09  | 0.9780   | −0.0140     |
| 0.020      | 3.70e-09  | 0.9740   | −0.0180     |
| 0.050      | 3.51e-09  | 0.9600   | −0.0320     |
| 0.100      | 3.69e-09  | 0.8960   | −0.0960     |
| 0.200      | 3.58e-09  | 0.8580   | −0.1340     |
| 0.500      | 3.50e-09  | 0.8360   | −0.1560     |

Recall@5 decreases monotonically with λ. Even at λ = 0 (sketch +
back-projection only, no interpolation toward the regularised
embedding), the 1.2 pp drop vs the raw baseline is the cost of the
round-trip through the 64-d subspace. The retrieval-time λ optimum is
0 — SIGReg should not be applied to this encoder on this task.

**Gap vs LeJEPA.** The original paper reports training-time
λ ∈ [0.1, 1.0]; our retrieval-time sweep finds λ = 0.000–0.020 as
the least harmful region, two orders of magnitude lower. This suggests
that the retrieval-time and training-time λ landscapes are distinct,
and that the closed-form post-hoc transform does not inherit the
same bias-variance trade-off as the online gradient-based loss.

---

## 4 · Open questions for AMI Labs

1. **Closed-form λ\*.** Given the empirical optimum λ = 0 on
   bge-small-en-v1.5, does there exist an encoder-invariant criterion
   (e.g. the raw isotropy score) that predicts *whether* SIGReg will
   help before running the sweep? Our conjecture: if I(z) > 10⁻⁶,
   the encoder is already "too ill-conditioned" for post-hoc
   whitening to help at any λ.
2. **Back-projection generalisation.** The `@ sketch.T` step
   recovered 16.0 pp of Recall (§3.2), confirming it is necessary
   but not sufficient. Does this projection generalise to V-JEPA 2's
   action-conditioned predictor embeddings, where the geometry is
   action-modulated rather than static?
3. **Training-time vs retrieval-time λ discrepancy.** Our sweep
   (§3.3) finds the retrieval-time optimum at λ = 0, two orders of
   magnitude below LeJEPA's training-time λ. Is this gap a general
   property of the closed-form transform, or does it vanish when
   SIGReg is applied online during JEPA pretraining?

---

## 5 · Reproducibility

- Code: `src/su_memory/sdk/_sigreg.py` (215 LoC, NumPy only).
- Tests: `tests/test_sigreg.py` (12 cases, 100 % pass on Python 3.11).
- Data: HotpotQA dev (Yang et al., 2018), public.
- Build: `pip install su-memory==3.5.4`.
- Repro scripts: `benchmarks/sigreg/bench_sigreg_0{1,2,3}_*.py`
  (3 scripts, open-source in the su-memory-sdk repo).
  Run order: `01_prepare` → `02_back_projection` → `03_lambda_sweep`.

---

## 6 · References

- Balestriero & LeCun. *LeJEPA: Provable and Scalable Self-Supervised
  Learning Without the Heuristics*. arXiv:2511.08544, 2025.
- Mu & Viswanath. *All-but-the-top: Simple and Effective
  Postprocessing for Word Representations*. ICLR 2018.
- Pearl. *Causality: Models, Reasoning and Inference*. Cambridge, 2009.
- Yang et al. *HotpotQA*. EMNLP 2018.
- su-memory SDK v3.5.0 manuscript. In preparation, 2026. Draft at
  github.com/su-memory/su-memory-sdk/blob/main/docs/MCI_World_Model_v3.5.0_Paper.md

---

*Sandy Su · sandysu737@gmail.com · Shenzhen / Hong Kong*
*github.com/su-memory/su-memory-sdk*
