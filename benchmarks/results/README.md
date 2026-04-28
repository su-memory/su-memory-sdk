# su-memory v2.0.0 Benchmark Results

> **Note**: These are initial micro-benchmark results using synthetic data.
> Full benchmark runs require real datasets (LongMemEval/HotpotQA/BEIR) from HuggingFace.
> FAISS vector index disabled — pure keyword retrieval only.

## Quick Results

| Benchmark | su-memory | Best Competitor | Status |
|-----------|-----------|-----------------|--------|
| LongMemEval Acc | Pending* | 52.3% (Hindsight) | Framework ready |
| HotpotQA Bridge EM | Pending* | 50.1% (Hindsight) | Framework ready |
| BEIR NDCG@5 | Pending* | 0.499 (SPLADE++) | Framework ready |

*Requires real datasets + FAISS vector index for meaningful comparison.

## Synthetic Data Test (keyword-only, no FAISS)

```
LongMemEval:  0.0%   — synthetic parameter names not matchable (expected)
HotpotQA:    40.0%   — basic keyword retrieval on bridge questions
BEIR:        0.022   — random relevance on shuffled documents (expected)
```

## Next Steps for Production Scores

1. Install FAISS: `pip install faiss-cpu`
2. Get LongMemEval dataset: `huggingface-cli download xiaowu0162/longmemeval-cleaned`
3. Run with full vector search: `python benchmarks/run_all.py --with-vector`
4. Compare against Hindsight baseline

**Expected scores with FAISS + real datasets**: targeting 55-60% on LongMemEval, 55-65% EM on HotpotQA, 0.45-0.50 NDCG@10 on BEIR.
