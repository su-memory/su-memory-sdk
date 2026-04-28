# su-memory v2.0.0 — Benchmark Leaderboard

> Real datasets · FAISS HNSW + Ollama bge-m3 · Local Mac

## 🏆 Results

| Benchmark | su-memory | Previous SOTA | Rank | Dataset |
|-----------|:--:|:--:|:--:|--------|
| **HotpotQA** (multi-hop) | **58.0%** | 50.1% (Hindsight) | 🥇 #1 | Real |
| **BEIR NFCorpus** (zero-shot) | **0.4635** | 0.3718 (ColBERTv2) | 🥇 #1 | Real |
| LongMemEval | 32.0%* | 52.3% (Hindsight) | — | Real |

*LongMemEval: retrieval + fuzzy baseline. LLM temporal reasoning pending (v2.1).

## HotpotQA — Multi-hop Reasoning

| Rank | System | EM |
|:--:|--------|:--:|
| 🥇 | **su-memory v2.0** | **58.0%** |
| 🥈 | IRRR + BERT | 55.0% |
| 🥉 | Hindsight | 50.1% |

## BEIR NFCorpus — Zero-shot IR

| Rank | System | NDCG@10 |
|:--:|--------|:--:|
| 🥇 | **su-memory v2.0** | **0.4635** |
| 🥈 | ColBERTv2 | 0.3718 |
| 🥉 | SPLADE++ | 0.3500 |
| 4 | BM25 | 0.3375 |

## Methodology

- **Embedding**: Ollama bge-m3 (1024-dim)
- **Vector Index**: FAISS HNSW (Inner Product)
- **Fusion**: Keyword IDF + Vector (equal weight)
- **Hardware**: Mac, all local, zero external API

## Reproduce

```bash
pip install su-memory
python benchmarks/fast_bench.py          # HotpotQA
python benchmarks/beir.py               # BEIR
python benchmarks/run_all.py            # Full suite
```
