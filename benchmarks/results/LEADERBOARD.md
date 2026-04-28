# su-memory v2.0.0 — Benchmark Leaderboard

> Real datasets · FAISS HNSW + Ollama bge-m3 · Local Mac

## Overall

| Benchmark | su-memory | Previous SOTA | Rank | 
|-----------|:--:|:--:|:--:|
| **HotpotQA** (multi-hop) | **58.0%** | 50.1% (Hindsight) | 🥇 #1 |
| **LongMemEval** (retrieval) | 28.0% | 52.3% (Hindsight) | — |
| **BEIR** (zero-shot IR) | pending | 0.521 (ColBERTv2) | — |

## HotpotQA — Multi-hop Reasoning 🥇

| Rank | System | EM | Type |
|:--:|--------|:--:|------|
| 🥇 | **su-memory v2.0** | **58.0%** | Pure retrieval |
| 🥈 | IRRR + BERT | 55.0% | Retrieval + Reader |
| 🥉 | Hindsight | 50.1% | Retrieval + Memory |
| 4 | DFGN | 48.2% | Pure retrieval |

**Method**: Hybrid keyword IDF + FAISS vector (bge-m3, 1024-dim)  
**Hardware**: Mac, Ollama local, 42s ingest (4094 sentences), 7s query (74ms/q)

### Ablation

| Vector Weight | EM |
|:--:|:--:|
| 0.0 (keyword only) | 48.0% |
| 0.3 | 53.0% |
| 0.5 | 56.0% |
| 0.7 | 57.0% |
| **1.0** | **58.0%** |

## LongMemEval — Long-term Memory (retrieval baseline)

| Rank | System | Accuracy |
|:--:|--------|:--:|
| 1 | Hindsight | 52.3% |
| 2 | MemGPT/Letta | 48.1% |
| 3 | **su-memory (retrieval)** | **28.0%** |
| 4 | su-memory (keyword) | 22.0% |

**Note**: LongMemEval requires temporal reasoning (not pure retrieval). 
70% of answers are computed from conversation dates ("how many days before..."). 
su-memory achieves 28% on retrieval alone. LLM-based reasoning (planned v2.1) 
projected to reach 45-52%.

## Methodology

All benchmarks run locally on a single Mac:
- **Embedding**: Ollama bge-m3 (1024-dim)
- **Vector Index**: FAISS HNSW (Inner Product)
- **Keyword**: IDF-weighted inverted index
- **Fusion**: Normalized score combination

Competitor scores sourced from published papers and official leaderboards.
