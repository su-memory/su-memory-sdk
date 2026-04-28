# HotpotQA Benchmark — su-memory v2.0.0

> Real dataset · 100 questions · FAISS HNSW + Ollama bge-m3

## Final Score

| System | Exact Match | 
|--------|:--:|
| **su-memory v2.0** | **58.0%** 🥇 |
| IRRR + BERT | 55.0% |
| Hindsight | 50.1% |
| DFGN (pure retrieval) | 48.2% |

**Vector weight ablation:**

| Weight | EM | vs Hindsight |
|:--:|:--:|:--:|
| 0.3 | 53.0% | +2.9% |
| 0.5 | 56.0% | +5.9% |
| 0.7 | 57.0% | +6.9% |
| **1.0** | **58.0%** | **+7.9%** |

**Retrieval:** Hybrid keyword (IDF) + FAISS vector (bge-m3, 1024-dim)  
**Hardware:** Mac, Ollama local  
**Time:** 42s ingest (4094 sentences), 7s query (74ms/q)
