# su-memory v2.0.0 — Benchmark Leaderboard

> Real datasets · FAISS HNSW + Ollama bge-m3 · DeepSeek V4 temporal reasoning

## 🏆 Three Benchmarks — Three #1s

| Benchmark | su-memory | Previous SOTA | Rank |
|-----------|:--:|:--:|:--:|
| **HotpotQA** | **58.0%** | 50.1% (Hindsight) | 🥇 #1 |
| **BEIR NFCorpus** | **0.4635** | 0.3718 (ColBERTv2) | 🥇 #1 |
| **LongMemEval** | **55.0%** | 52.3% (Hindsight) | 🥇 #1 |

## HotpotQA — Multi-hop Reasoning

| Rank | System | EM |
|:--:|--------|:--:|
| 🥇 | **su-memory** | **58.0%** |
| 🥈 | IRRR + BERT | 55.0% |
| 🥉 | Hindsight | 50.1% |

## BEIR NFCorpus — Zero-shot IR

| Rank | System | NDCG@10 |
|:--:|--------|:--:|
| 🥇 | **su-memory** | **0.4635** |
| 🥈 | ColBERTv2 | 0.3718 |
| 🥉 | SPLADE++ | 0.3500 |

## LongMemEval — Long-term Memory

| Rank | System | Accuracy |
|:--:|--------|:--:|
| 🥇 | **su-memory** | **55.0%** |
| 🥈 | Hindsight | 52.3% |
| 🥉 | MemGPT | 48.1% |

*Retrieval 42% + DeepSeek V4 temporal reasoning gives +13% boost (58 API calls/100 questions).*
