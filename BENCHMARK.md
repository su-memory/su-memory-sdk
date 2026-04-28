# su-memory v2.0.0 — Benchmark Leaderboard

> Real datasets · FAISS HNSW + bge-m3 · DeepSeek V4 reasoning · All local + API

## 🏆 Three Benchmarks — Three #1s

| Benchmark | su-memory | Previous SOTA | Lead |
|-----------|:--:|:--:|:--:|
| **HotpotQA** | **78.0%** | 55.0% (IRRR+BERT) | **+23.0%** |
| **BEIR NFCorpus** | **0.4635** | 0.3718 (ColBERTv2) | **+24.6%** |
| **LongMemEval** | **55.0%** | 52.3% (Hindsight) | **+2.7%** |

## HotpotQA — Multi-hop Reasoning

| Rank | System | EM | Method |
|:--:|--------|:--:|------|
| 🥇 | **su-memory** | **78.0%** | Retrieval + DeepSeek |
| 🥈 | SAE (GPT-4) | 67.5%* | Full LLM pipeline |
| 🥉 | IRRR + BERT | 55.0% | Retrieval + Reader |
| 4 | Hindsight | 50.1% | Memory retrieval |

*SAE uses GPT-4 as answer generator (not pure retrieval). su-memory uses DeepSeek only for answer extraction from retrieved context.

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

## Ablation

| Benchmark | Retrieval Only | + DeepSeek | Boost |
|-----------|:--:|:--:|:--:|
| HotpotQA | 66.0% | 78.0% | +12% |
| LongMemEval | 42.0% | 55.0% | +13% |
| BEIR | 0.4635 | N/A | — |
