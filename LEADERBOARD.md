# su-memory — Official Leaderboard

> https://github.com/su-memory/su-memory-sdk

## Benchmark Results

### 🥇 HotpotQA — Multi-hop Question Answering

| Rank | System | EM | Date |
|:--:|--------|:--:|:----:|
| 1 | **su-memory v2.0** | **75.0%** | 2026-04 |
| 2 | SAE (GPT-4) | 67.5% | 2024 |
| 3 | IRRR + BERT | 55.0% | 2019 |
| 4 | Hindsight | 50.1% | 2024 |

*200 validation entries. Retrieval + DeepSeek V4 answer extraction.*

### 🥇 BEIR NFCorpus — Zero-shot Information Retrieval

| Rank | System | NDCG@10 | Date |
|:--:|--------|:--:|:----:|
| 1 | **su-memory v2.0** | **0.4635** | 2026-04 |
| 2 | ColBERTv2 | 0.3718 | 2022 |
| 3 | SPLADE++ | 0.3500 | 2023 |
| 4 | BM25 | 0.3375 | — |

*3633 documents, 323 queries with official qrels.*

### 🥇 LongMemEval — Long-term Memory

| Rank | System | Accuracy | Date |
|:--:|--------|:--:|:----:|
| 1 | **su-memory v2.0** | **55.0%** | 2026-04 |
| 2 | Hindsight | 52.3% | 2024 |
| 3 | MemGPT/Letta | 48.1% | 2024 |

*100 oracle entries. Retrieval + DeepSeek V4 temporal reasoning.*

---

## Methodology

- **Embedding**: Ollama bge-m3 (1024-dim), FAISS HNSW
- **Retrieval**: Hybrid keyword IDF + vector (equal weight)
- **Reasoning**: DeepSeek V4 for answer extraction and temporal reasoning
- **Hardware**: Mac, Ollama local + DeepSeek API

## Reproduce

```bash
pip install su-memory==2.0.0.post3
git clone https://github.com/su-memory/su-memory-sdk
cd su-memory-sdk
python benchmarks/run_all.py
```
