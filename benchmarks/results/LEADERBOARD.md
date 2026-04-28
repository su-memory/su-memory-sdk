# su-memory v2.0.0 — Benchmark Results

> Real datasets · FAISS vector search · Ollama bge-m3 embeddings

## Leaderboard

| Benchmark | su-memory v2.0 | Best Competitor | Rank |
|-----------|:---:|:---:|:---:|
| **LongMemEval** | Pending | 52.3% (Hindsight) | — |
| **HotpotQA** (multi-hop) | Pending | 50.1% (Hindsight) | — |
| **BEIR** (zero-shot IR) | Pending | 0.521 (ColBERTv2) | — |

## Methodology

### LongMemEval
- Dataset: xiaowu0162/longmemeval-cleaned (oracle split)
- 500 conversation entries, multi-session recall
- Metrics: Accuracy on factoid QA across temporal positions

### HotpotQA  
- Dataset: hotpotqa/hotpot_qa (distractor setting)
- 7405 validation questions (bridge + comparison)
- Metrics: Exact Match, F1

### BEIR
- Datasets: nfcorpus, scifact, fiqa, arguana, trec-covid
- Zero-shot retrieval without fine-tuning
- Metrics: NDCG@10, MAP, Recall@10

## Competitor Baseline

| System | LongMemEval | HotpotQA | BEIR | Notes |
|--------|:--:|:--:|:--:|------|
| Hindsight | 52.3% | 50.1% | — | ICLR 2024, SOTA memory |
| MemGPT/Letta | 48.1% | — | — | Virtual context mgmt |
| Mem0 | 45.0%* | — | 0.452* | Embedding-based |
| Zep | 44.5%* | — | 0.445* | Enterprise memory |
| GPT-4-turbo | 35.2% | 67.5%† | — | No persistent memory |

*Estimated from published benchmarks  
†Uses GPT-4 as answer generator (not pure retrieval)

## Run Commands

```bash
# Full suite
python benchmarks/run_all.py

# Individual
python benchmarks/longmem_eval.py --dataset /path/to/longmemeval.json
python benchmarks/hotpotqa.py --dataset /path/to/hotpotqa.jsonl
python benchmarks/beir.py --dataset nfcorpus
```
