# su-memory — Official Leaderboard

> https://github.com/su-memory/su-memory-sdk

## ⚠️ 诚实声明（必读）

本文件曾包含 v2.0 时期的 **HotpotQA 75.0% EM / BEIR NFCorpus 0.4635 / LongMemEval 55.0%** 等成绩。
经核实，上述数字来自**合成数据自测**，不可复现，**已全部删除**。

> **`BENCHMARK.md` 是本仓库性能数字的唯一真值源。** 所有数字必须可复现
> （`python benchmarks/real_microbench.py` / `python benchmarks/hotpotqa_full_eval.py`）。
> 如本文件与 `BENCHMARK.md` 或 `README.md` 不一致，以后两者为准。

---

## Benchmark Results

### 🥇 HotpotQA — Multi-hop Question Answering

> 官方 validation set，200 题，全 hard level，**标准严格 EM 口径**
> （reader 抽取 span == gold answer）。
> 复现：`python benchmarks/hotpotqa_full_eval.py --top-k 7 --api`（DeepSeek）
> 或 `--omlx qwen3-32b`（OMLX）或默认（本地 7B）。

| Rank | System | 标准 EM | F1 | Date |
|:--:|--------|:--:|:--:|:----:|
| 1 | Hindsight (SOTA) | 70.83% | — | 2024 |
| 2 | **su-memory v4.0.1 (DeepSeek-V3 + CausalDAG 桥接)** | **62.5%** | **75.4%** | 2026-06 |
| 3 | **su-memory v4.0.1 (OMLX Qwen3-32B, Metal GPU)** | **61.5%** | **73.8%** | 2026-06 |
| 4 | IRRR + BERT | 55.0% | — | 2019 |
| 5 | **su-memory v4.0.1 (本地 7B MLX reader)** | **48.0%** | **58.6%** | 2026-06 |
| 6 | DFGN (pure retrieval) | 48.2% | — | — |

**关键结论（均已复现）**：
- DeepSeek-V3 + CausalDAG 桥接（标准 EM 62.5%）**真实超越 DFGN(48.2%) 与本地 7B(48%) 约 14 个百分点**。
- comparison 题 EM **73.5%，已超 Hindsight SOTA(70.83%)**。
- CausalDAG 罕见实体桥接发现率 **90%**（vs 传统 title 匹配 44%）。
- 本地 32B Metal GPU 无需 API：标准 EM 61.5%，宽松 EM 77.5%。
- 整体 62.5% 距 Hindsight 70.83% 仍差约 8 点；bridge 题 60.2% 拖累整体，
  真实突破需专属多跳模型微调。

### BEIR NFCorpus / LongMemEval

> **暂无可复现数据。** 旧版曾列 BEIR 0.4635 / LongMemEval 55.0%，经核实为
> 合成数据自测，已移除。待建立可复现评测流程后补回。

---

## Methodology

- **Embedding**: sentence-transformers bge-m3 (1024-dim)，FAISS HNSW
- **Retrieval**: 混合检索（关键词 IDF + 向量），三路融合 MultiHopReader（direct + title-bridge + entity-bridge）
- **Reasoning**: DeepSeek-V3 / OMLX Qwen3-32B / 本地 7B MLX（三档 reader）
- **桥接增强**: CausalDAG（IDF 加权罕见实体桥接，DF≤3）
- **Hardware**: Apple M5 Pro（Metal GPU 推理 OMLX）；DeepSeek API（云端）

## Reproduce

```bash
pip install su-memory
git clone https://github.com/su-memory/su-memory-sdk
cd su-memory-sdk

# 性能微基准（内存/延迟，离线可跑）
python benchmarks/real_microbench.py --no-lite-pro

# HotpotQA 评测（需对应 reader）
python benchmarks/hotpotqa_full_eval.py --top-k 7 --api       # DeepSeek
python benchmarks/hotpotqa_full_eval.py --top-k 7 --omlx qwen3-32b  # OMLX 32B
python benchmarks/hotpotqa_full_eval.py --top-k 7             # 本地 7B
```
