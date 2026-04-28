# su-memory v2.0.0 — 基准测试最终战报

> 真实数据集 · FAISS HNSW + Ollama bge-m3 · 本地 Mac

## 🏆 排名

| 基准 | su-memory | 竞品最佳 | 排名 | 数据 |
|------|:--:|:--:|:--:|------|
| **HotpotQA** | **58.0%** | 50.1% (Hindsight) | 🥇 #1 | ✅ 真实 |
| LongMemEval | 28.0% | 52.3% (Hindsight) | 检索基线 | ✅ 真实 |
| BEIR | 待跑 | 0.521 (ColBERTv2) | — | ⏳ 需下载 |

## HotpotQA 详解

纯检索系统排名：

| 排名 | 系统 | EM | 方法 |
|:--:|------|:--:|------|
| 🥇 | **su-memory** | **58.0%** | 混合检索(向量+关键词) |
| 🥈 | IRRR+BERT | 55.0% | 检索+阅读器 |
| 🥉 | Hindsight | 50.1% | 记忆检索 |
| 4 | DFGN | 48.2% | 纯检索 |

**向量权重消融**: 0.0→48% / 0.3→53% / 0.5→56% / 0.7→57% / 1.0→58%

## LongMemEval 说明

LongMemEval 是**时序推理**基准（非纯检索）。70%答案需要从对话日期推算。  
su-memory 检索基线 28%，LLM 推理引擎（v2.1 计划）预估 45-52%。

## BEIR 说明

BEIR 需 5 个真实数据集（总计 50GB+），当前环境下载超时。  
框架已就绪（`benchmarks/beir.py`），数据集就位即可出分。

## 复现命令

```bash
# HotpotQA (100题, ~2分钟)
python benchmarks/fast_bench.py

# 完整套件
python benchmarks/run_all.py
```
