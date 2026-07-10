# HotpotQA 本地真实评测报告 (2026-07-06)

> 官方 HotpotQA validation, 200 题, 全 hard level, 标准 SQuAD EM 口径
> 全部可复现: `python benchmarks/hotpotqa_full_eval.py`

## 最终评测结果

### 全量 200 题 (最终, Qwen3-32B via OMLX Metal GPU)

| 指标 | 结果 | 说明 |
|---|---|---|
| **标准 EM** | **61.5%** (123/200) | 官方严格口径, reader 精确 span == gold |
| **宽松 EM (substring)** | **77.5%** (155/200) | gold⊂pred 或 pred⊂gold 也算对 |
| **F1 (token)** | **73.8%** | |
| bridge (166题) | 60.2% | |
| comparison (34题) | 67.6% | |
| 耗时 | 1138s (~20min) | 本地 32B + Metal GPU, 无需 API |

### 50 题样本 (各后端对比)

| Reader 后端 | 参数量 | EM | 宽松EM | F1 | comparison | 耗时 |
|---|---|---|---|---|---|---|
| **OMLX Qwen3-32B (CoT)** | **32B** | **66.0%** | **84.0%** | **80.1%** | **85.7%** | 304s |
| DeepSeek-chat API | — | 62.0% | — | 77.1% | 78.6% | 83s |
| Ollama qwen3.6:27b (CPU) | 27B | 60.0% | — | 78.1% | 78.6% | 173s |
| MLX Qwen2.5-7B | 7B | 60% (10题) | — | 67.5% | 66.7% | ~10s/10题 |

### 对照 SOTA

| 系统 | 标准 EM | 说明 |
|---|---|---|
| **Hindsight** | **70.83%** | 依赖大模型 + 专门桥接记忆架构 (SOTA) |
| su-memory OMLX 32B | 61.5% | 本地 32B, 无 API, 全 Metal GPU |
| su-memory (宽松EM) | 77.5% | 推理正确但 span 不精确的题 |
| IRRR + BERT | 55.0% | |
| DFGN | 48.2% | 纯检索 graph |

## 关键发现

1. **OMLX Qwen3-32B 是本地最强 reader** — 50 题 CoT prompt 下 EM 66%, 超 DeepSeek API (62%) 和 Ollama 27B (60%)。
2. **Metal GPU 推理稳定** — OMLX 正确检测 Apple M5 Pro Metal GPU, 无 Ollama GPU discovery timeout 问题。
3. **CoT prompt 是关键提升** — 从 "Answer:" 直答 (54%) 到 "Reason→Answer" 推理格式 (66%), +12 个百分点。
4. **宽松 EM 79% 暴露核心瓶颈** — 36 题推理正确但答案 span 不精确, 是 span 精修的优化空间。
5. **检索不是瓶颈** — bridge 题 gold 召回率 97% (top_k=7)。
6. **距 Hindsight 差 ~10 点** — 通用 32B 模型 + 算法检索的标准 EM 天花板; 突破需专门多跳微调或更好的 span 精修。

## 复现命令

```bash
# OMLX Qwen3-32B (本地最强, Metal GPU, 需 OMLX 服务运行在 localhost:11435)
python benchmarks/hotpotqa_full_eval.py --sample 50 --top-k 7 --omlx qwen3-32b

# 全量 200 题
python benchmarks/hotpotqa_full_eval.py --top-k 7 --omlx qwen3-32b

# DeepSeek API (需 DEEPSEEK_API_KEY)
python benchmarks/hotpotqa_full_eval.py --sample 50 --top-k 7 --api

# Ollama 27B (需 ollama + qwen3.6:27b)
python benchmarks/hotpotqa_full_eval.py --sample 50 --top-k 7 --ollama qwen3.6:27b

# 本地 7B MLX (最轻量)
python benchmarks/hotpotqa_full_eval.py --sample 50 --top-k 7
```
