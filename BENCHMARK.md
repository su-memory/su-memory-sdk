# su-memory v3.3.0 — 真实性能基准（honest benchmark）

> **重要说明**：本文件所有数字均来自 `benchmarks/real_microbench.py`，可在本机完整复现（误差 <15%）。
> 此前的版本曾出现「7 维全 1.000 / HotpotQA 78% 超 SOTA / 插入 97K/s / 内存 0.33MB」等宣称，
> 经核实为**合成数据自测 + 硬编码成绩 + 测试泄漏**，不可复现，已全部删除。

---

## 测试环境

| 项 | 值 |
|----|----|
| Python | 3.13.12 (conda-forge) |
| 平台 | macOS 26.5.1 arm64 (Apple Silicon) |
| 依赖 | faiss-cpu 1.14 · sentence-transformers · numpy 2.4 |
| 测试日期 | 2026-06-28 |
| 复现命令 | `python benchmarks/real_microbench.py` |

---

## ⚠️ 能力边界声明（必读）

| 产品 | 检索方式 | 语义能力 |
|------|----------|----------|
| **SuMemoryLite** | N-gram TF-IDF 倒排索引 | **关键词/语素检索，非真向量语义** |
| **SuMemoryLitePro** | FAISS HNSW + sentence-transformers | **真向量语义检索** |

> Lite 与 LitePro 能力边界不同，**性能数字禁止混报**。
> 下表「语义召回」一栏如实反映 Lite 在零语素重叠改写下**无法命中**（0%），而非宣称的高分。

---

## Lite 性能（3 轮中位数，tracemalloc 实测）

### 插入吞吐 / 内存

| 规模 | 插入吞吐 | Peak 内存 |
|:----:|:--------:|:---------:|
| 100 | ~10.7K/s | 0.98 MB |
| 1K | ~9.1K/s | 6.92 MB |
| 5K | ~8.3K/s | 45.56 MB |
| 10K | ~7.4K/s | **79.81 MB** |

> 真实瓶颈：5K 已达 45.56MB，逼近文档历史宣称的 `<50MB`；10K 达 79.81MB，**已超出**。
> 解决方案：激活已有 `TieredStorage` 冷热分层；超大库切分片索引（`_PARTITION_DF_THRESHOLD` 雏形已存在）。

### 查询延迟（每次清缓存，消除缓存干扰）

| 规模 | P50 | P95 |
|:----:|:---:|:---:|
| 100 | 0.032 ms | 0.106 ms |
| 1K | 0.122 ms | 0.235 ms |
| 5K | 0.271 ms | 0.439 ms |
| 10K | 0.493 ms | 0.785 ms |

> 查询延迟是 Lite 的真实优势：10K 规模 P95 仍 <1ms（亚毫秒级）。

### 语义召回边界（Lite，N-gram TF-IDF）

| 改写类型 | 命中率 |
|----------|:------:|
| 共享中文语素的改写（如「去年赚了多少」→「营收」） | **100.0%** |
| 零语素重叠的改写（如「该名新成员何时到岗」→「张三」） | **0.0%** |

> 结论：Lite 能命中共享语素的改写，但对零重叠的纯语义改写**完全无法命中**。
> 这是「Lite ≠ 真语义」的实证依据。需要真语义请使用 **LitePro**。

---

## LitePro 性能（FAISS + sentence-transformers）

> 需可下载 embedding 模型；如本机不可联网或模型未安装，运行 `--no-lite-pro` 跳过。
> 运行 `python benchmarks/real_microbench.py` 获取本机实测值。

---

## 历史宣称已删除项（不可复现）

以下内容曾在旧版 BENCHMARK/README 出现，经核实为造假或不可复现，已删除：

- ❌ 「Memory Engine SOTA 7 维全 1.000，OVERALL 0.986 A+」——合成数据自测 + 测试泄漏
- ❌ 「HotpotQA 78.0% 超 IRRR+BERT(55%)」——`_generate_benchmark_dataset` 合成题对比真实榜单，不成立
- ❌ 「BEIR NFCorpus 0.4635 超 ColBERTv2」——同上
- ❌ 「LongMemEval 55.0% 超 Hindsight」——同上
- ❌ 「插入 97K/s / 83K/s」——两份"官方"数字自相矛盾，实测 ~9K/s
- ❌ 「1K 内存 0.33MB」——实测 6.92MB
- ❌ 「7 维 #1 SOTA 对比表（Hindsight/Mem0/Zep/...）」——su-memory 同代码库此前自测 overall 仅 65%，全面 LOSE

> 证据：`benchmarks/hindsight_comparison_report.json`（2026-04-23 真跑）记录 su-memory
> 多跳推理 30%、单跳 70%、overall 65%，**全面低于 Hindsight(91.4%)**，且 embedding 为 `hash-based`。

---

## 复现

```bash
# 完整（含 LitePro，需联网下载模型）
python benchmarks/real_microbench.py

# 仅 Lite（快速，无外部依赖）
python benchmarks/real_microbench.py --no-lite-pro
```

输出写入 `benchmarks/results/real_microbench_{timestamp}.json`。
