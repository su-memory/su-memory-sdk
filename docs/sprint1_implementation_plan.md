# su-memory-sdk v2.6.0 — Sprint 1 实施计划

**Sprint**: 测试补全
**周期**: 1周 | **约束**: Zero new feature，仅加固
**基线**: 756 tests (34文件) → **目标**: +170 tests (5个新/扩展文件)

---

## Task 1: lite_pro 核心测试 (`test_lite_pro_comprehensive.py`)

**优先级**: 🔴 P0 | **目标**: 6 → 80+ tests

| ID | 范围 | 预期 tests |
|:---|------|:---:|
| T1.1 | 初始化参数矩阵 (enable_*) | 8 |
| T1.2 | add 单条记忆 (正常/边界/异常) | 10 |
| T1.3 | add_batch 批量写入 | 5 |
| T1.4 | query 向量检索 (3种fusion模式) | 10 |
| T1.5 | query_multihop 多跳推理 | 8 |
| T1.6 | query_multihop_spacetime | 5 |
| T1.7 | FAISS 索引生命周期 | 8 |
| T1.8 | embedding 懒加载/回退 | 6 |
| T1.9 | MemoryGraph CRUD | 6 |
| T1.10 | predict / explain | 6 |
| T1.11 | get_stats / 统计信息 | 4 |
| T1.12 | 记忆生命周期 (forget/decay) | 4 |

---

## Task 2: FAISS 索引专项 (`test_faiss_index.py`)

**优先级**: 🔴 P0 | **目标**: 0 → 25+ tests

| ID | 范围 | 预期 tests |
|:---|------|:---:|
| T2.1 | 索引创建参数 (m, efConstruction) | 5 |
| T2.2 | add + search 精确/近似 | 6 |
| T2.3 | 量化压缩 (FP32/FP16/INT8/Binary) | 5 |
| T2.4 | 持久化 save→load→search | 4 |
| T2.5 | 增量索引 | 3 |
| T2.6 | 空索引/维度不匹配 | 2 |

---

## Task 3: VectorGraphRAG 多跳推理 (`test_multihop_reasoning.py` 扩展)

**优先级**: 🔴 P0 | **目标**: 4 → 30+ tests

| ID | 范围 | 预期 tests |
|:---|------|:---:|
| T3.1 | 单跳/两跳/三跳推理 | 6 |
| T3.2 | 4种因果类型 | 5 |
| T3.3 | 3种fusion模式对比 | 5 |
| T3.4 | max_hops/min_score 边界 | 5 |
| T3.5 | 环形图/孤立节点 | 4 |
| T3.6 | 嵌入不可用回退 | 3 |
| T3.7 | 大规模图谱性能 | 2 |

---

## Task 4: 并发安全 (`test_concurrency.py`)

**优先级**: 🟡 P1 | **目标**: 0 → 15+ tests

| ID | 范围 | 预期 tests |
|:---|------|:---:|
| T4.1 | 多线程并发 add | 4 |
| T4.2 | 读写混合并发 | 3 |
| T4.3 | FAISS 并发 search | 3 |
| T4.4 | FAISS 并发 add+train | 2 |
| T4.5 | 存储并发 (SQLite WAL) | 3 |

---

## Task 5: 降级路径 (`test_fallback.py`)

**优先级**: 🟡 P1 | **目标**: 0 → 20+ tests

| ID | 范围 | 预期 tests |
|:---|------|:---:|
| T5.1 | FAISS→线性检索 | 3 |
| T5.2 | Ollama→MiniMax→sentence-transformers | 4 |
| T5.3 | Embedding→TF-IDF (全部不可用) | 3 |
| T5.4 | 图谱→纯向量 | 3 |
| T5.5 | 时空索引→时序衰减 | 3 |
| T5.6 | 存储→内存 | 4 |

---

## 📅 执行顺序

```
Day 1-2: T1 (lite_pro) — 最大缺口，最核心
Day 3:   T2 (FAISS) + T3 (多跳) — 并行
Day 4:   T4 (并发) + T5 (降级) — 并行
Day 5:   验证 + 报告
```
