# su-memory-sdk v2.6.0 稳定性 — Sprint 详细实施计划

> **代码库**: `su-memory-sdk`（纯净记忆引擎，不含智能体层）
>
> **基线**: v2.5.5 | **目标**: 测试 756→400+ / 覆盖率 60%→90% / P99≤50ms / 降级100% / 文档100%
>
> **约束**: Zero new feature，仅加固

---

## 📊 Sprint 0: 基线审计（当前状态）

| 指标 | 值 |
|------|:---:|
| 测试总数 | **756** (34 文件) |
| 源码 .py 文件 | 87 |
| 核心引擎行数 | ~15,000 |
| 最大单文件 | `lite_pro.py` 3,706 行 — 仅 6 tests |
| 最弱测试模块 | `lite_pro.py` (6) / `multihop_reasoning` (4) / `persistence` (5) / `ollama_embedding` (4) |
| 并发测试 | **0 tests** |
| 降级路径测试 | **0 tests** |
| 10万记忆压测 | **无** |
| API文档自动生成 | **无** |
| CI性能门禁 | **无** |

---

## Sprint 1: 测试补全（第1周）

### 1.1 SuMemoryLitePro 核心测试 (目标: 6 → 80+)

**文件**: `tests/test_lite_pro.py` + 新建 `tests/test_lite_pro_comprehensive.py`

| ID | 测试项 | 内容 |
|:---|------|------|
| T1.1 | 初始化参数矩阵 | 测试 9 个 enable_* 参数的 2^9 组合中关键路径 |
| T1.2 | add 单条记忆 | 正常/空字符串/特殊字符/超长文本/emoji/中英混合 |
| T1.3 | add_batch 批量写入 | 10/100/1000 条吞吐基准 |
| T1.4 | query 向量检索 | 精确匹配/语义近似/无结果/空查询 |
| T1.5 | query 三种 fusion 模式 | vector_first / hybrid / graph_first 对比 |
| T1.6 | query_multihop 多跳推理 | 2-hop / 3-hop / max_hops=0 边界 / 环形引用 |
| T1.7 | query_multihop_spacetime | 时间范围约束 + 多跳融合 |
| T1.8 | FAISS 索引生命周期 | 创建→持久化→恢复→增量更新→删除 |
| T1.9 | embedding 懒加载 | Ollama不可用→自动回退 MiniMax→sentence-transformers |
| T1.10 | 记忆图谱 CRUD | add_node / add_edge / 因果类型验证 |
| T1.11 | 能量推断准确率 | 已知标签对比（≥85%） |
| T1.12 | predict 时序预测 | 趋势方向正确性 / 空数据 / 单条数据 |
| T1.13 | explain_query 可解释性 | 输出格式完整性 / 置信度范围 |

### 1.2 FAISS HNSW 索引专项 (目标: 0 → 25+)

**文件**: 新建 `tests/test_faiss_index.py`

| ID | 测试项 | 内容 |
|:---|------|------|
| T2.1 | 索引创建参数 | m=16/32/64, efConstruction=40/64/128 |
| T2.2 | 向量添加与搜索 | 精确搜索 top_k=1/5/10 |
| T2.3 | 量化压缩对比 | FP32/FP16/INT8/Binary 4种模式精度对比 |
| T2.4 | 持久化与恢复 | save→load→search 一致性 |
| T2.5 | 增量索引 | 分批 add 后 search 一致性 |
| T2.6 | 空索引行为 | 搜索空索引 / 添加0向量 |
| T2.7 | 维度不匹配 | 添加错误维度向量时的异常 |
| T2.8 | 大规模索引 | 10,000 条写入+搜索 (pytest.mark.slow) |

### 1.3 VectorGraphRAG 多跳推理专项 (目标: 4 → 30+)

**文件**: 扩展现有 `tests/test_multihop_reasoning.py`

| ID | 测试项 | 内容 |
|:---|------|------|
| T3.1 | 单跳推理 | 1-hop 精确路径 |
| T3.2 | 两跳推理 | A→B→C 路径完整性 |
| T3.3 | 三跳推理 | 3-hop 衰减系数 0.85 验证 |
| T3.4 | 四种因果类型 | cause/condition/result/sequence 分别验证 |
| T3.5 | 三种 fusion 模式 | vector_first/hybrid/graph_first 结果差异 |
| T3.6 | max_hops 边界 | 0/1/5/None |
| T3.7 | min_score 阈值 | 不同阈值下的结果数量 |
| T3.8 | 空图谱行为 | 无边的孤立节点查询 |
| T3.9 | 环形图 | A→B→C→A 环不无限循环 |
| T3.10 | 嵌入不可用回退 | embedding=None 时纯图谱推理 |

### 1.4 并发安全测试 (目标: 0 → 15+)

**文件**: 新建 `tests/test_concurrency.py`

| ID | 测试项 | 内容 |
|:---|------|------|
| T4.1 | 4线程并发 add | 1000条无丢失无重复 |
| T4.2 | 读写混合并发 | 2写+2读, 结果一致性 |
| T4.3 | FAISS 并发搜索 | 4线程同时 search |
| T4.4 | FAISS 并发写入 | 4线程同时 add + train |
| T4.5 | 存储并发 | SQLite WAL模式 4线程写入 |
| T4.6 | 死锁检测 | 故意竞争场景 + timeout |

### 1.5 降级路径测试 (目标: 0 → 20+)

**文件**: 新建 `tests/test_fallback.py`

| ID | 测试项 | 内容 |
|:---|------|------|
| T5.1 | FAISS→线性检索 | 模拟 FAISS 不可用 |
| T5.2 | Ollama→MiniMax | 模拟 Ollama 超时 |
| T5.3 | MiniMax→sentence-transformers | 模拟 MiniMax API 错误 |
| T5.4 | Embedding→TF-IDF | 所有嵌入后端不可用 |
| T5.5 | 图谱→纯向量 | MemoryGraph 不可用时 |
| T5.6 | 时空索引→时序衰减 | SpacetimeIndex 关闭时 |
| T5.7 | 存储→内存 | SQLite 写入失败时 |

### Sprint 1 交付标准

| 指标 | 当前 | 目标 |
|------|:---:|:---:|
| lite_pro 测试 | 6 | **≥80** |
| FAISS 测试 | 0 | **≥25** |
| 多跳推理测试 | 4 | **≥30** |
| 并发测试 | 0 | **≥15** |
| 降级路径测试 | 0 | **≥20** |
| Sprint 1 新增 | — | **≥170 tests** |

---

## Sprint 2: 异常体系与结构加固（第2周）

### 2.1 统一错误码体系

**文件**: 新建 `src/su_memory/exceptions.py`

```python
class SuMemoryError(Exception):
    """su-memory 所有异常的基类"""
    def __init__(self, code: ErrorCode, detail: str = None):
        self.code = code
        self.detail = detail or code.default_message_zh
        super().__init__(f"[{code.name}] {self.detail}")

class ErrorCode(Enum):
    # 向量/索引
    FAISS_INDEX_CORRUPTED = ("FAISS_E001", "FAISS索引文件已损坏", "请删除 {path} 后重新初始化")
    FAISS_DIMENSION_MISMATCH = ("FAISS_E002", "向量维度不匹配", "期望 {expected}，实际 {actual}")
    # 嵌入
    EMBEDDER_UNAVAILABLE = ("EMB_E001", "所有嵌入后端均不可用", "请至少安装一个: pip install su-memory[ollama]")
    EMBEDDER_TIMEOUT = ("EMB_E002", "嵌入服务超时", "请检查 {backend} 服务是否运行")
    # 查询
    QUERY_EMPTY = ("QRY_E001", "查询文本不能为空", "")
    QUERY_NO_RESULTS = ("QRY_W001", "未找到匹配的记忆", "尝试更具体的查询词")
    # 存储
    STORAGE_WRITE_FAILED = ("STO_E001", "存储写入失败", "磁盘空间不足或路径权限问题")
    STORAGE_READ_FAILED = ("STO_E002", "存储读取失败", "数据文件可能已损坏")
    # 并发
    CONCURRENCY_DEADLOCK = ("CON_E001", "检测到死锁", "请减少并发写入线程数")
    # ... (共30+错误码)
```

### 2.2 异常链修复（扫描整个 `src/`）

| 步骤 | 内容 |
|:---|------|
| 扫描 | `grep -rn "except" src/su_memory/` 找出所有异常处理点 |
| 修复 | 所有 `raise NewError(...)` → `raise NewError(...) from e` |
| 替换 | 所有 `print/logger.error` → 抛出 `SuMemoryError` |
| 验证 | 每个异常路径写一个测试用例 |

### 2.3 降级矩阵实现与文档化

| 组件 | 主路径 | 降级路径1 | 降级路径2 | 测试 |
|------|--------|----------|----------|:---:|
| 嵌入 | Ollama(bge-m3) | MiniMax API | sentence-transformers(all-MiniLM) | ✅ |
| 嵌入 | 以上全部 | TF-IDF | — | ✅ |
| 向量索引 | FAISS HNSW | 线性检索 (numpy) | — | ✅ |
| 图谱 | MemoryGraph | 纯向量检索 | — | ✅ |
| 时空 | SpacetimeIndex | TemporalSystem(时序衰减) | — | ✅ |
| 存储 | Qdrant | SQLite | 内存 Dict | ✅ |
| 能量推断 | LLM(≥85%) | 关键词规则(≥60%) | 默认值 | 🟡 |

### 2.4 `__init__.py` 巨量导入修剪

**当前问题**: 596行，加载所有子模块，启动慢  
**目标**: 保留公开 API 导出，延迟加载内部 `_sys/` 模块

| 任务 | 内容 |
|:---|------|
| `_sys/` 懒加载 | `from su_memory._sys import X` → `lazy_import('su_memory._sys.X', 'X')` |
| 公开 API 保留 | `__all__` 列表中 ~200 个符号不变 |
| 启动时间基准 | `python -c "import su_memory"` 测量前后对比 |

### Sprint 2 交付标准

| 指标 | 目标 |
|------|:---:|
| 统一错误码 | ✅ 30+ ErrorCode |
| 异常链保留 | ✅ `raise ... from e` 100% 覆盖 |
| 降级矩阵覆盖率 | ✅ 6/7 组件 |
| `__init__.py` 启动 | ≤ 500ms |

---

## Sprint 3: 性能基准与优化（第3周）

### 3.1 性能基准套件

**目录**: `benchmarks/`

| 文件 | 指标 | 门禁阈值 |
|------|------|:---:|
| `bench_add.py` | 单条写入吞吐 | ≥ 80条/s |
| `bench_add_batch.py` | 批量写入吞吐 (100/1000/10000) | ≥ 500条/s |
| `bench_query.py` | P50/P95/P99 查询延迟 | P99 ≤ 50ms |
| `bench_multihop.py` | 多跳推理延迟 vs hop数 | 3-hop ≤ 200ms |
| `bench_faiss.py` | FAISS 构建/搜索/持久化 | search ≤ 10ms |
| `bench_memory.py` | 1K/10K/100K 内存占用 | 100K ≤ 500MB |
| `bench_concurrency.py` | 4/8线程读写吞吐 | 线性扩展 > 2.5x |

### 3.2 CI 性能门禁（GitHub Actions）

```yaml
# .github/workflows/perf-gate.yml
jobs:
  benchmark:
    steps:
      - run: pytest benchmarks/ --benchmark-only --benchmark-json=results.json
      - run: python scripts/check_perf_gate.py results.json
        # query_p99 < 50ms, write_throughput > 80/s, mem_10k < 200MB
```

### 3.3 优化项

| 任务 | 预期效果 |
|------|------|
| FAISS IVF 聚类数自动调参 | search 延迟 -20% |
| 嵌入缓存 LFU + TTL | 命中率 > 90% → query -30% |
| `_sys/` 延迟加载 | 首次 import -60% |

### Sprint 3 交付标准

| 指标 | 目标 |
|------|:---:|
| 查询 P99 | ≤ 50ms |
| 写入吞吐 | ≥ 80条/s（单条）/ ≥ 500条/s（批量） |
| 启动时间 | ≤ 500ms |
| 10万记忆压测 | ✅ 全功能可用 |
| CI 性能门禁 | ✅ |

---

## Sprint 4: 文档与发布（第4周）

### 4.1 API 文档自动生成

| 任务 | 工具 | 输出 |
|------|------|------|
| Sphinx 配置 | `sphinx-quickstart` + `autodoc` + `napoleon` | `docs/api/` |
| docstring 补全 | 所有公开 API 添加 Google-style docstring | 100% 覆盖 |
| ReadTheDocs 部署 | `.readthedocs.yaml` | `su-memory.readthedocs.io` |

### 4.2 README 重写

| 更新项 | 说明 |
|------|------|
| 版本号 | `v1.4.0` → `v2.6.0` |
| 安装方式 | 更新为当前 GitHub repo URL |
| 架构图 | 反映 v2.6.0 现状（移除过时引用） |

### 4.3 迁移指南

`docs/MIGRATION_v2.5_to_v2.6.md`:
- 降级行为说明（新异常类型）
- 配置参数变化
- 数据兼容性保证

### 4.4 发布清单

- [ ] CHANGELOG.md 更新（Keep a Changelog 格式）
- [ ] `pyproject.toml` version → 2.6.0 + `__init__.py` `__version__` → 2.6.0
- [ ] `git tag v2.6.0` + GitHub Release
- [ ] PyPI 发布：`pip install su-memory==2.6.0`

---

## 📅 Sprint 时间线

```
Week 1          Week 2          Week 3          Week 4
Sprint 1        Sprint 2        Sprint 3        Sprint 4
测试补全         异常体系         性能基准          文档发布
├─ lite_pro     ├─ ErrorCode    ├─ benchmarks/   ├─ Sphinx API
├─ FAISS        ├─ 异常链扫描    ├─ CI 性能门禁    ├─ README
├─ VectorGraph  ├─ 降级矩阵      ├─ 优化项        ├─ 迁移指南
├─ 并发测试      ├─ init 修剪    ├─ 10万压测      ├─ PyPI 发布
└─ 降级测试      └─             └─               └─ v2.6.0 🚀
+170 tests      +30 错误码      P99≤50ms        文档100%
```

---

## 📈 v2.6.0 终态验收标准

| 维度 | 指标 | 当前 | 目标 |
|------|------|:---:|:---:|
| **测试** | 总测试数 | 756 | **≥400 (核心新增)** |
| | 核心模块覆盖 | ~60% | **≥90%** |
| | 并发覆盖 | 0 | **≥15** |
| | 降级覆盖 | 0 | **≥20** |
| **性能** | 查询 P99 | ~76ms | **≤50ms** |
| | 写入吞吐 | ? | **≥80条/s** |
| | 启动时间 | ~2s | **≤500ms** |
| | 10万压测 | ❌ | **✅** |
| **质量** | 统一错误码 | ❌ | **✅ 30+** |
| | 异常链保留 | ~40% | **100%** |
| | 降级矩阵 | ❌ | **✅ 6/7** |
| | CI 性能门禁 | ❌ | **✅** |
| **文档** | API Reference | ❌ | **✅ Sphinx** |
| | README 版本 | v1.4.0 | **v2.6.0** |
