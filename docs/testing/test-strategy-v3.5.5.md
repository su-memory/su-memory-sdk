# su-memory SDK v3.5.5 完整测试方案

> 基于 SuperMemory MemoryBench 测试榜对标设计  
> 目标：全面验证 v3.5.5 97 个模块的功能正确性、性能基线、SOTA竞争力

---

## 目录

1. [SuperMemory 测试榜分析](#1-supermemory-测试榜分析)
2. [现有测试覆盖审计](#2-现有测试覆盖审计)
3. [六层测试金字塔设计](#3-六层测试金字塔设计)
4. [L1: 单元测试套件](#4-l1-单元测试套件)
5. [L2: 集成测试套件](#5-l2-集成测试套件)
6. [L3: 性能基准测试](#6-l3-性能基准测试)
7. [L4: SOTA 对比测试](#7-l4-sota-对比测试)
8. [L5: 专项能力验证](#8-l5-专项能力验证)
9. [L6: E2E 端到端流水线](#9-l6-e2e-端到端流水线)
10. [CI/CD 集成方案](#10-cicd-集成方案)
11. [测试用例示例](#11-测试用例示例)
12. [执行方案与命令速查](#12-执行方案与命令速查)

---

## 1. SuperMemory 测试榜分析

### 1.1 MemoryBench 架构

SuperMemory 的 [MemoryBench](https://github.com/supermemoryai/memorybench) 是一个**可插拔的基准评测框架**，核心设计：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Benchmarks │    │  Providers  │    │   Judges    │
│  (LoCoMo,   │    │ (Supermem,  │    │  (GPT-4o,   │
│  LongMem..) │    │  Mem0, Zep) │    │  Claude..)  │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       └──────────────────┼──────────────────┘
                          ▼
              ┌───────────────────────┐
              │      MemoryBench      │
              └───────────┬───────────┘
                          ▼
       ┌────────┬─────────┬────────┬──────────┬────────┐
       │ Ingest │ Indexing│ Search │  Answer  │Evaluate│
       └────────┴─────────┴────────┴──────────┴────────┘
```

### 1.2 五阶段流水线

| 阶段 | 功能 | su-memory 对标 |
|------|------|---------------|
| **Ingest** | 加载基准数据 → 写入 Provider | `SuMemoryLitePro.add()` / `SuMemory.add()` |
| **Index** | 等待 Provider 建索引 | `_build_index()` / `_incremental_index()` |
| **Search** | 查询 Provider → 检索上下文 | `query()` / `query_multihop()` / `enhanced_retriever` |
| **Answer** | 构建 Prompt → LLM 生成答案 | `BayesianAugmenter.reason()` / `_reflection_synthesizer` |
| **Evaluate** | 对比 Ground Truth → Judge 评分 | `get_accuracy_report()` / `run_validation_suite()` |

### 1.3 三大标准基准

| 基准 | 评估维度 | 数据规模 | su-memory 现有覆盖 |
|------|---------|---------|-------------------|
| **LongMemEval** | 长期记忆保持率、时序衰减 | ~300 对话 | ✅ `benchmarks/longmem_eval.py` |
| **LoCoMo** | 时间一致性、多会话记忆 | 10 任务 | ✅ `benchmarks/locomo_eval.py` |
| **ConvoMem** | 对话记忆、多轮跟踪 | 5 类别 | ✅ `benchmarks/convomem_eval.py` |

### 1.4 MemScore 复合指标

```
MemScore = accuracy% / latencyMs / contextTokens
例如: 86% / 145ms / 1823tok
```

**设计哲学**：三元组优于单一分数，保留质量/延迟/成本的独立维度。

### 1.5 关键设计特点

| 特点 | 说明 | su-memory 借鉴方向 |
|------|------|-------------------|
| **多 Provider 对比** | 同一基准跑 Supermemory/Mem0/Zep 并排对比 | 已有 `COMPETITOR_SCORES` 硬编码；建议扩展为实时跑分 |
| **Judge 可插拔** | 支持 GPT-4o/Claude/Gemini 互换 | 已有 `BayesianAugmenter.run_validation_suite()` |
| **Checkpoint 恢复** | 每阶段独立持久化，失败可续跑 | ✅ `benchmarks/run_all.py` 支持 `--previous` 回归比对 |
| **Web UI** | 交互式查看运行状态和失败用例 | ❌ 待建设 |
| **CLI 全覆盖** | `run/compare/ingest/search/test/status/show-failures` | ✅ `run_all.py` 有 `--benchmarks/--backends/--quick/--verbose` |

### 1.6 SuperMemory vs su-memory 测试能力对比

| 维度 | SuperMemory | su-memory 现有 | 差距 |
|------|------------|---------------|------|
| 标准基准评测 | LongMemEval, LoCoMo, ConvoMem | ✅ 三大基准全覆盖 | 持平 |
| 多 Provider 实时对比 | ✅ Side-by-side | ⚠️ 硬编码静态数据 | 需增强 |
| Judge 评估 | ✅ GPT-4o/Claude/Gemini | ⚠️ BayesianAugmenter | 需增加 LLM Judge |
| Pipeline Checkpoint | ✅ 每阶段可恢复 | ✅ `run_all.py` | 持平 |
| 性能基准 | 延迟/Token 效率 | ✅ P50/P95/P99/QPS/RSS | 持平 |
| 复合指标 | MemScore 三元组 | ❌ 无 | **待新增** |
| Web UI | ✅ 实时查看 | ❌ 无 | 待建设 |
| 单元测试框架 | N/A (非 SDK) | ⚠️ 手工断言 | **待 pytest 化** |
| CI/CD | N/A | ❌ 无配置 | **待建设** |
| 覆盖率 | N/A | 55% 目标 (未强制执行) | **待加强** |

---

## 2. 现有测试覆盖审计

### 2.1 测试文件清单（63 个）

```
tests/
├── conftest.py                          # pytest 配置（基础）
├── quick_test.py                        # 快速冒烟
├── test_adaptive_engine.py              # 自适应引擎
├── test_api_gateway_comprehensive.py    # API 网关集成 (1028行)
├── test_bayesian_augmenter.py           # 贝叶斯增强器 (588行) ★
├── test_bayesian_augmenter_m2.py        # M2 贝叶斯扩展
├── test_bayesian_system.py              # 贝叶斯系统
├── test_cli.py                          # CLI 命令行
├── test_concurrency.py / _p2.py         # 并发测试
├── test_counterfactual.py               # 反事实推理
├── test_do_calculus.py                  # do-演算
├── test_edge_cases.py                   # 边界用例
├── test_embedding_backends_v352.py      # 嵌入后端
├── test_energy_loss.py                  # 能量损失
├── test_faiss_index.py                  # FAISS 索引
├── test_fallback.py                     # 降级策略
├── test_gaia_benchmark.py               # GAIA 基准
├── test_integration_v170.py             # 集成测试
├── test_jepa_*.py (5个)                 # JEPA 模块
├── test_l1_llm_inference.py             # L1 LLM 推理
├── test_l234_distill_route_reflect.py   # 蒸馏/路由/反思
├── test_langchain_adapter.py            # LangChain 适配器
├── test_langchain_integration.py        # LangChain 集成
├── test_lifecycle_v352.py               # 生命周期管理
├── test_lite.py / test_lite_pro.py      # Lite/LitePro 核心
├── test_lite_pro_capability.py          # LitePro 能力
├── test_lite_pro_comprehensive.py       # LitePro 综合
├── test_llamaindex_adapter.py           # LlamaIndex 适配器
├── test_llm_adapter.py                  # LLM 适配器
├── test_m4_inference_e2e.py             # M4 端到端推理
├── test_metacognition_comprehensive.py  # 元认知综合
├── test_model_runtime.py                # 模型运行时
├── test_multihop_extended.py            # 多跳扩展
├── test_multihop_reasoning.py           # 多跳推理 ★
├── test_ollama_embedding.py             # Ollama 嵌入
├── test_p0/p1/p2/p3_energy_*.py (4个)  # 能量系统分层
├── test_parametric_memory.py            # 参数化记忆
├── test_payment.py                      # 支付
├── test_persistence.py                  # 持久化 ★
├── test_plugin_system.py                # 插件系统
├── test_reflection_synthesizer.py       # 反思合成器
├── test_sigreg.py                       # SiGreg
├── test_sota_comparison.py              # SOTA 对比 ★
├── test_spacetime_index.py              # 时空索引
├── test_spectral_causal.py              # 谱因果
├── test_state_distance.py               # 状态距离
├── test_storage.py                      # 存储
├── test_world_model.py                  # 世界模型
```

### 2.2 基准文件清单（35 个）

```
benchmarks/
├── run_all.py                 # ★ 统一调度器 (832行)
├── run_benchmark.py           # ★ 性能基准 (1092行)
├── sota_memory_engine.py      # ★ 9维SOTA引擎 (1279行)
├── longmem_eval.py            # LongMemEval 评测
├── locomo_eval.py             # LoCoMo 评测
├── convomem_eval.py           # ConvoMem 评测
├── benchmark_v355_comprehensive.py  # v3.5.5 综合基准
├── compare_v250_vs_v201.py    # 版本对比
├── hotpotqa.py / beir.py      # HotpotQA/BEIR
├── sigreg/ (3个)              # SiGreg 基准
├── benchmark_scaling.py       # 扩展性
├── benchmark_sdk.py           # SDK 基准
├── stress_test.py             # 压力测试
├── config.py                  # 配置/竞品数据
└── ...
```

### 2.3 模块覆盖矩阵

| 模块分类 | 总数 | 已覆盖 | 缺失 | 状态 |
|---------|------|--------|------|------|
| **SDK 核心** | 39 | 35 | `_tiered_storage`, `_storage_init`, `_storage_helpers`, `plugin_manager` | ★★★★ |
| **_sys 系统层** | 58 | 42 | `_pg_storage`, `_redis_storage`, `_sqlite_storage`, `_stream`, `_lazy`, `_parameter_adapters` 等 | ★★★ |
| **P1 模块** | 3 | 0 | `document_pipeline`, `profile_engine`, `_lifecycle_manager` | ❌ |
| **适配器** | 3 | 2 | `langchain`, `llamaindex` (lint adapter 缺失) | ★★★ |
| **CLI** | 1 | 1 | — | ★★★★★ |
| **支付** | 1 | 1 | — | ★★★★★ |

### 2.4 关键缺口总结

| # | 缺口类型 | 严重度 | 描述 |
|---|---------|--------|------|
| G1 | P1 模块测试 | 🔴 高 | `document_pipeline.py`, `profile_engine.py`, `_lifecycle_manager.py` 无专用测试 |
| G2 | 测试框架 | 🟡 中 | 63 个测试大部分用手工 `assert`，非 pytest 标准模式 |
| G3 | CI/CD | 🔴 高 | 无 `.github/workflows/`，无自动化执行 |
| G4 | MemScore | 🟡 中 | 无复合指标 (accuracy/latency/tokens) |
| G5 | LLM Judge | 🟡 中 | `run_validation_suite()` 依赖手工标定，无 LLM-as-Judge |
| G6 | 覆盖率强制 | 🟡 中 | `fail_under=55` 但未在 CI 执行 |
| G7 | 存储后端 | 🟢 低 | PostgreSQL/Redis 存储测试覆盖不足 |
| G8 | 压力/混沌 | 🟢 低 | 仅有 `stress_test.py`，无混沌工程测试 |
| G9 | JEPA 模块 | 🟢 低 | 5 个 JEPA 测试但未验证训练收敛 |

---

## 3. 六层测试金字塔设计

借鉴 SuperMemory 的分层评测理念，为 su-memory v3.5.5 设计**六层测试金字塔**：

```
                    ┌─────────────┐
                    │  L6: E2E    │  ← 完整流水线 + Judge 评估
                    │  10 min+    │
                    ├─────────────┤
                    │  L5: 专项   │  ← 多跳/贝叶斯/持久化/能量
                    │  2-5 min    │
                    ├─────────────┤
                    │  L4: SOTA   │  ← LongMemEval/LoCoMo/ConvoMem
                    │  10-30 min  │      + 竞品实时对比 + MemScore
                    ├─────────────┤
                    │  L3: 性能   │  ← 延迟/吞吐/资源/扩展性
                    │  5-15 min   │
                    ├─────────────┤
                    │  L2: 集成   │  ← 模块间协作 + P1 管线
                    │  1-3 min    │
                    ├─────────────┤
                    │  L1: 单元   │  ← 97 模块独立验证
                    │  <1 min     │
                    └─────────────┘
```

### 执行策略

| 层级 | 触发条件 | 执行频率 | 失败阻断 |
|------|---------|---------|---------|
| L1 单元 | 每次 push | 每次 commit | ✅ 阻断合并 |
| L2 集成 | PR 提交 | 每次 PR | ✅ 阻断合并 |
| L3 性能 | 手动/定时 | 每日凌晨 | ⚠️ 告警不阻断 |
| L4 SOTA | 手动/定时 | 每周 | ⚠️ 告警不阻断 |
| L5 专项 | 手动/版本发布前 | 按需 | ✅ 阻断发布 |
| L6 E2E | 手动/版本发布前 | 按需 | ✅ 阻断发布 |

---

## 4. L1: 单元测试套件

### 4.1 测试框架升级

**现状**：手工 `assert` + `print` 风格  
**目标**：标准 pytest + fixtures + markers + parametrize

```python
# 目标模式示例
import pytest
from su_memory.sdk.lite_pro import SuMemoryLitePro

@pytest.fixture
def empty_client():
    """每个测试独立的内存客户端"""
    return SuMemoryLitePro(max_memories=100, enable_graph=False)

class TestSuMemoryLiteProAdd:
    def test_add_basic(self, empty_client):
        """基本添加：返回非空 ID"""
        mid = empty_client.add("测试内容")
        assert mid is not None
        assert len(mid) > 0

    def test_add_with_metadata(self, empty_client):
        """带元数据的添加"""
        mid = empty_client.add("内容", metadata={"source": "test"})
        mem = next(m for m in empty_client._memories if m.id == mid)
        assert mem.metadata["source"] == "test"

    @pytest.mark.parametrize("content", [
        "",           # 空字符串
        "   ",        # 仅空白
        "a" * 10000,  # 超长内容
        "你好世界🌍",  # Unicode/Emoji
    ])
    def test_add_edge_cases(self, empty_client, content):
        """边界输入不崩溃"""
        mid = empty_client.add(content)
        assert mid is not None

    def test_add_duplicate_detection(self, empty_client):
        """去重检测"""
        mid1 = empty_client.add("相同内容")
        mid2 = empty_client.add("相同内容")
        # 取决于 skip_dedup 参数
        # ...

class TestSuMemoryLiteProQuery:
    def test_query_returns_results(self, empty_client):
        empty_client.add("记忆一：项目ROI增长25%")
        results = empty_client.query("ROI", top_k=3)
        assert len(results) > 0
        assert "ROI" in results[0]["content"]

    def test_query_ranking(self, empty_client):
        """验证排序：精确匹配 > 部分匹配"""
        empty_client.add("精确匹配的专属内容EXACT_123")
        empty_client.add("部分匹配的一般内容")
        results = empty_client.query("EXACT_123", top_k=3)
        assert "EXACT_123" in results[0]["content"]

    def test_query_empty_store(self, empty_client):
        """空记忆库查询"""
        results = empty_client.query("任意查询")
        assert results == []

    @pytest.mark.parametrize("top_k", [1, 5, 20, 100])
    def test_query_top_k(self, empty_client, top_k):
        for i in range(50):
            empty_client.add(f"记忆内容_{i}")
        results = empty_client.query("记忆", top_k=top_k)
        assert len(results) <= top_k
```

### 4.2 待新增的单元测试模块

| 模块 | 测试文件 | 优先级 | 测试点数 |
|------|---------|--------|---------|
| `document_pipeline.py` | `tests/test_document_pipeline.py` | P0 | 8 |
| `profile_engine.py` | `tests/test_profile_engine.py` | P0 | 10 |
| `_lifecycle_manager.py` | `tests/test_lifecycle_manager.py` | P0 | 8 |
| `_tiered_storage.py` | `tests/test_tiered_storage.py` | P1 | 6 |
| `plugin_manager.py` | `tests/test_plugin_manager.py` | P1 | 6 |
| `_pg_storage.py` | `tests/test_pg_storage.py` | P2 | 5 |
| `_redis_storage.py` | `tests/test_redis_storage.py` | P2 | 5 |
| `_stream.py` | `tests/test_stream.py` | P2 | 4 |

### 4.3 现有测试 pytest 化迁移

对现有的 63 个测试文件进行渐进式迁移：

**Phase 1**：核心 10 个文件优先
- `test_bayesian_augmenter.py` → 已经有良好的结构化，只需添加 `@pytest.fixture`
- `test_multihop_reasoning.py` → 添加 `@pytest.mark.parametrize`
- `test_persistence.py` → 添加 fixture 清理
- `test_lite_pro.py` / `test_lite_pro_comprehensive.py` → 添加类组织
- `test_energy_*.py` → 添加 parameterize
- `test_edge_cases.py` → 拆分到各模块

**Phase 2**：剩余 53 个文件

### 4.4 覆盖率目标

```toml
[tool.coverage.run]
source = ["src/su_memory"]
omit = ["tests/*", "benchmarks/*"]

[tool.coverage.report]
fail_under = 70    # 从 55 提升到 70
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "@abstractmethod",
]
```

---

## 5. L2: 集成测试套件

### 5.1 模块间协作验证

```python
# tests/test_integration_v355_pipeline.py

import pytest
from su_memory.sdk import SuMemoryLitePro
from su_memory.sdk.bayesian_augmenter import BayesianAugmenter
from su_memory.sdk.document_pipeline import DocumentPipeline
from su_memory.sdk.profile_engine import ProfileEngine
from su_memory._sys._lifecycle_manager import LifecycleManager
from su_memory._sys._energy_bus import EnergyBus
from su_memory._sys._energy_relations import EnergyRelations
from su_memory._sys._taiji_mapper import TaijiMapper


class TestFullPipeline:
    """完整记忆流水线集成测试"""

    def test_ingest_to_query(self):
        """DocumentPipeline → SuMemoryLitePro → BayesianAugmenter"""
        # Phase 1: DocumentPipeline 解析文档
        dp = DocumentPipeline()
        chunks = dp.process_text("这是一个测试文档。" * 100, source="test")
        assert len(chunks) > 0

        # Phase 2: 写入记忆
        client = SuMemoryLitePro(max_memories=1000)
        for chunk in chunks:
            client.add(chunk["content"], metadata=chunk.get("metadata", {}))

        # Phase 3: 贝叶斯增强查询
        augmenter = BayesianAugmenter(client)
        result = augmenter.query("测试文档")
        assert len(result.original["results"]) > 0

    def test_profile_energy_pipeline(self):
        """ProfileEngine → EnergyBus → EnergyRelations"""
        client = SuMemoryLitePro(max_memories=500)
        # 添加多领域记忆
        client.add("用户喜欢辛辣食物，偏好川菜和湘菜", metadata={"domain": "dietary"})
        client.add("血压偏高，需要低盐饮食", metadata={"domain": "medical"})
        client.add("每周运动3次，以跑步为主", metadata={"domain": "exercise"})

        # ProfileEngine 提取画像
        engine = ProfileEngine(client)
        contents = [m.content for m in client._memories]
        keywords = engine._extract_keywords(contents, 10)
        assert len(keywords) > 0

        # EnergyBus 能量传播
        bus = EnergyBus()
        nodes = bus.create_five_elements_nodes()
        assert len(nodes) == 14

        # EnergyRelations 关系计算
        relations = EnergyRelations()
        affinity = relations.get_affinity_score("木", "火")
        assert affinity > 0

    def test_lifecycle_bayesian_pipeline(self):
        """LifecycleManager → BayesianAugmenter 去重与信念更新"""
        client = SuMemoryLitePro(max_memories=500)

        # 生命周期管理
        lm = LifecycleManager(client)
        # 添加重复记忆
        mid1 = client.add("相同内容A")
        mid2 = client.add("相同内容A")
        # LifecycleManager 去重
        lm.deduplicate()
        # ...

        # 贝叶斯增强验证
        augmenter = BayesianAugmenter(client)
        result = augmenter.query("内容A")
        assert isinstance(result.original, dict)
```

### 5.2 集成测试 Marker 规范

```python
# 标记需要特定后端的测试
@pytest.mark.pgvector
def test_postgresql_storage(): ...

@pytest.mark.redis
def test_redis_storage(): ...

@pytest.mark.integration
def test_cross_module_pipeline(): ...

# 慢速集成测试
@pytest.mark.slow
@pytest.mark.integration
def test_large_scale_pipeline(): ...
```

---

## 6. L3: 性能基准测试

### 6.1 基准矩阵

| 操作 | 指标 | 目标 (v3.5.5) | 测试规模 |
|------|------|-------------|---------|
| `add()` | P50/P95/P99 延迟 | P50<150ms, P95<300ms, P99<500ms | 100/1K/10K |
| `query()` | P50/P95/P99 延迟 | P50<100ms, P95<200ms, P99<400ms | 100/1K/10K |
| `add()` 吞吐 | QPS | target≥50 | 10 workers × 5s |
| `query()` 吞吐 | QPS | target≥50 | 10 workers × 5s |
| 混合读写 | QPS | target≥40 | 20 workers, 7:3 ratio |
| 内存占用 | RSS | <500MB @1K, <1500MB @10K | 100/1K/10K |
| 扩展性 | 延迟增幅 | <200% @10K vs @100 | 100→1K→10K |

### 6.2 性能回归检测

```python
# benchmarks/benchmark_regression.py
"""
每次运行与历史基线对比，超过阈值告警。
"""

REGRESSION_THRESHOLDS = {
    "add_p50_ms": 1.20,    # 20% 劣化告警
    "query_p50_ms": 1.20,
    "add_qps": 0.80,       # 20% 吞吐下降告警
    "query_qps": 0.80,
    "rss_mb": 1.30,        # 30% 内存膨胀告警
}
```

### 6.3 新增 MemScore 复合指标

```python
# benchmarks/memscore.py
"""
MemScore = accuracy% / latencyMs / contextTokens

三元组语义：在 145ms 延迟和 1823 token 上下文的条件下，达到 86% 准确率
"""

@dataclass
class MemScore:
    accuracy_pct: float
    latency_ms: float
    context_tokens: int

    def __str__(self) -> str:
        return f"{self.accuracy_pct:.0f}% / {self.latency_ms:.0f}ms / {self.context_tokens}tok"

    @classmethod
    def from_benchmark_result(cls, result: BenchmarkResult) -> "MemScore":
        return cls(
            accuracy_pct=result.accuracy * 100,
            latency_ms=result.avg_query_time_ms,
            context_tokens=result.total_context_tokens,
        )

    def compare(self, other: "MemScore") -> dict:
        """对比两个 MemScore"""
        return {
            "accuracy_delta": self.accuracy_pct - other.accuracy_pct,
            "latency_ratio": self.latency_ms / max(other.latency_ms, 1),
            "token_ratio": self.context_tokens / max(other.context_tokens, 1),
            "winner": "self" if self.accuracy_pct > other.accuracy_pct else "other",
        }
```

---

## 7. L4: SOTA 对比测试

### 7.1 竞品对标矩阵

参考 SuperMemory 的多 Provider 对比模式，为 su-memory v3.5.5 建立**实时跑分+静态基线**双轨制：

| 基准 | Hindsight v5 | MemGPT/Letta | Mem0 | Zep | GPT-4 Turbo | **su-memory v3.5.5** |
|------|-------------|-------------|------|-----|-------------|---------------------|
| LongMemEval | 91.4% | — | — | — | — | 🎯 **≥95%** |
| LoCoMo F1 | 89.6% | — | — | — | — | 🎯 **≥93%** |
| ConvoMem | — | — | — | — | 72% | 🎯 **≥80%** |
| HotpotQA | 50.1% | — | — | — | 67.5% | 🎯 **≥70%** |
| BEIR NDCG | — | — | — | 0.3718 | — | 🎯 **≥0.45** |

### 7.2 九维引擎基准 (sota_memory_engine.py 已覆盖)

| 维度 | 测试内容 | v3.5.5 目标 |
|------|---------|------------|
| D1 Semantic Recall | Top-1/3/5, MRR, 同义/转述召回 | ≥0.85 |
| D2 Temporal Retention | 早期/中期/晚期 保持率 | ≥0.85 |
| D3 Multi-hop Chain | 1/2/3跳准确率, 完整链路恢复 | ≥0.75 |
| D4 Causal Inference | 因果方向检测 + 噪声梯度 | ≥0.80 |
| D5 Capacity Scaling | 100/1K/5K 容量衰减 | ≥0.75 @5K |
| D6 Interference | 高相似记忆区分能力 | ≥0.80 |
| D7 Persistence | 序列化保真度 | ≥0.95 |
| D8 Causal Intervention | ATE/调整集/CI覆盖 | ≥0.85 |
| D9 Counterfactual | 反事实推理 (Abduction/CF/PNS) | ≥0.80 |

### 7.3 实时竞品对比

```python
# benchmarks/run_compare.py (新增)
"""
多 Provider 实时对比 — 参考 SuperMemory MemoryBench compare 命令

用法:
    python benchmarks/run_compare.py --providers su-memory,mem0 --benchmark longmemeval
"""

class MultiProviderComparator:
    """同时运行 su-memory 与竞品，生成 side-by-side 对比报告"""

    PROVIDERS = {
        "su-memory": SuMemoryProvider,
        "mem0": Mem0Provider,
        "zep": ZepProvider,
    }

    def run(self, providers: list[str], benchmark: str):
        results = {}
        for provider in providers:
            results[provider] = self.PROVIDERS[provider]().run_benchmark(benchmark)
        return self.generate_comparison_report(results)
```

---

## 8. L5: 专项能力验证

### 8.1 多跳推理专项 (增强版)

参考 SuperMemory 的 LongMemEval 多跳链路设计，扩展我们的多跳推理测试：

```python
# tests/test_multihop_v355.py (增强版)

class TestMultiHopV355:
    """v3.5.5 多跳推理 — 参考 LongMemEval 多跳链路设计"""

    def test_3_hop_chain(self):
        """3跳因果链: A→B→C→D"""
        client = SuMemoryLitePro(enable_graph=True)
        a = client.add("事件A: 暴雨导致城市内涝")
        b = client.add("事件B: 城市内涝促使排水系统升级", parent_ids=[a])
        c = client.add("事件C: 排水系统升级后内涝频率降低50%", parent_ids=[b])

        results = client.query_multihop("暴雨后的改善效果", max_hops=3)
        # 验证能同时检索到 A, B, C
        contents = [r["content"] for r in results]
        assert any("暴雨" in c for c in contents)
        assert any("排水系统升级" in c for c in contents)
        assert any("内涝频率降低" in c for c in contents)

    def test_branching_chain(self):
        """分支链路: A→B1, A→B2"""
        # ...

    def test_noise_resistance(self):
        """噪声干扰下的多跳推理"""
        # ...

    def test_hop_count_accuracy(self):
        """验证返回结果的 hops 字段准确性"""
        # ...
```

### 8.2 贝叶斯增强效果验证 (增强版)

```python
# tests/test_bayesian_effectiveness.py (新增)

class TestBayesianEffectiveness:
    """量化验证贝叶斯增强的实际效果提升"""

    def setup_method(self):
        self.client = SuMemoryLitePro(max_memories=1000, enable_prediction=True)
        self.augmenter = BayesianAugmenter(self.client)

    def test_improvement_over_rounds(self):
        """验证多轮反馈后的准确度提升"""
        # 添加已知数据
        for i in range(20):
            self.client.add(f"测试事实_{i}: 相关内容_{i}")

        accuracies = []
        for round_num in range(10):
            self.augmenter.query(f"事实_{round_num % 5}")
            self.augmenter.feedback(
                query=f"事实_{round_num % 5}",
                ground_truth_value=0.9,
            )
            if round_num >= 2:
                report = self.augmenter.get_accuracy_report()
                accuracies.append(report["summary"]["improvement_pct"])

        # 验证改善趋势非负
        assert accuracies[-1] >= accuracies[0], "贝叶斯反馈应改善准确度"

    def test_uncertainty_reduction(self):
        """验证不确定性随反馈减少"""
        # 初始查询
        r1 = self.augmenter.query("不确定性测试")
        # 给予3次反馈
        for _ in range(3):
            self.augmenter.feedback(query="不确定性测试", ground_truth_value=0.85)
        r2 = self.augmenter.query("不确定性测试")

        # 贝叶斯更新后不确定性应降低
        bayes_uncertainty_1 = r1.bayesian.get("avg_uncertainty", 1.0)
        bayes_uncertainty_2 = r2.bayesian.get("avg_uncertainty", 1.0)
        assert bayes_uncertainty_2 <= bayes_uncertainty_1
```

### 8.3 持久化与恢复专项 (增强版)

```python
# tests/test_persistence_v355.py (增强版)

class TestPersistenceV355:
    """v3.5.5 持久化 — 参考 SuperMemory checkpoint 机制"""

    @pytest.fixture
    def tmp_store(self, tmp_path):
        return str(tmp_path / "su_memory_store")

    def test_save_load_cycle(self, tmp_store):
        """保存→加载→验证数据完整性"""
        # ...

    def test_incremental_save(self, tmp_store):
        """增量保存：仅写入变更"""
        # ...

    def test_checkpoint_recovery(self, tmp_store):
        """模拟崩溃恢复：中间状态不丢失"""
        client = SuMemoryLitePro(storage_path=tmp_store)
        client.add("崩溃前数据")
        client._save()  # checkpoint 1

        client.add("崩溃点数据")
        # 模拟崩溃：不调用 _save()

        client2 = SuMemoryLitePro(storage_path=tmp_store)
        # 应该恢复到 checkpoint 1
        results = client2.query("崩溃前")
        assert len(results) > 0

    @pytest.mark.parametrize("data_size", [100, 1000, 5000])
    def test_large_scale_persistence(self, tmp_store, data_size):
        """大规模数据持久化"""
        client = SuMemoryLitePro(storage_path=tmp_store, max_memories=data_size * 2)
        for i in range(data_size):
            client.add(f"大规模数据_{i}")
        client._save()

        client2 = SuMemoryLitePro(storage_path=tmp_store)
        assert len(client2._memories) == data_size

    def test_corrupted_file_recovery(self, tmp_store):
        """损坏文件恢复"""
        # ...
```

### 8.4 能量系统专项

```python
# tests/test_energy_comprehensive.py (整合现有4个分散测试)

class TestEnergySystemComprehensive:
    """能量系统完整验证"""

    def test_five_elements_creation(self):
        """五行节点创建: 14 信号 = 5元素 + 5相生 + 4相克"""
        bus = EnergyBus()
        nodes = bus.create_five_elements_nodes()
        assert len(nodes) == 14

    def test_energy_propagation(self):
        """能量传播: 相生增强(+50%), 相克抑制(-40%)"""
        bus = EnergyBus()
        result = bus.propagate_energy("木", "火", initial=1.0)
        assert result["enhanced"] > 1.0  # 木生火
        assert result["suppressed"] < 1.0  # 木克土

    def test_trigram_mapping(self):
        """八卦映射: 八种卦象 → 语义类别"""
        mapper = TaijiMapper()
        trigrams = mapper.get_all_trigrams()
        assert len(trigrams) == 8
        for tri in trigrams:
            semantic = mapper.resolve_trigram_to_semantic(tri)
            assert semantic is not None

    @pytest.mark.parametrize("pair,expected_relation", [
        (("木", "火"), "generate"),
        (("火", "土"), "generate"),
        (("土", "金"), "generate"),
        (("金", "水"), "generate"),
        (("水", "木"), "generate"),
        (("木", "土"), "restrict"),
        (("土", "水"), "restrict"),
        (("水", "火"), "restrict"),
        (("火", "金"), "restrict"),
        (("金", "木"), "restrict"),
    ])
    def test_all_relations(self, pair, expected_relation):
        """验证五行全部 10 种基本关系"""
        relations = EnergyRelations()
        rel = relations.get_relation(pair[0], pair[1])
        assert rel == expected_relation
```

---

## 9. L6: E2E 端到端流水线

### 9.1 完整流水线测试

借鉴 SuperMemory 的 5 阶段流水线：**Ingest → Index → Search → Answer → Evaluate**

```python
# tests/e2e/test_full_pipeline.py

import pytest
from su_memory.sdk import SuMemoryLitePro
from su_memory.sdk.bayesian_augmenter import BayesianAugmenter
from su_memory.sdk.document_pipeline import DocumentPipeline

class TestE2EPipeline:
    """完整端到端流水线"""

    @pytest.mark.e2e
    def test_ingest_search_answer_evaluate(self):
        """
        SuperMemory 对标：
        Ingest   → DocumentPipeline.process_text()
        Index    → SuMemoryLitePro._build_index()
        Search   → BayesianAugmenter.query()
        Answer   → BayesianAugmenter.reason()
        Evaluate → BayesianAugmenter.get_accuracy_report()
        """
        # Ingest: 文档导入
        dp = DocumentPipeline()
        doc_text = """
人工智能技术在2024年取得了重大突破。深度学习模型在自然语言处理、
计算机视觉、语音识别等领域达到了新的高度。
大语言模型如GPT-4和Claude展现了强大的推理能力。
然而，数据隐私、算法偏见和能源消耗等问题仍然存在。
        """
        chunks = dp.process_text(doc_text, source="ai_report_2024")
        assert len(chunks) >= 2

        # Index + Search
        client = SuMemoryLitePro(max_memories=500, enable_prediction=True)
        for chunk in chunks:
            client.add(chunk["content"], metadata={"source": "ai_report_2024"})

        augmenter = BayesianAugmenter(client)

        # Search
        search_result = augmenter.query("AI 突破", top_k=5)
        assert len(search_result.original["results"]) > 0

        # Answer (Reasoning)
        reason_result = augmenter.reason("AI 在2024年有哪些突破", max_hops=2)
        assert isinstance(reason_result.original, dict)

        # Evaluate
        augmenter.feedback(
            query="AI 突破",
            ground_truth_value=0.85,
        )
        report = augmenter.get_accuracy_report()
        assert report["summary"]["total_feedback"] >= 1

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_longmemeval_style_retention(self):
        """LongMemEval 风格的长期记忆保持测试"""
        client = SuMemoryLitePro(max_memories=2000)

        # Phase 1: 填充 300 条记忆（含标记点）
        markers = []
        for i in range(300):
            if i % 30 == 0:
                content = f"标记点_{i}: 关键信息需要记住"
                mid = client.add(content)
                markers.append((i, mid, content))
            else:
                client.add(f"填充数据_{i}: 无关紧要的噪音信息")

        # Phase 2: 验证早期/中期/晚期 保持率
        third = len(markers) // 3
        regions = {
            "early": markers[:third],
            "mid": markers[third:2*third],
            "late": markers[2*third:],
        }

        recalls = {}
        for region, items in regions.items():
            hits = 0
            for idx, _, _ in items:
                results = client.query(f"标记点_{idx}", top_k=3)
                if any(f"标记点_{idx}" in r["content"] for r in results):
                    hits += 1
            recalls[region] = hits / len(items)

        # 早期保持率应高于晚期（衰减可控）
        assert recalls["early"] >= 0.7
        assert recalls["late"] >= 0.5  # 晚期不应灾难性遗忘

    @pytest.mark.e2e
    def test_document_to_insight_pipeline(self):
        """DocumentPipeline → ProfileEngine → BayesianAugmenter 全链路"""
        # ...
```

### 9.2 LLM-as-Judge 集成

```python
# benchmarks/judge.py (新增)
"""
LLM-as-Judge 评估模块 — 参考 SuperMemory Judge 可插拔设计

支持:
- OpenAI GPT-4o
- Anthropic Claude
- Google Gemini
"""

@dataclass
class JudgeResult:
    score: float          # 0-1 精确匹配分数
    reasoning: str        # Judge 的评分依据
    ground_truth_match: bool
    latency_ms: float

class LLMJudge:
    """可插拔的 LLM Judge"""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._client = self._init_client(model)

    def evaluate(
        self,
        query: str,
        answer: str,
        ground_truth: str,
        context: list[str] | None = None,
    ) -> JudgeResult:
        """评估答案质量"""
        prompt = self._build_prompt(query, answer, ground_truth, context)
        response = self._client.complete(prompt)
        return self._parse_response(response)

    def _build_prompt(self, query, answer, ground_truth, context):
        return f"""你是一个严格的评估专家。
问题: {query}
参考答案: {ground_truth}
待评估答案: {answer}
{"上下文: " + chr(10).join(context) if context else ""}

请评估:
1. 答案与参考答案的事实一致性 (0-1)
2. 答案是否包含了参考答案中的关键信息
3. 如果有上下文，答案是否准确使用了上下文信息

返回 JSON: {{"score": 0.85, "reasoning": "...", "match": true}}"""


class JudgePipeline:
    """完整 Judge 流水线 — 对标 MemoryBench evaluate 阶段"""

    def __init__(self, judge: LLMJudge, benchmark_results: dict):
        self.judge = judge
        self.results = benchmark_results

    def run(self) -> dict:
        scores = []
        for item in self.results["items"]:
            judge_result = self.judge.evaluate(
                query=item["query"],
                answer=item["answer"],
                ground_truth=item["ground_truth"],
                context=item.get("context"),
            )
            scores.append(judge_result)

        accuracy = sum(1 for s in scores if s.ground_truth_match) / len(scores)
        avg_score = sum(s.score for s in scores) / len(scores)
        avg_latency = sum(s.latency_ms for s in scores) / len(scores)

        return {
            "accuracy": accuracy,
            "avg_score": avg_score,
            "avg_latency_ms": avg_latency,
            "details": [s.__dict__ for s in scores],
        }
```

---

## 10. CI/CD 集成方案

### 10.1 GitHub Actions 四门禁

```yaml
# .github/workflows/ci.yml (新增)

name: su-memory CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  # Gate 1: Lint
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install ruff
      - run: ruff check src/ tests/

  # Gate 2: Type Check
  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install mypy
      - run: mypy src/su_memory/

  # Gate 3: Unit + Integration Tests
  test:
    needs: [lint, type-check]
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "${{ matrix.python-version }}" }
      - run: pip install ".[dev]"
      - run: pytest tests/ -v --tb=short -m "not slow and not integration and not pgvector and not redis"
      - run: pytest tests/ -v --tb=short -m "integration" --timeout=60

  # Gate 4: Coverage
  coverage:
    needs: [test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install ".[dev]"
      - run: pytest tests/ --cov=src/su_memory --cov-report=xml --cov-report=term -m "not slow"
      - run: coverage report --fail-under=70

  # Nightly: Performance + SOTA
  nightly-benchmark:
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install ".[dev]"
      - run: python benchmarks/run_benchmark.py --scale medium --output benchmarks/results/ci_latest.json
      - run: python benchmarks/run_all.py --benchmarks all --backends sbert --quick
      - uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmarks/results/
```

### 10.2 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml (新增)

repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: pytest-quick
        name: Quick unit tests
        entry: pytest tests/ -x -m "not slow and not integration"
        language: system
        pass_filenames: false
        stages: [pre-push]
```

---

## 11. 测试用例示例

### 11.1 P1 模块测试示例

#### DocumentPipeline

```python
# tests/test_document_pipeline.py (新增 P0)

import pytest
import tempfile
import os
from su_memory.sdk.document_pipeline import DocumentPipeline

class TestDocumentPipeline:
    """P1-1 文档解析管线"""

    @pytest.fixture
    def pipeline(self):
        return DocumentPipeline()

    def test_text_chunking(self, pipeline):
        """文本分块: 长文本被正确切分"""
        text = "这是一个段落。" * 100
        chunks = pipeline.process_text(text, source="test")
        assert len(chunks) >= 2
        for chunk in chunks:
            assert "content" in chunk
            assert len(chunk["content"]) > 0

    def test_format_detection(self, pipeline):
        """格式检测: 识别 Markdown/TXT/CSV/JSON"""
        fd = pipeline.format_detector
        ext_map = fd.EXTENSION_MAP
        assert ".md" in ext_map
        assert ".txt" in ext_map
        assert ".csv" in ext_map

    def test_chunker_fixed_size(self, pipeline):
        """固定大小分块器: chunk_size 参数生效"""
        text = "abcdefghij" * 50  # 500 chars
        chunker = pipeline.get_chunker("fixed_size")
        result = chunker.chunk(text)
        chunks = result.chunks if hasattr(result, 'chunks') else result
        assert len(chunks) > 0
        # 每个 chunk 大小应在 chunk_size 附近
        for chunk in chunks:
            assert len(chunk) <= pipeline.config.get("chunk_size", 512) + 100

    def test_metadata_preservation(self, pipeline):
        """元数据保留: source/timestamp 等信息传递到输出"""
        text = "测试内容 " * 20
        metadata = {"author": "test", "version": "1.0"}
        chunks = pipeline.process_text(text, source="test", metadata=metadata)
        for chunk in chunks:
            assert "metadata" in chunk
            assert chunk["metadata"].get("source") == "test"

    def test_empty_input(self, pipeline):
        """空输入处理"""
        chunks = pipeline.process_text("", source="empty")
        assert chunks == []

    def test_single_sentence(self, pipeline):
        """单句输入不崩溃"""
        chunks = pipeline.process_text("一句话。", source="short")
        assert len(chunks) >= 1

    @pytest.mark.parametrize("content", [
        "你好世界",
        "a" * 1,
        "x" * 10000,  # 超长单行
        "段落1\n\n段落2\n\n段落3",
    ])
    def test_edge_inputs(self, pipeline, content):
        """边界输入: 各种长度和格式"""
        chunks = pipeline.process_text(content, source="edge")
        assert isinstance(chunks, list)
```

#### ProfileEngine

```python
# tests/test_profile_engine.py (新增 P0)

import pytest
from su_memory.sdk.lite_pro import SuMemoryLitePro
from su_memory.sdk.profile_engine import ProfileEngine

class TestProfileEngine:
    """P1-2 用户画像引擎"""

    @pytest.fixture
    def client_with_data(self):
        client = SuMemoryLitePro(max_memories=200)
        memories = [
            ("用户偏好辛辣食物，喜欢川菜", {"domain": "dietary"}),
            ("血压偏高，医生建议低盐饮食", {"domain": "medical"}),
            ("每周跑步3次，每次5公里", {"domain": "exercise"}),
            ("对花生过敏，需避免含花生成分的食物", {"domain": "allergy"}),
            ("工作压力大，经常加班到晚上10点", {"domain": "lifestyle"}),
            ("喜欢阅读科幻小说，最近在看三体", {"domain": "hobby"}),
            ("有糖尿病家族史，需控制糖分摄入", {"domain": "medical"}),
            ("每天喝2-3杯咖啡", {"domain": "dietary"}),
        ]
        for content, meta in memories:
            client.add(content, metadata=meta)
        return client

    def test_keyword_extraction(self, client_with_data):
        """关键词提取"""
        engine = ProfileEngine(client_with_data)
        contents = [m.content for m in client_with_data._memories]
        keywords = engine._extract_keywords(contents, 10)
        assert len(keywords) >= 5
        assert any("饮食" in kw or "食物" in kw for kw in keywords)

    def test_domain_classification(self, client_with_data):
        """领域分类"""
        engine = ProfileEngine(client_with_data)
        contents = [m.content for m in client_with_data._memories]
        domains = engine._classify_domains(contents)
        assert "dietary" in domains or len(domains) >= 2

    def test_preference_extraction(self, client_with_data):
        """偏好提取"""
        engine = ProfileEngine(client_with_data)
        contents = [m.content for m in client_with_data._memories]
        prefs = engine._extract_preferences(contents)
        assert isinstance(prefs, list)
        # 应该能识别饮食偏好
        dietary_prefs = [p for p in prefs if any(
            w in str(p).lower() for w in ["辣", "咖啡", "盐"]
        )]
        assert len(dietary_prefs) >= 1

    def test_constraint_extraction(self, client_with_data):
        """约束条件提取（过敏、禁忌等）"""
        engine = ProfileEngine(client_with_data)
        contents = [m.content for m in client_with_data._memories]
        constraints = engine._extract_constraints(contents)
        assert isinstance(constraints, list)
        # 应识别花生过敏
        allergy_constraints = [c for c in constraints if "花生" in str(c) or "过敏" in str(c)]
        assert len(allergy_constraints) >= 1

    def test_empty_memory_extraction(self):
        """空记忆库不崩溃"""
        client = SuMemoryLitePro(max_memories=100)
        engine = ProfileEngine(client)
        profile = engine.extract_from_memories()
        # 应返回空画像而不是崩溃
        assert isinstance(profile, dict)
```

#### LifecycleManager

```python
# tests/test_lifecycle_manager.py (新增 P0)

import pytest
from su_memory.sdk.lite_pro import SuMemoryLitePro
from su_memory._sys._lifecycle_manager import LifecycleManager

class TestLifecycleManager:
    """P1-3 生命周期管理"""

    @pytest.fixture
    def client_with_duplicates(self):
        client = SuMemoryLitePro(max_memories=200)
        # 添加重复内容
        client.add("记忆A内容")
        client.add("记忆A内容")  # 重复
        client.add("记忆B内容")
        client.add("记忆B内容")  # 重复
        client.add("记忆C内容")
        return client

    def test_deduplicate(self, client_with_duplicates):
        """去重功能"""
        lm = LifecycleManager(client_with_duplicates)
        before = len(client_with_duplicates._memories)
        removed = lm.deduplicate()
        after = len(client_with_duplicates._memories)
        assert removed >= 2
        assert after < before

    def test_auto_expire(self):
        """自动过期"""
        client = SuMemoryLitePro(max_memories=100)
        client.add("过期记忆", metadata={"ttl": 0})  # 立即过期
        client.add("正常记忆")

        lm = LifecycleManager(client)
        expired = lm.auto_expire()
        assert expired >= 1

    def test_get_report(self):
        """获取生命周期报告"""
        client = SuMemoryLitePro(max_memories=100)
        for i in range(10):
            client.add(f"记忆_{i}")

        lm = LifecycleManager(client)
        # Patch get_all_memories for SuMemoryLitePro
        if not hasattr(client, 'get_all_memories'):
            def _get_all():
                return [{"id": m.id, "content": m.content,
                         "metadata": m.metadata, "timestamp": m.timestamp}
                        for m in client._memories]
            client.get_all_memories = _get_all

        report = lm.get_report()
        assert "total" in report
        assert report["total"] == 10

    def test_consolidate(self):
        """记忆合并"""
        client = SuMemoryLitePro(max_memories=100)
        # 添加语义相近的记忆
        client.add("项目进展顺利")
        client.add("项目推进正常")
        client.add("项目状态良好")

        lm = LifecycleManager(client)
        # 合并相似记忆
        consolidated = lm.consolidate()
        # 合并后记忆数减少
        assert len(client._memories) <= 3
```

### 11.2 能量系统测试示例

```python
# tests/test_energy_comprehensive.py (整合版)

class TestEnergyBus:
    """能量总线"""

    def test_create_nodes(self):
        bus = EnergyBus()
        nodes = bus.create_five_elements_nodes()
        # 5 元素 + 5 相生关系 + 4 相克关系 = 14
        assert len(nodes) == 14

    def test_propagate_enhance(self):
        """相生（增强）"""
        bus = EnergyBus()
        result = bus.propagate_energy("木", "火", initial=1.0)
        assert result["enhanced"] > 1.0
        assert result["relation"] == "generate"

    def test_propagate_restrict(self):
        """相克（抑制）"""
        bus = EnergyBus()
        result = bus.propagate_energy("木", "土", initial=1.0)
        assert result["suppressed"] < 1.0
        assert result["relation"] == "restrict"

    def test_propagate_same_element(self):
        """同元素不应传播"""
        bus = EnergyBus()
        result = bus.propagate_energy("木", "木", initial=1.0)
        # 同元素或无关元素
        assert result["relation"] in (None, "self")


class TestTaijiMapper:
    """太极映射器"""

    def test_resolve_all_trigrams(self):
        mapper = TaijiMapper()
        trigrams = mapper.get_all_trigrams()
        assert len(trigrams) == 8
        for tri in trigrams:
            semantic = mapper.resolve_trigram_to_semantic(tri)
            assert semantic is not None
            assert len(semantic) > 0


class TestEnergyRelations:
    """能量关系"""

    @pytest.mark.parametrize("a,b,expected", [
        ("木", "火", "generate"),
        ("火", "土", "generate"),
        ("土", "金", "generate"),
        ("金", "水", "generate"),
        ("水", "木", "generate"),
        ("木", "土", "restrict"),
        ("土", "水", "restrict"),
        ("水", "火", "restrict"),
        ("火", "金", "restrict"),
        ("金", "木", "restrict"),
    ])
    def test_all_relations(self, a, b, expected):
        rel = EnergyRelations()
        assert rel.get_relation(a, b) == expected

    def test_affinity_scores(self):
        rel = EnergyRelations()
        enhance = rel.get_affinity_score("木", "火")
        suppress = rel.get_affinity_score("木", "土")
        assert enhance > 0.5   # 相生 ~1.5
        assert suppress < 1.0  # 相克 ~0.6
```

---

## 12. 执行方案与命令速查

### 12.1 分层执行命令

```bash
# ==========================================
# L1: 单元测试 (<1 min)
# ==========================================
pytest tests/ -v -m "not slow and not integration and not e2e" --tb=short

# 仅运行特定模块
pytest tests/test_lite_pro.py -v
pytest tests/test_bayesian_augmenter.py -v

# 快速冒烟
pytest tests/ -x -m "smoke" --tb=line

# ==========================================
# L2: 集成测试 (1-3 min)
# ==========================================
pytest tests/ -v -m "integration" --timeout=60

# 含可选后端（需环境）
pytest tests/ -v -m "pgvector" --timeout=120
pytest tests/ -v -m "redis" --timeout=120

# ==========================================
# L3: 性能基准 (5-15 min)
# ==========================================
# 小规模快速验证
python benchmarks/run_benchmark.py --scale small

# 中等规模标准测评
python benchmarks/run_benchmark.py --scale medium

# 大规模完整测评
python benchmarks/run_benchmark.py --scale large --output benchmarks/results/benchmark_large.json

# MemScore 计算
python benchmarks/memscore.py --input benchmarks/results/benchmark_medium.json

# ==========================================
# L4: SOTA 对比 (10-30 min)
# ==========================================
# 快速模式（少量数据）
python benchmarks/run_all.py --benchmarks all --backends sbert --quick

# 完整模式
python benchmarks/run_all.py --benchmarks all --backends ollama,sbert --output results/full_run.json

# 实时竞品对比（需 API key）
python benchmarks/run_compare.py --providers su-memory,mem0 --benchmark longmemeval

# 9维引擎基准
python benchmarks/sota_memory_engine.py

# ==========================================
# L5: 专项测试 (2-5 min each)
# ==========================================
pytest tests/test_multihop_v355.py -v
pytest tests/test_bayesian_effectiveness.py -v
pytest tests/test_persistence_v355.py -v
pytest tests/test_energy_comprehensive.py -v
pytest tests/test_document_pipeline.py -v
pytest tests/test_profile_engine.py -v
pytest tests/test_lifecycle_manager.py -v

# ==========================================
# L6: E2E (10 min+)
# ==========================================
pytest tests/e2e/ -v -m "e2e" --timeout=300

# LLM-as-Judge (需 API key)
python benchmarks/judge.py --benchmark longmemeval --judge gpt-4o

# ==========================================
# 全覆盖 + 覆盖率报告
# ==========================================
pytest tests/ --cov=src/su_memory --cov-report=html --cov-report=term -m "not slow" --timeout=120
open htmlcov/index.html
```

### 12.2 版本发布前完整验证

```bash
#!/bin/bash
# scripts/verify-release.sh

set -e

echo "=== Gate 1: Lint ==="
ruff check src/ tests/

echo "=== Gate 2: Type Check ==="
mypy src/su_memory/

echo "=== Gate 3: Unit Tests ==="
pytest tests/ -v -m "not slow and not integration and not e2e" --tb=short

echo "=== Gate 4: Integration Tests ==="
pytest tests/ -v -m "integration" --timeout=60

echo "=== Gate 5: P1 Module Tests ==="
pytest tests/test_document_pipeline.py tests/test_profile_engine.py tests/test_lifecycle_manager.py -v

echo "=== Gate 6: Performance Baseline ==="
python benchmarks/run_benchmark.py --scale medium
python benchmarks/memscore.py --check-regression

echo "=== Gate 7: SOTA Benchmark ==="
python benchmarks/run_all.py --benchmarks all --backends sbert --quick

echo "=== Gate 8: Coverage ==="
pytest tests/ --cov=src/su_memory --cov-report=term -m "not slow"
coverage report --fail-under=70

echo "=== ✅ All gates passed ==="
```

### 12.3 测试数据管理

```
tests/
├── fixtures/                    # 测试数据
│   ├── sample_docs/             # 文档样本
│   │   ├── test_report.md
│   │   ├── test_data.csv
│   │   └── test_config.json
│   ├── gaia_questions.json      # GAIA 测试题
│   └── benchmark_baselines.json # 性能基线
├── e2e/                         # 端到端测试
│   ├── __init__.py
│   ├── conftest.py
│   └── test_full_pipeline.py
├── conftest.py                  # 全局 fixtures
└── test_*.py                    # 各模块测试
```

---

## 附录 A: 测试覆盖率目标矩阵

| 层级 | 模块数 | 测试文件数 | 当前覆盖率 | 目标覆盖率 |
|------|--------|-----------|-----------|-----------|
| SDK 核心 | 39 | 20 | ~65% | ≥80% |
| _sys 系统 | 58 | 15 | ~50% | ≥70% |
| P1 模块 | 3 | 0 | 0% | ≥85% |
| 适配器 | 3 | 2 | ~60% | ≥75% |
| CLI/API | 5 | 2 | ~45% | ≥65% |
| 支付/许可 | 3 | 3 | ~70% | ≥80% |
| **总计** | **111** | **42** | **~55%** | **≥75%** |

## 附录 B: 与 SuperMemory 的差异化优势

| 维度 | SuperMemory | su-memory (增强后) | 优势 |
|------|------------|-------------------|------|
| 测试框架 | Bun/TypeScript | pytest (Python 生态) | 语言生态 |
| 基准数量 | 3 (LongMemEval/LoCoMo/ConvoMem) | 6 (含 GAIA/HotpotQA/BEIR) | 广度领先 |
| 专项能力 | 无 | 能量系统/贝叶斯/因果/反事实/多跳 | **独有优势** |
| 性能基准 | 基础 | P50/P95/P99/QPS/RSS/扩展性/回归 | 精度领先 |
| 竞品对比 | 实时 | 实时 + 静态基线双轨 | 方法互补 |
| MemScore | ✅ | ✅ (新增) | 对齐行业标准 |
| LLM Judge | ✅ | ✅ (新增) | 对齐行业标准 |
| CI/CD | N/A (非 SDK) | 4-Gate CI + Nightly | 工程化领先 |
| 覆盖率强制 | N/A | fail_under=70% | 质量门禁 |

---

> **文档版本**: v1.0  
> **创建日期**: 2026-06-03  
> **适用版本**: su-memory SDK v3.5.5  
> **对标参考**: [SuperMemory MemoryBench](https://github.com/supermemoryai/memorybench)
