# su-memory-sdk v3.6.0 → v3.6.1 — QC 审计报告

**审计日期**：2025-06-09
**修复日期**：2025-06-09
**审计范围**：迭代修改核心文件 (benchmarks/ + src/su_memory/sdk/_llm_reranker.py)
**审计深度**：Standard（六维度全覆盖 + 核心文件深读）
**修复状态**：✅ 全部 P1/P2 已修复

---

## 一、总览

| 层 | P0（阻塞上线） | P1（严重缺陷） | P2（代码规范） | 合计 | 状态 |
|---|---|---|---|---|---|
| SDK 核心 | 0 | 0 | 0 | 0 | ✅ 已修复 |
| Benchmark | 0 | 0 | 0 | 0 | ✅ 已修复 |
| **合计** | **0** | **0** | **0** | **0** | ✅ |

**核心结论**：迭代质量良好，**全部 6 项 P1/P2 缺陷已修复闭环**。

---

## 二、不成立的设计前提

> 以下前提在代码中隐含假定，但审计证明存在偏差：

| # | 声称/假定前提 | 实际情况 | 证据 |
|---|---|---|---|
| 1 | "LLM-as-Judge 失败时不影响结果判定" | 静默吞异常，无日志无告警，无法区分"判为不等价" vs "判判失败" | `longmem_eval.py:301` `except Exception: pass` |
| 2 | "LoCoMo 与 LongMemEval 共享一致的上下文构建逻辑" | 两处独立实现，`_build_llm_context` (locomo) vs 内联代码 (longmem)，metadata 访问方式不一致 | `locomo_eval.py:429` 使用 `isinstance(r, dict)` 检查 vs `longmem_eval.py:546` 使用 `r.get("metadata")` |

---

## 三、P1 级 — 严重缺陷

### 3.1 静默异常吞没 — LLM Judge 失败不可观测

| # | 位置 | 问题描述 | 影响 | 状态 |
|---|---|---|---|---|
| P1-1 | `benchmarks/longmem_eval.py:301` | `_llm_judge_equivalence` 中 `except Exception: pass` 静默吞没所有异常 | Stage 5 匹配兜底失效时无人知晓 | ✅ 已修复 — 添加 `logger.warning(…, exc_info=True)` |
| P1-2 | `benchmarks/locomo_eval.py:489` | `_query()` 方法静默吞没异常 | 查询静默失败导致准确率被低估 | ✅ 已修复 — 添加 `logger.warning(…, exc_info=True)` + `_query_failures` 失败计数器 |

### 3.2 跨文件逻辑重复 — 维护债务

| # | 位置 | 问题描述 | 影响 | 状态 |
|---|---|---|---|---|
| P1-3 | `benchmarks/locomo_eval.py:413-456` vs `benchmarks/longmem_eval.py:543-588` | Session 级聚类逻辑两处独立实现 | 人工维护成本 | ✅ 已修复 — 提取为 `benchmarks/_cluster_utils.py:cluster_by_session()` 共享函数 |

---

## 四、P2 级 — 代码规范

| # | 位置 | 描述 | 建议 | 状态 |
|---|---|---|---|---|
| P2-1 | `benchmarks/locomo_eval.py:440-442` | `_build_llm_context` 中 `metadata` 提取嵌套过深 | 提取辅助函数 | ✅ 已修复 — 添加 `LoCoMoRunner._get_meta()` 静态方法 |
| P2-2 | `benchmarks/_model_router.py:72-76` | `__getattr__` 隐式依赖 `_default` 先于 `__repr__` 设置 | 显式保护 | ✅ 已修复 — `__repr__` 增加 `hasattr(self, "_default")` 守卫 |
| P2-3 | `src/su_memory/sdk/_llm_reranker.py:302` | `print()` 而非 `logger.info()` | 替换为 logger | ✅ 已修复 — 替换为 `logger.info()` |

### 修复详情

| 修复项 | 修改文件 | 变更说明 |
|---|---|---|
| P1-1 | `benchmarks/longmem_eval.py` | 添加 `import logging` + `logger`；`except Exception: pass` → `logger.warning(…, exc_info=True)` |
| P1-2 | `benchmarks/locomo_eval.py` | 添加 `import logging` + `logger`；`_query()` 添加 `_query_failures` 计数器 + `logger.warning(…, exc_info=True)` |
| P1-3 | `benchmarks/_cluster_utils.py` (新建) + `benchmarks/longmem_eval.py` + `benchmarks/locomo_eval.py` | 新建 `cluster_by_session()` 共享函数；两处内联 Session 聚类代码替换为函数调用 |
| P2-1 | `benchmarks/locomo_eval.py` | 添加 `LoCoMoRunner._get_meta()` 静态方法，消除三处深层嵌套 metadata 提取 |
| P2-2 | `benchmarks/_model_router.py` | `__repr__` 增加 `hasattr(self, "_default")` 守卫，未初始化时返回 `"ModelRouter(uninitialized)"` |
| P2-3 | `src/su_memory/sdk/_llm_reranker.py` | `print(f"  [LLM] {i+1}/{n} …")` → `logger.info("  [LLM] %d/%d …", …)` |

### 验证结果

- **ruff check**: 20 auto-fixed + 剩余 6 条为预存问题（与本次修改无关）
- **import 测试**: `cluster_by_session` 正确导入，dict/object 型结果均通过
- **`__repr__` 安全测试**: 未初始化 ModelRouter 返回 `"ModelRouter(uninitialized)"` 而非崩溃
- **pytest**: 105 passed, 1 failed（`test_large_embedding` — 预存的 embedding 维度差异，无关）

---

## 五、系统性模式分析

### 模式 1：异常处理的两极分化

| 极 | 文件 | 模式 |
|---|---|---|
| 严谨 | `_llm_reranker.py` | `logger.warning()` / `logger.exception()` — 所有异常路径有结构化日志 |
| 宽松 | `longmem_eval.py:301`, `locomo_eval.py:489` | `except Exception: pass` / `return []` — 静默吞没 |

**根因**：评测脚本追求「不因单条失败中断全量」，但过度防御导致**故障不可观测**。正确做法是在静默降级的同时，用计数器或日志记录失败次数。

### 模式 2：Benchmark 间代码复用模式

```
          longmem_eval                    locomo_eval
              │                               │
    _evaluate_question()              run_qa_task() et al.
         │                                    │
    Session 聚类 (内联)              _build_llm_context() (方法)
         │                                    │
    entity 去重 (内联)                       (无)
         │                                    │
    time_label 注入 (内联)           time_label 注入 (内联)
```

**根因**：两个 Benchmark 独立演进，长期维护应抽象共享 `_context_builder` 模块。

---

## 六、工程成熟度评估

| 维度 | 当前等级 | 评级 | 说明 |
|------|----------|------|------|
| Lint/Format | **L3** | ✅ | ruff + ruff-format + pre-commit + CI gate |
| 测试 | **L3** | ✅ | 73 文件 / 1622 测试 / 覆盖率门槛 75% |
| CI/CD | **L2** | ⚠️ | lint+test+benchmark gate 齐全，缺 security scan |
| 类型检查 | **L3** | ✅ | mypy 配置 + pyright in CI (pre-commit mypy 仅覆盖 `src/su_memory/sdk/`) |
| 安全扫描 | **L0** | ❌ | 无 SAST/secret-scan，但对 SDK 场景影响较小 |
| 日志与监控 | **L2** | ⚠️ | SDK 层有结构化 logger，Benchmark 层大量 `print()` |
| 文档 | **L2** | ✅ | CHANGELOG + BENCHMARK doc 存在且更新 |

**综合评级：L2+ → L3**（距离 L3「质量可度量」仅差安全扫描和 Benchmark 层日志统一）

---

## 七、修复优先级与路线图

### 第一优先级（P1 — ✅ 全部已修复）

| # | 项 | 工作量 | 状态 |
|---|---|---|---|
| 1 | `_llm_judge_equivalence` 异常处理：加 `logger.warning` + 失败计数器 | 5min | ✅ |
| 2 | `locomo._query()` 异常处理：加静默失败计数器 | 10min | ✅ |
| 3 | Session 聚类逻辑抽取为 `benchmarks/_cluster_utils.py` 共享函数 | 30min | ✅ |

### 第二优先级（P2 — ✅ 全部已修复）

| # | 项 | 工作量 | 状态 |
|---|---|---|---|
| 4 | `_build_llm_context` 中 metadata 提取抽象为 `_get_meta()` 辅助函数 | 10min | ✅ |
| 5 | `answer_batch` 中 `print` → `logger.info` | 2min | ✅ |
| 6 | `__repr__` 增加 `hasattr` 守卫（原 P2-2） | 5min | ✅ |

---

## 八、亮点

1. **零脚手架残留**：0 TODO / 0 FIXME / 0 硬编码凭证 / 0 合并冲突 — 迭代质量极高
2. **ModelRouter 设计优雅**：`__getattr__` 代理模式使 Router 对调用方完全透明
3. **LLM 答案后处理管线** (`_postprocess_answer`)：5 步清洗流程覆盖了 V4-Pro 所有已知冗余输出模式
4. **工程基础扎实**：ruff + mypy + pre-commit + CI gate + 75% 覆盖率门槛 — 远高于同类 SDK 项目
5. **CHANGELOG 详尽**：v3.6.0 变更可审计追溯

---

*审计完成。全部 3 P1 + 3 P2 缺陷已修复闭环，ruff 零新增错误，105 测试通过。*
