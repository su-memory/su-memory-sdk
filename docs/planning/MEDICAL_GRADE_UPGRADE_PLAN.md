# su-memory SDK · 从「医疗专用」迈向「医疗级」KPI 修复计划书

> 版本：v1.0 · 日期：2026-07-17
> 定位依据：`docs/planning/POSITIONING_ANALYSIS.md`（第一性原理对照分析）
> 落点依据：6 个缺口均已代码级精确定位（文件:行号见各 Sprint）
> 执行原则：引擎核心零侵入、所有新代码在 `clinical/` + `fhir/` 下、区隔 MCI World Model

---

## 一、计划目标（一句话）

把 su-memory 从「医疗专用（specialized）」推进到「医疗级（medical-grade）」——
即不仅**功能覆盖**医疗场景，更在**召回质量、安全门控、合规审计**上达到可进医院的保证等级。

**修复后预期定位**：
- 召回：医疗同义召回率从 0% → ≥85%（可量化基准）
- 安全：recall 结果 100% 经风险门控（零禁忌泄露）
- 合规：每条记忆可溯源到原始来源（provenance 链完整）
- 质量：记忆抽取信噪比可量化（压缩比 + 事实保真度）

---

## 二、缺口全景与优先级

| 编号 | 缺口 | 现状 | 医疗要求 | 优先级 | 复杂度 |
|---|---|---|---|---|---|
| C5 | 来源溯源 provenance | MemoryNode 无 source 字段 | FDA SaMD 可溯源 | **P0** | 低 |
| C3 | 风险门控检索 | recall 不校验禁忌 | 禁忌零泄露 | **P0** | 中 |
| C1 | 医疗同义召回 | 同义词召回率 0% | 同义/近义命中 | **P1** | 中 |
| C2 | 记忆抽取层 | 入库即原文 | 结构化要点+引用 | **P1** | 高 |
| C4 | 双时间模型 | 单时间戳 | 事件时间vs记录时间 | **P2** | 中 |
| C6 | 版本化冲突消解 | 无版本链 | 诊断变更可回溯 | **P2** | 中 |

---

## 三、Sprint 分解（每个 Sprint 含：目标 / 落点 / 改动 / KPI / 验收 / 测试）

---

### Sprint-M0 · C5 来源溯源 provenance（P0，合规硬门槛）

**目标**：每条记忆可追溯来源（医嘱/检验报告/患者自述/AI 推断），满足 FDA SaMD 可溯源要求。

**落点**（已验证）：
- `src/su_memory/sdk/lite_pro.py:172` — `class MemoryNode`（加字段）
- `src/su_memory/sdk/lite_pro.py:2202` — `def add()`（透传 source）
- `src/su_memory/clinical/compliance.py:160` — `class AuditEntry`（加 source 字段）
- `src/su_memory/clinical/compliance.py:186` — `def log()`（记录来源链）
- `src/su_memory/clinical/client.py` — `add_patient_event()` 加 `source` 参数

**改动**：
1. `MemoryNode` 新增字段（向后兼容，默认值）：
   ```python
   source_type: str = "unknown"        # order|lab_report|patient|ai_inferred|imported
   source_id: str = ""                 # 原始记录 ID（病历号/对话ID/FHIR Resource ID）
   source_confidence: float = 1.0      # 来源可信度（医嘱=1.0, 患者自述=0.6, AI推断=0.4）
   ```
2. `add()` 签名加 `source_type`/`source_id` 可选参数，透传到 node
3. `ClinicalMemoryClient.add_patient_event()` 加 `source` 参数，自动映射
4. `AuditEntry` 加 `source_type`/`source_id` 字段，`log()` 记录来源链

**KPI**：
| 指标 | 现状 | 目标 | 验收方法 |
|---|---|---|---|
| 记忆含 source 字段率 | 0% | **100%**（新写入） | `rg "source_type" tests/` + 新增测试断言 |
| 审计日志含来源链 | 否 | **是** | `test_provenance.py` 断言 AuditEntry.source_type 非空 |
| 向后兼容 | — | **旧记忆 source_type="unknown"** | 现有 121 测试全绿 |

**验收**：新增 `tests/test_provenance.py`，断言写入→读取→审计三环 source 链完整；现有 121 测试无回归。

**预估工作量**：0.5 天

---

### Sprint-M1 · C3 风险门控检索（P0，安全关键）

**目标**：recall 返回前，所有结果经风险门控——禁忌药物交互/过敏冲突被标记或拦截，实现「禁忌零泄露」。

**落点**（已验证）：
- `src/su_memory/clinical/client.py:181` — `def recall()`（插入门控）
- `src/su_memory/clinical/knowledge.py:282/304/374` — `check_drug_interaction`/`get_contraindicated_nutrients`/`check_allergy`（复用现有能力）

**改动**（新建模块，零侵入引擎）：
1. 新建 `src/su_memory/clinical/safety_gate.py`：
   ```python
   class SafetyGate:
       """检索结果风险门控——召回后、返回前的一道安全校验。"""
       def screen(self, results: list[dict], patient_allergies: list[str] = None) -> list[dict]:
           # 1. 从每条记忆 content 提取药名（复用 knowledge 的 drug_name 子串匹配）
           # 2. 调 check_drug_interaction 查交互
           # 3. 调 check_allergy 查过敏冲突
           # 4. 给每条记忆打 risk_flags + risk_level (safe|caution|contraindicated)
           # 5. 默认策略：contraindicated 拦截（可配置为标记不拦截）
   ```
2. `recall()` 加 `safety_screen: bool = True` 参数，开启时经 SafetyGate
3. 返回结构新增字段：`risk_flags: list[str]`、`risk_level: str`

**KPI**：
| 指标 | 现状 | 目标 | 验收方法 |
|---|---|---|---|
| 禁忌泄露率 | 100%（不校验） | **0%** | `test_safety_gate.py` 注入华法林+维K记忆，断言被拦截 |
| 风险标记覆盖率 | 0% | **100%**（含药名的记忆） | 断言每条含药名记忆有 risk_level |
| 门控延迟开销 | — | **<2ms/条** | benchmark 断言 |
| 向后兼容 | — | `safety_screen=False` 时行为不变 | 现有测试全绿 |

**验收**：新增 `tests/test_safety_gate.py`，覆盖：禁忌拦截、过敏冲突标记、安全记忆放行、门控关闭降级。

**预估工作量**：1 天

---

### Sprint-M2 · C1 医疗同义召回（P1，先量化再优化）

**目标**：建立医疗同义召回基准，量化当前差距，再通过同义词扩展提升召回率。

**落点**（已验证）：
- `src/su_memory/sdk/lite_pro.py:1676` — `_tokenize()`（同义词扩展挂载点）
- `benchmarks/real_microbench.py`（新增同义召回基准）
- `tests/test_clinical_client.py`（新增召回质量测试）

**改动**（分两步：先测后修）：

**步骤 1 — 建基准（先量化）**：
1. 新建 `benchmarks/medical_synonym_bench.py`，内置 50 对中文医学术语同义词对：
   - 华法林↔warfarin、白蛋白↔albumin、禁忌症↔过敏/不耐受、
   - 糖尿病↔DM、高血压↔HTN、血红蛋白↔HbA1c ...
2. 基准逻辑：写入术语A，用同义术语B查询，统计召回率（recall@5）
3. 跑出当前基线数字（预期：关键词路 0%，向量路部分命中）

**步骤 2 — 同义词扩展**：
1. 新建 `src/su_memory/clinical/synonym_dict.py`：医学术语同义词典（可 load_from_file 扩展）
2. 在 `_tokenize()` 或新增 `_medical_expand()` 钩子：分词后查同义词典，扩展同义语素
3. 扩展后的语素进入倒排索引，提升关键词路召回

**KPI**：
| 指标 | 现状 | 目标 | 验收方法 |
|---|---|---|---|
| 医疗同义召回率 | **未测量（推测~0%）** | **≥85%**（50 对基准） | `python benchmarks/medical_synonym_bench.py` |
| 精确率不劣化 | — | **≥90%**（同义扩展不引入噪音） | 同基准测 precision@5 |
| 向后兼容 | — | 非医疗查询召回不降 | 现有测试全绿 |

**验收**：基准脚本输出 recall@5/precision@5 数字；新增 `tests/test_synonym_recall.py` 断言 ≥5 个典型同义对命中。

**预估工作量**：1.5 天

---

### Sprint-M3 · C2 记忆抽取层（P1，信噪比提升）

**目标**：入库前把长原文压缩为结构化要点 + 原文引用，提升信噪比，降低 token 成本。

**落点**（已验证）：
- `src/su_memory/sdk/lite_pro.py:2202` — `add()`（当前直接存原文）
- 无独立 extraction 模块（全仓确认）

**改动**（新建模块，clinical 层，不改引擎 add）：
1. 新建 `src/su_memory/clinical/extractor.py`：
   ```python
   class ClinicalMemoryExtractor:
       """入库前抽取——长原文→结构化要点 + 原文引用。"""
       def extract(self, content: str, source_type: str) -> ExtractedFact:
           # 1. 规则抽取（药名/剂量/检验值/诊断 正则 + 字典）
           # 2. 可选 LLM 抽取（opt-in，有 LLM 时用，无则降级规则）
           # 3. 返回：{summary, entities[], original_ref, confidence}
   ```
2. `ClinicalMemoryClient.add_patient_event()` 加 `extract: bool = False` 参数
3. 抽取后存 summary 到 content，原文存 metadata.`_original_content`

**区隔声明**：抽取是「信息压缩」，不是「因果推断」；事实保真由规则+引用保证，不做医学推理。

**KPI**：
| 指标 | 现状 | 目标 | 验收方法 |
|---|---|---|---|
| 压缩比（原文/摘要） | 1.0（不压缩） | **≥3.0**（长病历） | `test_extractor.py` 测典型病历 |
| 事实保真度（关键实体不丢） | — | **≥95%** | 断言药名/剂量/检验值抽取后仍可检索 |
| LLM 降级 | — | **无 LLM 时规则兜底** | 无网络环境冒烟测试 |

**验收**：新增 `tests/test_extractor.py`，覆盖：规则抽取、LLM 降级、原文引用保留、关键实体保真。

**预估工作量**：2 天

---

### Sprint-M4 · C4 双时间模型（P2，时序准确性）

**目标**：区分「事件发生时间」与「记录入库时间」，修正时间衰减错误。

**落点**（已验证）：
- `src/su_memory/sdk/lite_pro.py:172` — `MemoryNode.timestamp`（单字段）
- `src/su_memory/sdk/lite_pro.py:2330` — spacetime 调用点（传入入库时间）
- `src/su_memory/sdk/spacetime_index.py:383` — `_calculate_time_decay`（基于 timestamp）

**改动**：
1. `MemoryNode` 新增 `event_time: int = 0`（事件发生时间，缺省=入库时间）
2. `add()` 加 `event_time` 可选参数
3. spacetime 索引优先用 `event_time`（无则降级 `timestamp`）
4. `ClinicalMemoryClient.add_lab_value()` 支持传检验发生时间

**KPI**：
| 指标 | 现状 | 目标 | 验收方法 |
|---|---|---|---|
| 事件时间可独立设置 | 否 | **是** | `test_dual_time.py` |
| 时间衰减基于事件时间 | 否（入库时间） | **是** | 断言旧事件录入后衰减正确 |
| 向后兼容 | — | event_time 缺省=timestamp | 现有测试全绿 |

**验收**：新增 `tests/test_dual_time.py`，覆盖：事件时间设置、衰减基于事件时间、缺省降级。

**预估工作量**：1 天

---

### Sprint-M5 · C6 版本化冲突消解（P2，诊疗变更可回溯）

**目标**：同一患者同一事实多次更新时，建立版本链，可回溯历史版本。

**落点**（已验证）：
- `src/su_memory/sdk/lite_pro.py:3926` — `consolidate()`（单向归并，无版本链）
- `MemoryNode` 无 version 字段

**改动**：
1. `MemoryNode` 新增 `version: int = 1`、`prev_version_id: str = ""`、`superseded_by: str = ""`
2. 新建 `src/su_memory/clinical/versioning.py`：
   ```python
   class ClinicalVersionChain:
       """同一患者同一事实的版本链管理。"""
       def update_fact(self, patient_id, fact_key, new_content) -> str:
           # 1. 查当前活跃版本（superseded_by=""）
           # 2. 新建版本，prev_version_id 指向旧版本
           # 3. 旧版本 superseded_by 指向新版本
       def get_history(self, fact_key) -> list[dict]:  # 版本链回溯
       def get_active(self, fact_key) -> dict | None:  # 当前生效版本
   ```
3. `ClinicalMemoryClient` 加 `update_clinical_fact()` 方法

**KPI**：
| 指标 | 现状 | 目标 | 验收方法 |
|---|---|---|---|
| 版本链可回溯 | 否 | **是** | `test_versioning.py` 断言 history 长度 |
| 旧版本不被覆盖 | 否（consolidate 吞并） | **是** | 断言历史版本仍可查 |
| 活跃版本正确 | — | **superseded_by="" 的最新版** | 断言 get_active 返回最新 |

**验收**：新增 `tests/test_versioning.py`，覆盖：版本更新、历史回溯、活跃版本、不与 consolidate 冲突。

**预估工作量**：1.5 天

---

## 四、执行顺序与依赖

```
Sprint-M0 (C5 provenance)  ──┐
                              ├──→ Sprint-M1 (C3 风险门控，依赖 source_confidence)
                              │
Sprint-M2 (C1 同义召回)  ─────┘ (独立，可并行)

Sprint-M3 (C2 抽取)  ──→ 依赖 M0 的 source（抽取结果要标来源）

Sprint-M4 (C4 双时间)  ──→ 独立
Sprint-M5 (C6 版本链)  ──→ 独立
```

**推荐批次**：
- **批次 1（P0 安全合规）**：M0 → M1（先 provenance 再门控，门控用 source_confidence）
- **批次 2（P1 质量基线）**：M2（同义召回）+ M3（抽取），可并行
- **批次 3（P2 时序与版本）**：M4 + M5，可并行

---

## 五、总体 KPI 汇总（修复后验收清单）

| 维度 | KPI | 目标值 | 验收手段 |
|---|---|---|---|
| 合规 | 记忆 source 字段率 | 100% | test_provenance.py |
| 合规 | 审计日志含来源链 | 是 | test_provenance.py |
| 安全 | 禁忌泄露率 | 0% | test_safety_gate.py |
| 安全 | 风险标记覆盖率 | 100% | test_safety_gate.py |
| 召回 | 医疗同义召回率 | ≥85% | medical_synonym_bench.py |
| 召回 | 精确率不劣化 | ≥90% | medical_synonym_bench.py |
| 质量 | 抽取压缩比 | ≥3.0 | test_extractor.py |
| 质量 | 事实保真度 | ≥95% | test_extractor.py |
| 时序 | 事件时间可独立 | 是 | test_dual_time.py |
| 版本 | 版本链可回溯 | 是 | test_versioning.py |
| 工程 | 现有测试无回归 | 121 全绿 | pytest |
| 工程 | 新增测试覆盖 | 6 个新测试文件 | pytest |
| 工程 | ruff/mypy | 0 errors | ruff check + mypy |

---

## 六、区隔约束（再次明确，不可逾越）

| 能力 | 归属 | 本计划是否触碰 |
|---|---|---|
| 记忆存储/检索/关联 | **su-memory（本仓库）** | ✅ 在 clinical/ 下扩展 |
| 因果推断/do-calculus | **MCI World Model** | ❌ 不碰 |
| Agent 编排/思考 | **MCI-SDK** | ❌ 不碰 |
| 临床营养业务逻辑 | **调用方项目** | ❌ 不碰 |

C3 风险门控只做「禁忌标记/拦截」，不做「因果风险预测」；
C2 抽取只做「信息压缩」，不做「医学推理」；
C6 版本链只做「事实变更回溯」，不做「诊断推理」。

---

## 七、风险与降级

| 风险 | 降级方案 |
|---|---|
| M2 同义词典不全 | 支持 load_from_file，医院可自维护 |
| M3 LLM 抽取不可用 | 规则抽取兜底（无网络也能跑） |
| M1 误拦截（假阳性） | `safety_screen=False` 可关闭，`policy=mark` 只标记不拦截 |
| M0 旧记忆无 source | source_type="unknown" 降级，不阻塞 |

---

## 八、预估总工作量

| 批次 | Sprint | 工作量 |
|---|---|---|
| 1 | M0 + M1 | 1.5 天 |
| 2 | M2 + M3 | 3.5 天 |
| 3 | M4 + M5 | 2.5 天 |
| **合计** | **6 个 Sprint** | **~7.5 天** |

修复完成后，su-memory 将从「医疗专用」正式迈入「医疗级」——
这是从「功能列表领先」到「安全保证等级领先」的质变，也是进医院的硬门槛。
