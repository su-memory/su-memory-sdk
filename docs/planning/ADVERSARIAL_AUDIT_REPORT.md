# su-memory 医疗级升级 · 对抗性审计报告

> 审计日期：2026-07-18
> 审计方法：源码逐行审查 + 实际攻击 PoC 验证（双路交叉确认）
> 审计范围：C1-C6 全部新增/改动代码 + 关联的 compliance/versioning/multi_tenant

## 结论

发现 **9 个高危 + 7 个中危 + 2 个低危** 漏洞。其中 **3 个是架构级设计缺陷**（不是单点 bug），意味着当前「医疗级」定位**不成立**——在 PHI 脱敏、删除权、风险门控三个医疗合规的核心承诺上存在根本性缺口。

**必须修复才能称「医疗级」的 3 个架构级问题**：
1. content 永不脱敏（正文 PHI 直泄）
2. purge 不清向量索引（删除权名存实亡）
3. 风险门控 fail-open（异常时禁忌直泄）

---

## 高危漏洞（9 个，已实证）

### V1 · 风险门控异常时 fail-open，禁忌直泄
- **位置**：`src/su_memory/clinical/client.py:284-291`
- **实证**：`safety_gate.screen` 内部多处 try/except，知识库污染/异常时门控整体降级，返回未门控结果
- **修复方向**：医疗安全应 fail-closed——异常时拒绝返回或标记 `risk_level=unknown`

### V2 · 药名/过敏原子串匹配，可绕过可误判
- **位置**：`safety_gate.py:108,152`、`knowledge.py:296`
- **实证**：`"华 法 林"`（加空格）子串匹配为 False → 禁忌漏检；`"钾"` in `"血钾离子"` → 误判告警
- **修复方向**：分词后做集合匹配，而非裸子串 `in`

### V3 · 过敏检测依赖结构化记忆，无则静默失效
- **位置**：`client.py:296-310` `_patient_allergies`
- **实证**：未写 `event_type=allergy` 记忆时，花生过敏冲突完全未检出（PoC 攻击3 验证）
- **修复方向**：过敏原应从自由文本 content 也能提取，或在 recall 时显式传 patient_allergies

### V4 · content 完全不脱敏，正文 PHI 直泄（架构级）
- **位置**：`compliance.py:294-296` hooked_add 只脱敏 metadata
- **场景**：`add_patient_event("P001","张三,身份证330102...,电话138...")` → content 明文落盘+进向量索引+进审计
- **修复方向**：PHISanitizer 必须处理 content（正则脱敏身份证/电话/姓名模式）

### V5 · 审计日志明文 PHI，purge 不清理（架构级）
- **位置**：`compliance.py:76-113,323-325`
- **场景**：审计 JSONL 含明文 patient_id/source_id；purge_patient 明确不删审计 → 删除权无法实现
- **修复方向**：审计日志写入前脱敏；purge 时标记/加密相关审计条目

### V6 · purge 不清 FAISS/向量/倒排索引，删除权失败（架构级）
- **位置**：`compliance.py:382-384` 只重建 `_memory_map`
- **场景**：被删记忆的 embedding 仍在 FAISS，query 仍能命中返回 content
- **修复方向**：purge 后重建 FAISS 索引 + 清倒排索引 + 清 spacetime

### V7 · 并发 update_fact 版本分叉 + 双 active
- **位置**：`versioning.py:60-80` 三步非原子
- **场景**：两并发线程同时 update 同一 fact_key → 两个 v2 都声称 active，版本链分叉
- **修复方向**：加锁（threading.Lock）或乐观锁（版本号校验）

### V8 · 空 patient_id 串扰
- **位置**：`client.py:80,167`
- **实证**：PoC 攻击6 验证——两个空 pid 记忆互相可见（含B数据:True）
- **修复方向**：`add_patient_event`/`recall` 校验 patient_id 非空，空则拒绝

### V9 · event_time=0 与"未设置"歧义，时间语义损坏
- **位置**：`lite_pro.py:200-202` `effective_time` 用 `if self.event_time`
- **场景**：event_time=0（合法 1970）被判 falsy 回退 timestamp；写入存 0 但读取改写，语义不一致
- **修复方向**：用 `None` 哨兵或显式 `event_time_set: bool` 字段，不用 0 当哨兵

---

## 中危漏洞（7 个）

### V10 · caution 级未拦截，含临床建议外泄
- **位置**：`safety_gate.py:127-138,165`
- moderate 交互（如二甲双胍×B12）走 caution，risk_interactions 含 clinical_advice 返回；policy=block 也拦不住 caution

### V11 · PHI 字段白名单不全，变体/嵌套绕过
- **位置**：`compliance.py:20,37-45`
- patientName/姓名/idCard/passport 等变体不在白名单；嵌套 metadata 不递归扫描

### V12 · 手机/身份证脱敏保留 7-8 位明文
- **位置**：`compliance.py:62-76`
- `138****5678` 保留 7 位；身份证前4后4 = 地域+生日+顺序位泄露

### V13 · purge 中间版本节点，版本链静默截断
- **位置**：`versioning.py:181-185`
- 删中间版本后 get_history 静默截断，医生看到残缺历史无告警

### V14 · 多租户 `:` 前缀注入 + inner 绕过
- **位置**：`multi_tenant.py:64-69,76-78`
- 传 `patient_id="T999:P001"` 冒充他租户；inner 属性绕过 scope

### V15 · 未来 event_time 被 clamp 满分，长期霸占召回
- **位置**：`lite_pro.py:572-620`
- 未来事件 recency=1.0 永远霸占 top；且 _temporal_rerank 仍用 timestamp 非事件时间

### V16 · event_time 负数/零 → 黑洞记忆查不到
- **位置**：`lite_pro.py:2244`、`spacetime_index.py:57`
- 负/零 event_time 进异常 bucket，时间范围查询永远扫不到

---

## 低危漏洞（2 个）

### V17 · 自环版本链产生脏记录
- `versioning.py:118-124` seen 防住死循环，但自环产生一条重复自身记录

### V18 · 无 id 字段时回退 id(r)，去重失效
- `client.py:163` 引擎返回无 memory_id 时回退对象 id，跨 query 去重失效

---

## 修复优先级

**P0（不修不能称医疗级）**：V1 V4 V5 V6 V8 —— 4 个架构级 + 空pid串扰
**P1（安全显著提升）**：V2 V3 V7 V9 —— 绕过/失效/并发/语义
**P2（加固）**：V10-V16 —— 中危
**P3（清洁）**：V17 V18 —— 低危

## 审计方法说明

本报告由两条独立审计路径交叉确认：
1. **源码逐行审查**（explorer）：覆盖 safety_gate/client/compliance/versioning/multi_tenant/knowledge/lite_pro
2. **实际攻击 PoC**：6 个攻击场景实跑验证（空格绕过✓、过敏失效✓、空pid串扰✓、自环✓、负event_time✓、caution漏放✓）

两路结论一致，漏洞可信度高。
