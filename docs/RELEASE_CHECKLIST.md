# su-memory SDK v2.0 — 发布前测试检查流程

> 本文档定义 su-memory SDK 发布前必须通过的完整测试标准。
> 测试维度：功能验收、性能达标、持久化正确、边界安全、文档一致。

---

## 一、测试前置条件

### 1.1 环境要求

```bash
# 必须通过 pip install -e . 安装（而非本地 src 路径）
pip show su-memory          # 版本显示正常
python -c "import su_memory; print(su_memory.__file__)"  # 确认来自 site-packages

# 依赖验证
python -c "import su_memory; from su_memory._sys.encoders import SemanticEncoder; print('encoder OK')"
```

### 1.2 测试数据准备

```python
# 标准测试数据集（必须使用这组数据，确保可复现）
TEST_MEMORIES = [
    ("Nutri-Brain 目标成为三甲医院营养科首选 AI 供应商", {"source": "strategy"}),
    ("首轮融资目标 500 万元，估值不超过 2000 万", {"source": "finance"}),
    ("核心算法基于代谢组学 + 大语言模型", {"source": "tech"}),
    ("Q3 计划对接 3 家三甲医院营养科", {"source": "milestone"}),
    ("团队目前 5 人，CEO 有临床营养背景，CTO 来自医疗 AI 独角兽", {"source": "team"}),
    ("投资人关注：市场规模、差异化、合规路径", {"source": "investor"}),
    ("竞品定价过高，医院采购决策周期长", {"source": "market"}),
    ("营养科主任最关注患者依从性数据", {"source": "feedback"}),
    ("需要补齐 II 类医疗器械注册证", {"source": "compliance"}),
    ("已建立临床营养知识图谱，覆盖 200+ 病种方案", {"source": "knowledge"}),
]

TEST_QUERIES = {
    "融资": ["融资", "估值", "投资"],          # 应命中 finance/investor
    "医院": ["医院", "营养科", "三甲"],          # 应命中 milestone/market
    "技术": ["算法", "模型", "技术"],            # 应命中 tech/knowledge
    "合规": ["医疗器械", "注册证", "合规"],      # 应命中 compliance
}
```

---

## 二、功能验收测试（必须 100% 通过）

### 2.1 核心 API 测试

| ID | 测试项 | 验收标准 | 严重级别 |
|----|-------|---------|---------|
| F-01 | `import su_memory` | 无 ImportError，无警告 | P0 |
| F-02 | `SuMemory()` 初始化 | 不抛异常，返回有效对象 | P0 |
| F-03 | `add()` 写入记忆 | 返回 memory_id，长度+1 | P0 |
| F-04 | `query()` 检索记忆 | 返回非空列表，按 score 降序 | P0 |
| F-05 | `link()` 建立关联 | 返回 True，两条记忆关联建立 | P1 |
| F-06 | `query_multihop()` 多跳 | 返回结果列表，包含多条记忆 | P1 |
| F-07 | `get_stats()` 统计 | 返回 total_memories、分布 | P2 |
| F-08 | `delete()` 删除 | 指定 ID 删除成功，数量正确 | P1 |

### 2.2 检索质量验收

> 验收方法：写入标准测试数据后，用每个 query 检索，Top-3 结果必须**语义相关**。

```python
# 判定规则
def is_semantically_related(query_key: str, content: str) -> bool:
    keywords = TEST_QUERIES.get(query_key, [])
    return any(kw in content for kw in keywords)

# 验收：每个 query 的 Top-3 中至少 2 条相关
# 修复后需通过率 ≥ 80%
```

**P0 标准（必须满足）：**
- 不同 query 的 Top-1 结果**不能完全相同**
- Top-1 结果必须与 query 语义相关（可直接含 query 关键词）

### 2.3 多维标签验收

```python
# 验证每条记忆的 encoding 包含有效标签
result = client.query("测试", top_k=1)[0]
assert hasattr(result.encoding, 'bagua')      # 标签存在
assert hasattr(result.encoding, 'wuxing')      # 属性存在
assert result.encoding.bagua in ["乾","坤","震","巽","坎","离","艮","兑"]  # 枚举合法
assert result.encoding.wuxing in ["金","木","水","火","土"]                  # 枚举合法
```

---

## 三、持久化测试（必须通过）

### 3.1 重启后记忆不丢失

```python
DATA_DIR = "./test_persist"
client = SuMemory(persist_dir=DATA_DIR)
ids = [client.add(m, metadata=meta) for m, meta in TEST_MEMORIES]

# 模拟重启：重新初始化同一目录
client2 = SuMemory(persist_dir=DATA_DIR)
assert client2.get_stats()["total_memories"] == len(TEST_MEMORIES), "P0: 重启后记忆丢失！"
```

**P0 — 任何情况下不能丢数据。**

### 3.2 并发安全（写入后立即读取）

```python
client = SuMemory(persist_dir="./test_concurrent")
mid = client.add("唯一标识内容 XYZABC")
result = client.query("XYZABC", top_k=1)
assert result[0].content == "唯一标识内容 XYZABC"  # P0
```

### 3.3 数据损坏恢复

```python
# 模拟 JSON 损坏：写入非法 JSON
import os, json
data_path = os.path.join(client.persist_dir, "memories.json")
with open(data_path, "w") as f:
    f.write("{ invalid json }")

# 重启应该能正常降级，不崩溃
client = SuMemory(persist_dir=client.persist_dir)
assert len(client) == 0  # 降级清空，但不抛异常
```

---

## 四、性能基准测试

### 4.1 延迟标准（单次操作 P50）

| 操作 | 达标阈值 | 严重级别 |
|------|---------|---------|
| encode 语义编码 | < 15ms | P0 |
| SDK 写入 add | < 150ms | P0 |
| SDK 检索 query（100条记忆） | < 100ms | P0 |
| 全息检索 | < 5ms | P1 |
| 象压缩 | < 10ms | P1 |

**测试方法：写入 100 条记忆后，测量 query P50（取 20 次中位数）**

### 4.2 吞吐量标准

| 操作 | 达标阈值（并发 10） | 严重级别 |
|------|---------|---------|
| 编码吞吐量 | > 100 QPS | P1 |
| SDK 写入吞吐量 | > 50 QPS | P1 |
| SDK 检索吞吐量 | > 20 QPS | P1 |

### 4.3 资源占用标准

| 规模 | 达标阈值 | 严重级别 |
|------|---------|---------|
| 1K 条记忆 | < 800MB | P0 |
| 10K 条记忆 | < 1.5GB | P0 |

---

## 五、边界条件测试

| ID | 场景 | 验收标准 | 级别 |
|----|-----|---------|------|
| B-01 | 空查询 `query("")` | 不崩溃，返回空列表 | P0 |
| B-02 | 空内容 `add("")` | 不崩溃，返回 memory_id | P1 |
| B-03 | 超长内容（10万字） | 不崩溃，截断或正常处理 | P1 |
| B-04 | 纯 Unicode 特殊字符 | 不崩溃，正常编码 | P2 |
| B-05 | 中英混合内容 | 正常处理 | P1 |
| B-06 | 重复 add 相同内容 | 不报错，生成不同 ID | P1 |
| B-07 | query 不存在记忆 | 返回空列表，不报错 | P0 |
| B-08 | delete 不存在的 ID | 不崩溃，返回 0 | P2 |

---

## 六、Hindsight 12 场景对标测试

> 每场景必须有独立测试用例，测试数据需与 Hindsight 原始场景对齐。

| # | 场景 | 验证方式 | 必须输出 |
|---|------|---------|---------|
| 1 | 单跳检索 | query 精确关键词 | Top-1 包含 query 关键词 |
| 2 | 多跳推理 | query_multihop 跨记忆问题 | 返回 ≥2 条不同 ID 的记忆 |
| 3 | 时序理解 | 写入含时间信息记忆后查询 | 时间相关记忆排序靠前 |
| 4 | 多会话 | 重启后 query | 重启后数据完整 |
| 5 | 开放域 | 非结构化自由文本 | 正常编码，不抛异常 |
| 6 | 全息检索 | query → 对比向量+全息评分 | 两者评分趋势一致 |
| 7 | 象压缩 | 压缩率测试 | 压缩后内容可还原 |
| 8 | 因果推理 | link 两记忆后 multi_hop | 关联记忆出现在结果中 |
| 9 | 时序权重 | 老记忆 vs 新记忆 query | 新记忆 score 更高（时间衰减） |
| 10 | 元认知 | 检测认知间隙 | 返回非空检测结果 |
| 11 | 可解释性 | query 结果 | 每条结果包含 score 来源说明 |
| 12 | 综合对比 | 生成对比报告 | 报告包含 12 维度评分 |

---

## 七、发布阻断标准（P0 必须全部满足）

> 以下任一项未通过，**禁止发布**。

```
[ ] F-01 ~ F-04 核心 API 全部通过
[ ] 检索质量：不同 query 的 Top-1 结果不完全相同
[ ] P0 性能指标全部达标
[ ] 持久化：重启后记忆不丢失
[ ] 边界条件：P0 场景（B-01, B-07）不崩溃
[ ] 297 项测试用例 100% 通过（pytest）
```

---

## 八、测试执行脚本

```bash
# 完整执行
cd ~/.openclaw/workspace/su-memory
pytest tests/ -v --tb=short

# 快速冒烟测试（3分钟）
pytest tests/test_sdk.py tests/test_persistence.py -v

# 性能基准测试
pytest tests/test_benchmark.py -v -s

# Hindsight 对标测试
pytest tests/test_hindsight_comparison.py -v
```

---

## 九、当前问题追踪

### 已修复 ✅

| 问题 | 修复内容 | 修复日期 |
|------|---------|---------|
| 记忆无法持久化 | 实现 `_save()` / `_load()`，每次 add 立即落盘 | 2026-04-23 |
| 检索候选集粗暴扩展全量 | 改为向量相似度补充，最小候选集保护 | 2026-04-23 |
| keyword bonus 均质化 | 上限封顶 0.05，防止短 query 评分趋同 | 2026-04-23 |

### 待优化 ⚠️

| 问题 | 原因 | 优先级 | 备注 |
|------|------|--------|------|
| 检索结果同质化 | 语义编码器多样性不足（10条仅归3个类别） | P1 | Phase 3 优化方向 |
| 向量存储占用大 | JSON 明文存储 float list（10条~1MB） | P2 | 可考虑 numpy 压缩或 SQLite blob |

---

*文档版本：v1.0 | 更新日期：2026-04-23*
*维护责任人：小源*
