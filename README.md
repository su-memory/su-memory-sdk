---
language:
  - zh
  - en
license: apache-2.0
tags:
  - semantic-memory
  - vector-search
  - causal-reasoning
  - rag
  - temporal-awareness
  - llm
  - python
  - chinese
  - retrieval-augmented-generation
  - local-first
  - error-handling
  - fallback
datasets:
  - su-memory/demo-data
library_name: su-memory
pypi: su-memory
---

# su-memory SDK v4.0 · Semantic Memory Engine

> **"你的 AI 记不住上次聊过什么？su-memory 给它一个不会忘的大脑。"**
>
> **"为什么这条建议？——点击查看完整推理链。"**
>
> v4.0：统一单一产品线（取消 Lite/LitePro 分级，全部能力释放）· 三路融合 MultiHopReader + 本地 LLM reader (MLX Qwen) · algebra 数学层（GF(2)³/概率图/群论）· 三路融合 MultiHopReader · FAISS + bge-m3 原生 batch
>
> **统一引擎**：`SuMemory` 一个类含全部能力（向量语义检索 + 多跳推理 + 因果推理 + 时空关联）。`SuMemoryLite`/`SuMemoryLitePro` 为向后兼容别名。

---

## 📊 性能基准（真实 HotpotQA，可复现）

> **真实可复现（标准 EM 口径）**（官方 HotpotQA validation，200 题，全 hard level，标准 EM 口径：reader 抽取 span == gold）：

| 系统 | HotpotQA 标准 EM | F1 |
|---|:---:|:---:|
| **su-memory v4.0 (DeepSeek + CausalDAG桥接)** | **62.5%** | **75.4%** |
| su-memory v4.0 (本地 7B reader) | 48.0% | 58.6% |
| Hindsight | 70.83% | — |
| IRRR + BERT | 55.0% | — |
| DFGN (pure retrieval) | 48.2% | — |

- su-memory 最强配置（DeepSeek reader + CausalDAG 罕见实体桥接）标准 EM **62.5%**、F1 **75.4%**，**真实超越 DFGN（48.2%）与本地 7B（48%）约 14 个百分点**；comparison 题 73.5% **已超 Hindsight**；CausalDAG 桥接发现率 90%（vs title 匹配 44%）；本地 7B reader 标准 EM 48.0%（轻量回退）
- **检索能力 SOTA 级**：支持事实双召回 Full@5 = 95%，答案词 94% 原样出现在召回段落（纯算法 span 抽取上限 ~89%）
- 复现：`python benchmarks/hotpotqa_full_eval.py`（标准 EM，自动用本地 MLX Qwen reader；`--no-llm` 回退启发式）

> **历史诚实声明**：
> - v3.x 曾宣称「58% 超 SOTA」——经核实为合成数据自测，已删除。
> - v4.0 早期曾宣称「82.5% 超 Hindsight」——经核实为**召回覆盖口径**（gold 词出现在召回段落，非 reader 精确抽取），与官方 EM 口径不符，**属口径虚标，已修正为标准 EM 48.0%**。
> - 当前最强配置（DeepSeek-V3 + CausalDAG 桥接）标准 EM 62.5% / F1 75.4%，**真实超越 DFGN(48.2%)与本地7B(48%)约14点，comparison题73.5%已超Hindsight**；CausalDAG罕见实体桥接发现率90%（vs title匹配44%，algebra层真实能力）；bridge题60.2%仍拖累整体，距Hindsight 70.83%差~8点。
> - 已系统验证所有路径（7B/DeepSeek/GLM reader × 直抽/多跳推理/CausalDAG桥接标注/span对齐/refine/self-consistency）。第一性原理分析表明：14.5点边界损失（F1 75.4% - EM 62.5%）源于 LLM 生成式答案与严格 EM 字符串匹配的固有不对齐（gold边界主观，无规则可循），不可被后处理消除。**Hindsight 70.83% 的优势来自专门的多跳架构微调**，真实突破需专属多跳模型微调或更强旗舰模型。
>
> 微基准与更多数字见 [BENCHMARK.md](BENCHMARK.md)（`python benchmarks/real_microbench.py` 复现）。

---

## 📚 文档

| 资源 | 说明 |
|------|------|
| [API 文档](https://su-memory.readthedocs.io) | Sphinx 自动生成的完整 API 参考 |
| [异常体系](src/su_memory/exceptions.py) | 42 ErrorCode 错误码速查 |
| [降级矩阵](docs/fallback-matrix.md) | 7 组件降级路径全景 |
| [迁移指南](docs/MIGRATION_v2.5_to_v2.6.md) | v2.5 → v2.6 迁移步骤 |
| [性能基准](BENCHMARK.md) | 真实可复现性能（honest benchmark） |
| [更新日志](CHANGELOG.md) | Keep a Changelog 格式 |

---

## ⚡ 安装

```bash
pip install su-memory
```

**一行代码，让 AI 拥有记忆能力：**

```python
from su_memory import SuMemory

client = SuMemory()
client.add("张总在周一会议上提到Q3目标增长25%")
results = client.query("Q3目标")  # 秒级返回，带推理路径
```

---

## ⚡ 安装指南

### 环境要求

- Python 3.10+
- 推荐使用虚拟环境 (venv) 或 conda

### 安装前检查

**重要**: 安装前请确认 `pip` 和 `python` 指向同一环境。

```bash
# 检查环境一致性
which python
which pip

# 如果不一致，使用以下方式安装
python -m pip install su-memory
```

### 安装方式

#### 方式1: 标准安装 (推荐)

```bash
pip install su-memory
```

> ✨ **开箱即用多跳推理** - 默认集成FAISS + sentence-transformers

#### 方式2: 使用 python -m pip (确保环境一致)

```bash
python -m pip install su-memory
```

#### 方式3: 从 GitHub 安装最新版本

```bash
pip install git+https://github.com/su-memory/su-memory-sdk.git
```

#### 方式4: 源码安装

```bash
git clone https://github.com/su-memory/su-memory-sdk.git
cd su-memory-sdk
pip install .
```

#### 方式5: 开发模式安装

```bash
git clone https://github.com/su-memory/su-memory-sdk.git
cd su-memory-sdk
pip install -e ".[dev]"
```

### 可选依赖

| 安装选项 | 命令 | 包含 |
|---------|------|------|
| **标准版** | `pip install su-memory` | ⭐ 核心 + FAISS + sentence-transformers |
| **完整版** | `pip install su-memory[full]` | + 向量存储 (Qdrant/SQLAlchemy) |
| **Dashboard** | `pip install su-memory[dashboard]` | + Flask可视化界面 |
| **REST API** | `pip install su-memory[api]` | + FastAPI + uvicorn |

```bash
# 标准版即包含多跳推理能力
pip install su-memory

# 可视化Dashboard
pip install su-memory[dashboard]
python -m su_memory.dashboard
# 访问 http://localhost:8765

# REST API（支持 JS/Go/curl 调用）
pip install su-memory[api]
uvicorn su_memory.api.server:app --reload --port 8000
# 访问 http://localhost:8000/docs 查看 API 文档
```

### 安装验证

安装完成后，运行验证脚本:

```bash
# 快速检查
python -c "from su_memory import SuMemoryLitePro; print('✅ 安装成功')"

# 完整验证
python -c "from su_memory.verify_install import main; main()"
```

### 常见问题排查

#### 问题1: ModuleNotFoundError

```
pip show su-memory  # 显示已安装
python -c "import su_memory"  # 报错
```

**原因**: pip 和 python 指向不同环境

**解决**:
```bash
python -m pip install --force-reinstall su-memory
```

#### 问题2: 环境不匹配警告

```
⚠️ pip 和 python 指向不同环境
```

**解决**:
```bash
# 方式1: 使用 python -m pip
python -m pip install su-memory

# 方式2: 创建虚拟环境
python -m venv myenv
source myenv/bin/activate
pip install su-memory
```

#### 问题3: 诊断工具

如果遇到其他问题，运行诊断工具:

```bash
python -c "from su_memory.diagnostics import main; main()"
```

---

## 🚀 快速开始

| 能力 | 用户感知价值 | 技术支撑 |
|------|-------------|----------|
| **记住一切** | 上周聊的项目，AI秒级回忆 | 本地向量存储 |
| **因果线索检测** | "为什么推荐这个？" | 基于中文连接词的关键词模式匹配（非统计因果推断） |
| **时间感知** | 越新的记忆越相关 | 时序衰减 |
| **可解释** | 推理路径透明可见 | Multi-hop RAG |

---

### 一行代码入门

```python
from su_memory import SuMemory

# 初始化（开箱即用多跳推理）
client = SuMemory()

# 添加记忆
client.add("用户偏好深色主题", metadata={"user": "alice"})
client.add("用户上周购买了笔记本电脑")

# 语义检索
results = client.query("电脑")

# 多跳推理（默认hybrid模式，向量+图谱融合）
chain = client.query_multihop("用户的购买偏好", max_hops=3)
```

### 推荐入口

| 类 | 场景 | 说明 |
|-----|------|------|
| **SuMemory** | ⭐推荐 | 一行代码，本地运行，简单易用 |
| SuMemoryLite | 轻量场景 | 内存<50MB |
| SuMemoryLitePro | 专业场景 | 向量推理+多跳 |

# 添加记忆
client.add("今天天气很好，阳光明媚")
client.add("明天可能下雨，记得带伞")
client.add("我喜欢学习编程")

# 查询记忆
results = client.query("天气", top_k=2)
for r in results:
    print(f"{r['content']} (score: {r['score']})")
```

### 增强版 Pro

```python
from su_memory.sdk import SuMemoryLitePro

# 创建增强版客户端
pro = SuMemoryLitePro(
    storage_path="./data",
    embedding_backend='ollama',  # 使用本地Ollama bge-m3
    enable_vector=True,
    enable_graph=True,
    enable_temporal=True,
    enable_session=True,
    enable_prediction=True,
    enable_explainability=True
)

# 添加记忆
pro.add("如果努力学习，成绩会提高")
pro.add("成绩提高了会获得奖学金")
pro.add("获得奖学金可以减轻家庭负担")

# 建立因果链
pro.link_memories(pro._memories[-3].id, pro._memories[-2].id)
pro.link_memories(pro._memories[-2].id, pro._memories[-1].id)

# 多跳推理查询
results = pro.query_multihop("学习", max_hops=3)
for r in results:
    print(f"{r['content']} (hops={r['hops']})")

# 时序预测
predictions = pro.predict(query="项目活动")
print(predictions)

# 可解释性查询
explanation = pro.explain_query("学习", results)
print(explanation['explanation'])
```

### 与LangChain集成

```python
from su_memory.sdk import SuMemoryLite
from su_memory.adapters import SuMemoryChatMemory

# 创建记忆客户端
client = SuMemoryLite()
memory = SuMemoryChatMemory(client=client)

# 保存对话上下文
memory.save_context(
    inputs={"input": "我叫张三"},
    outputs={"output": "你好张三，很高兴认识你！"}
)

# 加载记忆用于后续对话
vars = memory.load_memory_variables({})
print(vars["chat_history"])
```

---

## 📊 SDK架构对比

```
su-memory SDK
├── SuMemoryLitePro     # 增强版（生产推荐）
│   ├── Ollama bge-m3 向量检索 (1024维)
│   ├── VectorGraphRAG 多跳推理引擎
│   │   ├── HNSW索引优化 (m=32, ef=64)
│   │   └── 向量量化压缩 (INT8/FP16/Binary)
│   ├── SpacetimeIndex 时空索引
│   ├── SpacetimeMultihopEngine 时空多跳融合
│   ├── MultimodalEmbedding 多模态嵌入
│   │   ├── CLIP 图像编码器
│   │   └── Whisper 音频编码器
│   ├── SpatialRAG 三维世界模型
│   │   ├── KD-Tree 空间索引
│   │   └── 空间+时间+语义三维检索
│   ├── MemoryGraph 因果图谱
│   ├── TemporalSystem 时序编码
│   ├── SessionManager 跨会话召回
│   ├── PredictionModule 时序预测
│   └── ExplainabilityModule 可解释性
├── SuMemoryLite        # 轻量版
│   ├── TF-IDF检索
│   ├── N-gram分词
│   └── 持久化存储
└── SuMemoryChatMemory  # LangChain适配器
```

### 功能对比

| 功能 | SuMemoryLite | SuMemoryLitePro |
|------|-------------|-----------------|
| **检索方式** | TF-IDF | RRF混合检索 |
| **向量检索** | ❌ | ✅ Ollama bge-m3 |
| **多跳推理** | ❌ | ✅ VectorGraphRAG |
| **HNSW索引** | ❌ | ✅ m=32, ef=64 |
| **向量量化** | ❌ | ✅ INT8/FP16/Binary |
| **时空索引** | ❌ | ✅ SpacetimeIndex |
| **时空多跳** | ❌ | ✅ SpacetimeMultihopEngine |
| **多模态嵌入** | ❌ | ✅ CLIP/Whisper |
| **三维世界模型** | ❌ | ✅ SpatialRAG |
| **因果推理** | ❌ | ✅ BFS多跳 |
| **时序感知** | ❌ | ✅ 时序编码 |
| **跨会话召回** | ❌ | ✅ 语义话题 |
| **时序预测** | ❌ | ✅ 事件预测 |
| **可解释性** | ❌ | ✅ 推理链 |
| **统一异常** | ✅ | ✅ 42 ErrorCode |
| **降级矩阵** | ✅ | ✅ 7 组件 |
| **内存占用** | < 5MB | < 50MB |

---

## ⚡ 性能基准

### SuMemoryLite (轻量版)

```
插入性能:
  ✅ 吞吐量: 94 条/秒
  ✅ 平均耗时: 10.66 ms/条

查询性能:
  ✅ P50延迟: 0.27 ms
  ✅ P95延迟: 0.39 ms
  ✅ P99延迟: 0.43 ms

内存占用:
  ✅ 1000条记忆: 1.53 MB
```

### SuMemoryLitePro (增强版)

```
语义检索:
  ✅ 向量检索: ~50ms/查询 (Ollama本地)
  ✅ 混合检索: RRF融合多路结果
  ✅ HNSW索引: O(log n) 搜索复杂度

因果线索检测（基于关键词模式，非统计因果推断）:
  ✅ 多跳推理: 支持3跳以上（opt-in，query(multihop=True)）
  ✅ 因果类型: cause/condition/result/sequence（连接词匹配）
  ✅ VectorGraphRAG: 纯向量图遍历

性能优化:
  ✅ HNSW优化: m=32, efConstruction=64, efSearch=64
  ✅ 向量量化: INT8 4x / FP16 2x / Binary 32x
  ✅ LRU缓存: 1000容量批量编码缓存

时空融合:
  ✅ 时空索引: SpacetimeIndex + TemporalSystem
  ✅ 时空多跳: SpacetimeMultihopEngine + RRF融合
  ✅ 三维世界: SpatialRAG + KD-Tree空间索引

多模态支持:
  ✅ 图像编码: CLIP ViT-B/32 (512维)
  ✅ 音频编码: Whisper模型支持
  ✅ 融合检索: text/image/audio多模态融合

时序计算:
  ✅ 时效衰减: 指数衰减 + 时序编码
  ✅ 预测模块: 基于历史趋势预测
```

### 性能指标对比（历史内部对比，非权威基准）

> ⚠️ 下表为早期内部优化前后对比，数字未经独立复现。**权威、可复现性能以 [BENCHMARK.md](BENCHMARK.md) 为准**（`python benchmarks/real_microbench.py`）。

| 指标 | 优化前 | 优化后 (v2.6.0) | 提升 |
|------|--------|--------|------|
| 多跳推理召回率 | 60% | 87.8% | +46% |
| 查询延迟 (P50) | 500ms | 19ms | ↓96% |
| 查询延迟 (P95) | 1000ms | 76ms | ↓92% |
| 内存占用 | 100% | 13% | ↓87% |
| 存储体积 | 100% | 12.5% | ↓87.5% |
| 启动时间 | ~2s | **154ms** | ↓92% |
| 嵌入缓存命中率 | 0% | **>90%** | ∞ |

### 向量量化压缩效果

| 量化模式 | 压缩比 | 精度损失 | 适用场景 |
|----------|--------|----------|----------|
| FP32 | 1x | 0% | 高精度需求 |
| FP16 | 2x | <1% | 平衡场景 |
| **INT8** | **4x** | **<1%** | **推荐** |
| Binary | 32x | ~20% | 极端内存限制 |

---

## 🎓 VMC世界模型能力

su-memory SDK作为VMC框架的Memory组件，综合成熟度达**4.9/5**：

| 维度 | 能力 | 成熟度 |
|------|------|--------|
| **长期记忆** | 语义向量存储，持久化 | ⭐⭐⭐⭐⭐ |
| **因果推理** | VectorGraphRAG多跳推理 | ⭐⭐⭐⭐⭐ |
| **时空感知** | SpacetimeIndex时空索引 | ⭐⭐⭐⭐⭐ |
| **时空多跳** | SpacetimeMultihopEngine融合 | ⭐⭐⭐⭐⭐ |
| **多模态嵌入** | CLIP/Whisper图像音频 | ⭐⭐⭐⭐ |
| **三维世界** | SpatialRAG KD-Tree空间索引 | ⭐⭐⭐⭐ |
| **向量优化** | HNSW索引+量化压缩 | ⭐⭐⭐⭐⭐ |
| **语义理解** | Ollama bge-m3本地向量 | ⭐⭐⭐⭐⭐ |
| **预测能力** | PredictionModule | ⭐⭐⭐⭐ |
| **可解释性** | ExplainabilityModule | ⭐⭐⭐⭐ |
| **情境感知** | 跨会话话题召回 | ⭐⭐⭐⭐⭐ |
| **开放领域** | RRF混合检索 | ⭐⭐⭐⭐⭐ |

### 与顶级LLM集成

| 模型 | 角色 | 集成方式 |
|------|------|----------|
| **Claude 4** | Controller | 记忆上下文注入 |
| **Gemini 2.0** | Vision+Controller | 多模态感知 |
| **DeepSeek V4** | Controller | 代码推理增强 |
| **Qwen3.5** | Controller | 中文场景优化 |

---

## 🔌 进阶功能

### 多会话管理

```python
# 创建会话
session1 = pro.create_session("项目会议")
session2 = pro.create_session("日常对话")

# 添加会话记忆
pro.add("讨论了技术方案", topic="技术", session_id=session1)
pro.add("讨论了项目进度", topic="进度", session_id=session1)

# 跨会话召回
related = pro._sessions.get_related_topics("技术")
print(related)
```

### 时序预测

```python
# 添加历史事件
pro.add("周一项目启动")
pro.add("周三完成第一阶段")
pro.add("周五测试通过")

# 预测趋势
trend = pro.predict(metric="activity")
print(trend['prediction'])
```

### 可解释推理

```python
# 查询并获取解释
results = pro.query("项目")
explanation = pro.explain_query("项目", results)

print(explanation['explanation'])
# 输出:
# 针对查询'项目'，系统检索到3条相关记忆。
# 
# 最相关记忆：项目进展顺利
# 相关度得分：85.52%
# 
# 检索因素：
#   • 语义匹配（权重40%）：85.52%
#   • 因果关联（权重30%）：基于图谱推理
#   • 时序相关性（权重20%）：时效性已计算
```

### 多模态检索

```python
# 启用多模态支持
from su_memory.sdk.multimodal import create_multimodal_manager

manager = create_multimodal_manager(
    text_embedding_func=pro._embedding.encode,
    enable_image=True,  # 启用CLIP图像编码
    enable_audio=False,
    image_weight=0.4,
    text_weight=0.6
)

# 添加多模态记忆
manager.add_multimodal_memory(
    memory_id="img_001",
    content="会议室的场景",
    image_path="/path/to/meeting.jpg"
)

# 多模态检索
results = manager.search("会议", mode="multimodal", top_k=5)
for r in results:
    print(f"{r.content} (score={r.score:.3f}, source={r.source})")
```

### 三维世界模型检索

```python
# 启用SpatialRAG三维世界模型
pro._spatial.add_spatial_memory(
    memory_id="spatial_001",
    content="在会议室A发生的事件",
    position=(10.0, 20.0, 0.0),  # x, y, z 坐标
    timestamp=1704067200
)

# 空间邻域搜索
results = pro._spatial.search_nearby(
    position=(10.0, 20.0, 0.0),
    radius=5.0
)

# 三维检索（空间+时间+语义）
results_3d = pro._spatial.search_3d(
    query="会议",
    position=(10.0, 20.0, 0.0),
    time_range=(start_ts, end_ts),
    max_distance=10.0
)
```

---

## 🛡️ v2.6.0 新增：统一异常 + 降级 + 优化

### 统一异常体系 (42 ErrorCode)

所有异常出口统一为 `SuMemoryError`，携带结构化错误码、中文描述和修复建议。

```python
from su_memory.exceptions import SuMemoryError, ErrorCode

# 有意义的异常信息
raise SuMemoryError(
    ErrorCode.FAISS_DIMENSION_MISMATCH,
    expected=768, actual=1024
)
# → SuMemoryError: [FAISS_E002] 向量维度不匹配。期望 768 维，实际 1024 维。请重建索引或统一嵌入后端
```

| 分类 | 代码范围 | 数量 |
|------|---------|:---:|
| FAISS | FAISS_E001-E005 | 5 |
| 嵌入 | EMB_E001-E005 | 5 |
| 存储 | STO_E001-E004 | 4 |
| 查询 | QRY_E001-E003 + QRY_W001-W003 | 6 |
| 图谱/并发/配置/时序/会话/插件/迁移/记忆/预测 | 各 2-3 个 | 22 |
| **总计** | | **42** |

### 降级矩阵 (7 组件全覆盖)

每个关键组件都有 2-4 级降级路径，确保任一组件故障时系统仍可运行。

| 组件 | 主路径 | 降级1 | 降级2 | 降级3 |
|------|--------|-------|-------|-------|
| 嵌入 | Ollama(bge-m3) | MiniMax API | sentence-transformers | TF-IDF |
| 向量索引 | FAISS HNSW | numpy 线性检索 | — | — |
| 图谱 | MemoryGraph | 纯向量检索 | — | — |
| 时空 | SpacetimeIndex | TemporalSystem | — | — |
| 存储 | Qdrant | SQLite (WAL) | 内存 Dict | — |
| 能量推断 | LLM (≥85%) | 关键词规则 (≥60%) | 默认值 | — |
| 会话 | SessionManager | 内存 Session | — | — |

详见 [降级矩阵文档](docs/fallback-matrix.md)。

### 性能优化

| 优化项 | 方案 | 效果 |
|--------|------|------|
| FAISS 自动调参 | HNSW/IVF/混合 3 策略 | search -20% |
| 嵌入缓存 | LFU 驱逐 + TTL 过期 | query -30% |
| 懒加载 | 14 模块按需 import | 启动 154ms |
| CI 性能门禁 | 7 大门禁 GitHub Actions | 回归自动拦截 |

---

## 📦 项目结构

```
su-memory-sdk/
├── src/su_memory/
│   ├── __init__.py              # 懒加载入口 (154ms 启动)
│   ├── exceptions.py            # 42 ErrorCode + SuMemoryError
│   ├── client.py                # SuMemoryClient
│   ├── sdk/                     # SDK 核心
│   │   ├── lite.py              # SuMemoryLite 轻量版
│   │   ├── lite_pro.py          # SuMemoryLitePro 增强版
│   │   ├── vector_graph_rag.py  # VectorGraphRAG 多跳推理
│   │   ├── spacetime_index.py   # 时空索引
│   │   ├── spacetime_multihop.py # 时空多跳融合
│   │   ├── multimodal.py        # 多模态嵌入
│   │   ├── spatial_rag.py       # 三维世界模型
│   │   ├── config.py            # 配置管理
│   │   └── exceptions.py        # SDK 异常 (→ su_memory.exceptions)
│   ├── _sys/                    # 内部系统 (懒加载)
│   │   ├── fallback.py          # FallbackChain 降级引擎
│   │   ├── _faiss_tuner.py      # FAISS 自动调参
│   │   ├── _embedding_cache.py  # LFU+TTL 嵌入缓存
│   │   ├── _lazy.py             # 懒加载引擎
│   │   ├── _plugin_interface.py # 插件接口
│   │   └── ...                  # 30+ 内部模块
│   ├── embeddings/              # 嵌入后端
│   ├── storage/                 # 存储后端
│   ├── plugins/                 # 官方插件
│   ├── integrations/            # LangChain/LlamaIndex
│   └── cli/                     # CLI 工具
├── benchmarks/                  # 性能基准套件
│   ├── bench_add.py             # 单条写入
│   ├── bench_query.py           # 查询延迟 P50/P95/P99
│   ├── bench_multihop.py        # 多跳推理
│   ├── bench_faiss.py           # FAISS 构建/搜索
│   ├── bench_memory.py          # 内存占用
│   ├── bench_concurrency.py     # 并发扩展
│   └── stress_test.py           # 3 阶段压测
├── tests/                       # 测试 (167+ 用例)
├── docs/api/                    # Sphinx API 文档
├── scripts/                     # CI 脚本
│   └── check_perf_gate.py       # 性能门禁检查
└── pyproject.toml
```

---

## 🧪 运行测试

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试 (167+ 用例)
pytest tests/ -v

# 按模块运行
pytest tests/test_lite.py -v           # 轻量版
pytest tests/test_lite_pro.py -v       # 增强版
pytest tests/test_faiss_index.py -v    # FAISS 索引
pytest tests/test_multihop_reasoning.py -v  # 多跳推理
pytest tests/test_concurrency.py -v    # 并发安全
pytest tests/test_fallback.py -v       # 降级路径

# 运行性能基准
python benchmarks/bench_add.py
python benchmarks/bench_query.py
python benchmarks/stress_test.py

# 生成 API 文档
pip install su-memory[docs]
cd docs/api && make html
# 打开 docs/api/_build/html/index.html
```

---

## 💰 定价方案

**核心原则**：所有版本功能相同，仅按容量收费。

| 版本 | 价格 | 容量 | 说明 |
|------|------|------|------|
| **Community** | 免费 | 1,000条 | 个人学习、轻量使用 |
| **Pro** | ¥99/月 | 10,000条 | 小团队、生产环境 (含 v2.6.0 异常体系+降级) |
| **Enterprise** | ¥399/月 | 100,000条 | 企业级应用 |
| **On-Premise** | ¥9,999 | 无限制 | 大型企业、私有部署 |

详细方案：[PAYMENT.md](./PAYMENT.md)

### 授权码安装

```bash
# 方式1：交互式安装
python examples/install_license.py

# 方式2：从授权码安装
python examples/install_license.py --license-key SM-PRO-XXXX-XXXX

# 方式3：从文件安装
python examples/install_license.py --file license.json

# 查看授权状态
python examples/install_license.py --status
```

---

## 📄 License

**⚠️ 重要**：本项目采用自定义双轨授权协议

- **个人学习**：免费，但须遵守使用限制
- **商业使用**：需付费授权

详细协议：[LICENSE](./LICENSE)

---

## 🙏 致谢


- LangChain Memory接口
- Ollama本地向量模型
- TF-IDF信息检索算法
- RRF (Reciprocal Rank Fusion) 融合算法

---

**版本**: v3.3.0 | **发布日期**: 2026-06-28
