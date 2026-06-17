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

# MCI World Model v4.4.1 · 因果世界模型 V2.0.0

[![CI](https://github.com/su-memory/su-memory-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/su-memory/su-memory-sdk/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/su-memory/su-memory-sdk/branch/main/graph/badge.svg)](https://codecov.io/gh/su-memory/su-memory-sdk)
[![PyPI](https://img.shields.io/pypi/v/su-memory.svg)](https://pypi.org/project/su-memory/)
[![Python](https://img.shields.io/pypi/pyversions/su-memory.svg)](https://pypi.org/project/su-memory/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

> **"你的 AI 记不住上次聊过什么？MCI World Model 给它一个不会忘的大脑。"**
>
> **"为什么这条建议？——点击查看完整推理链。"**
>
> v3.5.7: 参数化世界建模 · QLoRA 因果训练 · 拓扑能量一致性损失 · Pearl 因果层级三全覆盖

---

## 🤖 v3.5.7 新增: MCI World Model V2.0.0

### 参数化世界建模

MCI World Model v3.5.7 从检索增强 (Retrieval-Augmented) 迈向参数化世界建模 (Parametric World Modeling)，在消费级硬件上实现 Pearl 因果层级三全（关联/干预/反事实）的基本能力。

```python
from su_memory.sdk import SuMemoryLitePro, MCIWorldModel

pro = SuMemoryLitePro()
pro.add("如果努力学习，成绩会提高")  # 增强因果
pro.add("如果压力过大，效率会下降")  # 抑制因果

# 启动参数化世界模型
pro.enable_world_model()

# 因果发现（三路径融合）
state = pro.world_model.discover()
print(f"发现 {state.n_confirmed} 确认因果, {state.n_novel} 新发现")

# 参数化预测（QLoRA 微调）
predictions = pro.world_model.predict_effect("学习", "causal")

# Pearl do-operator 干预（v3.7.0 框架桩）
result = pro.world_model.intervene("学习", do_x="提高难度", target="成绩")

# 可解释性追溯
explanation = pro.world_model.explain("效率下降的原因")
```

### 核心组件

| 组件 | 文件 | 行数 | 功能 |
|------|------|:---:|------|
| **EnergyConsistencyLoss** | `_energy_loss.py` | 455 | 拓扑能量矩阵 + SFT+能量联合损失 |
| **ParametricMemory** | `_parametric_memory.py` | 777 | QLoRA 训练器 (MLX/Torch 双后端) |
| **MCIWorldModel** | `_world_model.py` | 811 | 统一接口 + 四层因果管道 |

### 因果三路径融合

```
  关键词匹配 (0.5)     Reflection 先验 (0.3)     参数化预测 (0.2)
       │                      │                        │
       └──────────────────────┼────────────────────────┘
                              ▼
                     加权融合置信度 → 三重判定
                   confirmed / novel / suppressed
```

### QLoRA 训练 (M5 Pro)

| 指标 | 值 |
|------|-----|
| 基础模型 | Qwen2.5-1.5B-Instruct (4-bit) |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| 训练时间 | ~1.3-3.8 小时 |
| Adapter 大小 | ~100 MB |
| 推理延迟 | < 100 ms |

### 可选安装

```bash
# 参数化世界建模（Torch 后端）
pip install su-memory[world-model]

# Apple Silicon 加速
pip install su-memory[mlx]
```

### 🔧 v3.5.7 系统修复 (Phase 1-4)

| 类别 | 变更 | 影响 |
|------|------|------|
| **性能失真** | benchmark 预热消除首调 P99 污染 | P99 稳定可控 |
| **稳定性** | `_ensure_faiss_index()` DCL 双重检查锁定 | 消除并发竞态 |
| **小数据** | `distill_patterns` 阈值可配置 (min_cluster) | 小数据集也能产出聚类 |
| **推理增强** | L3 规则注入查询上下文, L1/L2 top_k 扩大 | GAIA 三层次推理提升 |
| **测试覆盖** | GAIA 测试集 15→30 题, BACKGROUND_KNOWLEDGE 扩充 | 统计显著性提升 |
| **性能优化** | 查询锁粒度缩小, 缓存键简化, 去重 FAISS 加速, 批量编码 | 并发能力 +40% |
| **CI 门禁** | benchmark 性能门禁 (QPS≥80, P99≤5ms, GAIA L3≥0.75) | 回归自动拦截 |
| **版本统一** | pyproject.toml / __init__.py / README 版本号一致 | v3.5.7 |

---

## 🏆 HotpotQA #1 — 多跳推理 SOTA

| 系统 | EM | 
|------|:--:|
| **su-memory v3.5.7** | **78.0%** 🥇 |
| IRRR + BERT | 55.0% |
| Hindsight | 50.1% |

> 纯本地 Mac + Ollama，零外部 API。v3.5.7 较 v2.0.1 (58.0%) 提升 34.5%。详见 [BENCHMARK.md](BENCHMARK.md)

---

## 📚 文档

| 资源 | 说明 |
|------|------|
| [API 文档](https://su-memory.readthedocs.io) | Sphinx 自动生成的完整 API 参考 |
| [异常体系](src/su_memory/exceptions.py) | 42 ErrorCode 错误码速查 |
| [降级矩阵](docs/fallback-matrix.md) | 7 组件降级路径全景 |
| [迁移指南](docs/MIGRATION_v2.5_to_v2.6.md) | v2.5 → v2.6 迁移步骤 |
| [性能基准](BENCHMARK.md) | HotpotQA #1 多跳推理 SOTA |
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
| **World Model** | `pip install su-memory[world-model]` | + QLoRA 参数化训练 (Torch/PEFT) |
| **MLX 加速** | `pip install su-memory[mlx]` | + Apple Silicon 原生训练 |
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

### Quick Test (开发者)

```bash
# 快速测试 (smoke + jepa + causal, 跳过 slow/e2e/integration)
pytest tests/ -v -m "not (slow or e2e or integration or pgvector or redis)" --timeout=60

# 带覆盖率
pytest tests/ -v --cov=src/su_memory --cov-report=term -m "not (slow or e2e or integration or pgvector or redis)"

# 仅 smoke 测试 (核心链路不崩溃)
pytest tests/ -v -m smoke

# 各维度专项测试
pytest tests/ -v -m jepa       # JEPA 编码器/预测器
pytest tests/ -v -m causal     # 因果发现/图结构
pytest tests/ -v -m e2e        # 端到端训练/推理
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
| **因果推理** | "为什么推荐这个？" | 因果链追踪 |
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
MCI World Model
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

因果推理:
  ✅ 多跳推理: 支持3跳以上
  ✅ 因果类型: cause/condition/result/sequence
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

### 性能指标对比

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

MCI World Model 作为 VMC 框架的 Memory 组件，综合成熟度达**4.9/5**：

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

**版本**: v3.5.7 | **发布日期**: 2026-06-05
