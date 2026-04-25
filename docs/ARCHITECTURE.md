# su-memory SDK 四位一体技术架构

> 版本: v1.4.0 | 更新日期: 2026-04-25

---

## 一、架构概览

su-memory SDK 采用**四位一体**核心架构，融合向量检索、图关系、时空索引和可解释性四大能力：

```
┌─────────────────────────────────────────────────────────────────┐
│                      SuMemoryLitePro                             │
│                   四位一体记忆引擎                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Memory    │  │   Vector    │  │ Spacetime   │            │
│  │   Graph    │  │  GraphRAG   │  │   Index     │            │
│  │ 因果图谱    │  │ 多跳推理    │  │ 时空索引    │            │
│  │ BFS遍历    │  │ HNSW量化    │  │ 时间衰减    │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│               ┌─────────────────────┐                           │
│               │ SpacetimeMultihop   │                           │
│               │  Engine (融合引擎)  │                           │
│               │  RRF混合排序        │                           │
│               └──────────┬──────────┘                           │
│                          │                                      │
│         ┌────────────────┼────────────────┐                    │
│         ▼                ▼                ▼                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Multimodal  │  │ SpatialRAG  │  │Explainable │             │
│  │Embedding   │  │ 三维世界    │  │Module      │             │
│  │ CLIP/音频  │  │ KD-Tree     │  │ 可解释推理 │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、核心模块详解

### 2.1 VectorGraphRAG 多跳推理引擎

**技术原理**：纯向量实现的多跳推理，无需Neo4j图库

```
传统Graph RAG:          VectorGraphRAG:
┌─────────┐              ┌─────────┐
│ Neo4j   │              │ 向量库  │
│ 图库    │              │ (FAISS)│
└────┬────┘              └────┬────┘
     │                        │
     ▼                        ▼
┌─────────┐              ┌─────────┐
│ 向量库  │              │ 边关系  │
│ (分离)  │              │ (向量)  │
└─────────┘              └─────────┘

优势: 仅需一套系统，部署简单
```

**核心方法**：

| 方法 | 功能 | 复杂度 |
|------|------|--------|
| `_semantic_search()` | 语义种子检索（第一跳） | O(log n) |
| `_find_neighbors()` | 邻居发现（第N跳） | O(k log n) |
| `multi_hop_query()` | BFS扩展多跳推理 | O(m^k) |
| `_encode_relation()` | 关系编码（因果/时序/语义） | O(1) |

**因果类型**：

```python
CAUSAL_KEYWORDS = {
    "导致": 0.9, "因为": 0.8, "如果": 0.7,
    "因此": 0.75, "所以": 0.75
}

TEMPORAL_KEYWORDS = {
    "首先": 0.6, "其次": 0.6, "最后": 0.5,
    "之前": 0.4, "之后": 0.4
}

SEMANTIC_KEYWORDS = {
    "属于": 0.6, "是": 0.5, "包含": 0.55
}
```

**融合模式**：

```
┌──────────────────────────────────────────────────────┐
│ fusion_mode="vector_first" (默认)                      │
│ ┌──────────────────┐    ┌──────────────────┐        │
│ │ VectorGraphRAG   │ →  │ 因果增强 (+10%)   │        │
│ │ 语义引导推理      │    │ MemoryGraph 链接  │        │
│ └──────────────────┘    └──────────────────┘        │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ fusion_mode="hybrid"                                  │
│ ┌──────────────────┐    ┌──────────────────┐        │
│ │ VectorGraphRAG   │ +  │ MemoryGraph BFS  │        │
│ │ 60% 权重         │    │ 40% 权重         │        │
│ └──────────────────┘    └──────────────────┘        │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ fusion_mode="graph_first"                             │
│ ┌──────────────────┐    ┌──────────────────┐        │
│ │ MemoryGraph BFS  │ →  │ 因果权重增强      │        │
│ │ 因果结构推理      │    │ cause: 1.5x     │        │
│ └──────────────────┘    └──────────────────┘        │
└──────────────────────────────────────────────────────┘
```

### 2.2 SpacetimeIndex 时空索引

**技术原理**：融合TemporalSystem与VectorGraphRAG的时空联合索引

```
时间维度                    空间维度
    │                          │
    ▼                          ▼
┌─────────┐              ┌─────────┐
│Temporal │              │Vector   │
│System   │              │GraphRAG │
└────┬────┘              └────┬────┘
     │                        │
     └────────┬───────────────┘
              ▼
     ┌─────────────────┐
     │ SpacetimeIndex  │
     │ 时空联合检索    │
     └─────────────────┘
```

**时间衰减函数**：

```python
def time_decay(ts, half_life_days=30):
    """指数衰减"""
    days = (current_time - ts) / (24 * 3600)
    return math.exp(-0.693 * days / half_life_days)
```

### 2.3 SpacetimeMultihopEngine 时空多跳融合

**技术原理**：融合VectorGraphRAG和SpacetimeIndex的RRF混合排序

```
Query: "深度学习的影响"

第一跳: VectorGraphRAG语义检索
  → [机器学习, 神经网络, CNN]

第二跳: Spacetime时序扩展
  → [机器学习@2024-01, 神经网络@2024-03]

融合: RRF (Reciprocal Rank Fusion)
  score = Σ (1 / (60 + rank_i))
```

### 2.4 MultimodalEmbedding 多模态嵌入

**架构**：

```
┌──────────────────────────────────────────────────────┐
│           MultimodalEmbeddingManager                  │
├──────────────────────────────────────────────────────┤
│                                                       │
│   文本 ──→ TextEncoder ──→ 文本向量 (1024d)           │
│     │                                                  │
│   图像 ──→ CLIP(ViT-B/32) ──→ 图像向量 (512d)       │
│     │                                                  │
│   音频 ──→ Whisper ──→ 音频向量 (512d)               │
│                                                       │
│   融合: final_score = w1×text + w2×image + w3×audio │
└──────────────────────────────────────────────────────┘
```

**检索模式**：

| 模式 | 说明 |
|------|------|
| `text` | 仅文本检索 |
| `image` | 仅图像检索 |
| `audio` | 仅音频检索 |
| `multimodal` | 多模态融合检索（默认） |

### 2.5 SpatialRAG 三维世界模型

**架构**：

```
┌──────────────────────────────────────────────────────┐
│              SpatialRAG 三维世界模型                 │
├──────────────────────────────────────────────────────┤
│                                                       │
│   空间维度 (x, y, z)  ──→ KD-Tree 空间索引          │
│     │                                                  │
│   时间维度 (t)      ──→ 时间戳排序                   │
│     │                                                  │
│   语义维度 (v)      ──→ 向量相似度                   │
│                                                       │
│   三维检索: f(x,y,z,t,v) = 空间×时间×语义           │
└──────────────────────────────────────────────────────┘
```

**搜索能力**：

| 方法 | 功能 |
|------|------|
| `search_nearby()` | 空间邻域搜索 |
| `search_3d()` | 三维检索（空间+时间+语义） |
| `search_path()` | 路径搜索 |

---

## 三、技术指标

### 3.1 HNSW索引参数

```python
{
    "m": 32,                    # 每层连接数
    "efConstruction": 64,      # 构建时搜索宽度
    "efSearch": 64,            # 搜索时搜索宽度
    "search_complexity": "O(log n)"
}
```

### 3.2 向量量化压缩

| 量化模式 | 压缩比 | 精度损失 | 适用场景 |
|----------|--------|----------|----------|
| FP32 | 1x | 0% | 高精度需求 |
| FP16 | 2x | <1% | 平衡场景 |
| **INT8** | **4x** | **<1%** | **推荐** |
| Binary | 32x | ~20% | 极端内存限制 |

### 3.3 性能对比

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 多跳推理召回率 | 60% | 87.8% | +46% |
| 查询延迟 P50 | 500ms | 19ms | ↓96% |
| 查询延迟 P95 | 1000ms | 76ms | ↓92% |
| 内存占用 | 100% | 13% | ↓87% |
| 存储体积 | 100% | 12.5% | ↓87.5% |

---

## 四、模块依赖关系

```
SuMemoryLitePro
├── _memories: Dict[str, MemoryNode]
│   │
├── _graph: MemoryGraph
│   │
├── _embedding: OllamaEmbedding
│   │
├── _vector_graph: VectorGraphRAG
│   ├── nodes: Dict[str, VectorNode]
│   ├── edges: List[Tuple[str, str, float]]
│   └── _faiss_index: faiss.IndexHNSW
│       │
├── _spacetime: SpacetimeIndex
│   ├── embedding_func
│   └── _temporal_nodes: Dict
│       │
├── _spacetime_engine: SpacetimeMultihopEngine
│   ├── vector_graph
│   ├── spacetime
│   └── memory_nodes
│       │
├── _multimodal: MultimodalEmbeddingManager
│   ├── _image_encoder: ImageEncoder (CLIP)
│   ├── _audio_encoder: AudioEncoder (Whisper)
│   └── _memories: Dict[str, MultimodalMemory]
│       │
├── _spatial: SpatialRAG
│   ├── _spatial_index: KDTree
│   ├── _spatial_nodes: Dict[str, SpatialNode]
│   └── _trajectories: Dict[str, TrajectoryTracker]
│       │
├── _temporal: TemporalSystem
├── _sessions: SessionManager
├── _prediction: PredictionModule
└── _explainability: ExplainabilityModule
```

---

## 五、API调用流程

### 5.1 添加记忆

```
pro.add(content, metadata)
    │
    ▼
┌─────────────────────────────────────┐
│ 1. 生成向量: _embedding.encode()     │
│ 2. 创建MemoryNode                   │
│ 3. 添加到_memories                  │
│ 4. 更新Graph: _graph.add()          │
│ 5. 更新VectorGraphRAG               │
│ 6. 更新SpacetimeIndex               │
│ 7. 更新SpacetimeMultihopEngine      │
│ 8. 更新MultimodalEmbedding          │
│ 9. 更新SpatialRAG                   │
└─────────────────────────────────────┘
```

### 5.2 多跳推理查询

```
pro.query_multihop(query, max_hops)
    │
    ▼
┌─────────────────────────────────────┐
│ 1. 编码查询: _embedding.encode()     │
│ 2. VectorGraphRAG.multi_hop_query() │
│    ├── _semantic_search() → 种子    │
│    └── BFS扩展 → 多跳结果           │
│ 3. SpacetimeMultihopEngine融合      │
│    └── RRF混合排序                  │
│ 4. 返回排序结果                     │
└─────────────────────────────────────┘
```

### 5.3 三维检索

```
pro._spatial.search_3d(query, position, time_range)
    │
    ▼
┌─────────────────────────────────────┐
│ 1. 语义检索: embedding_func(query)  │
│ 2. 空间搜索: KDTree.search_nearby() │
│ 3. 时间过滤: time_range检查         │
│ 4. 三维融合: 空间×时间×语义         │
│ 5. 返回SpatialSearchResult         │
└─────────────────────────────────────┘
```

---

## 六、版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v1.4.0 | 2026-04-25 | 四位一体+多模态+三维世界模型 |
| v1.3.0 | 2026-04-25 | PredictionModule+ExplainabilityModule |
| v1.2.0 | 2026-04-22 | SuMemoryLitePro增强版 |
| v1.1.0 | 2026-04-21 | 首次正式发布 |

---

**文档版本**: v1.0
**创建日期**: 2026-04-25
**技术参考**: Vector Graph RAG、DeepSeek-V4、微软端侧语音、世界模型