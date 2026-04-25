# su-memory SDK API 参考文档

> 版本: v1.4.0 | 更新日期: 2026-04-25

---

## 一、快速开始

### 1.1 安装

```bash
pip install su-memory
```

### 1.2 基础导入

```python
from su_memory import SuMemoryLite, SuMemoryLitePro
from su_memory.sdk import SuMemoryLitePro
```

---

## 二、SuMemoryLite 轻量版

### 2.1 类定义

```python
class SuMemoryLite(storage_path: str = "./storage")
```

### 2.2 方法

#### `add(content: str, metadata: dict = None) -> str`

添加一条记忆。

**参数**:
- `content` (str): 记忆内容
- `metadata` (dict, optional): 元数据

**返回**: str - 记忆ID

**示例**:
```python
client = SuMemoryLite()
memory_id = client.add("今天天气很好")
```

#### `query(query: str, top_k: int = 5) -> List[dict]`

查询相关记忆。

**参数**:
- `query` (str): 查询文本
- `top_k` (int): 返回结果数量，默认5

**返回**: List[dict] - 记忆列表

**示例**:
```python
results = client.query("天气", top_k=3)
for r in results:
    print(f"{r['content']} (score={r['score']:.3f})")
```

#### `delete(memory_id: str) -> bool`

删除记忆。

**参数**:
- `memory_id` (str): 记忆ID

**返回**: bool - 是否成功

#### `get(memory_id: str) -> dict`

获取记忆详情。

**参数**:
- `memory_id` (str): 记忆ID

**返回**: dict - 记忆详情

#### `count() -> int`

获取记忆数量。

**返回**: int - 记忆总数

---

## 三、SuMemoryLitePro 增强版

### 3.1 类定义

```python
class SuMemoryLitePro(
    storage_path: str = "./storage",
    embedding_backend: str = "ollama",
    model_name: str = "bge-m3",
    enable_vector: bool = True,
    enable_graph: bool = True,
    enable_temporal: bool = True,
    enable_session: bool = True,
    enable_prediction: bool = True,
    enable_explainability: bool = True,
    max_memories: int = 10000,
    cache_size: int = 1000
)
```

### 3.2 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `storage_path` | str | "./storage" | 存储路径 |
| `embedding_backend` | str | "ollama" | 向量后端: ollama/minimax/openai |
| `model_name` | str | "bge-m3" | 向量模型名 |
| `enable_vector` | bool | True | 启用向量检索 |
| `enable_graph` | bool | True | 启用图关系 |
| `enable_temporal` | bool | True | 启用时序系统 |
| `enable_session` | bool | True | 启用会话管理 |
| `enable_prediction` | bool | True | 启用预测模块 |
| `enable_explainability` | bool | True | 启用可解释性 |
| `max_memories` | int | 10000 | 最大记忆数 |
| `cache_size` | int | 1000 | LRU缓存大小 |

### 3.3 核心方法

#### `add(content: str, metadata: dict = None, energy_type: str = "土", topic: str = None, session_id: str = None) -> str`

添加记忆到所有索引。

**参数**:
- `content` (str): 记忆内容
- `metadata` (dict, optional): 元数据
- `energy_type` (str, optional): 五行类型，默认"土"
- `topic` (str, optional): 话题标签
- `session_id` (str, optional): 会话ID

**返回**: str - 记忆ID

**示例**:
```python
pro = SuMemoryLitePro()
memory_id = pro.add(
    "机器学习是人工智能的核心技术",
    metadata={"source": "教科书"},
    energy_type="金"
)
```

#### `query(query: str, top_k: int = 5, time_range: tuple = None, energy_filter: list = None) -> List[dict]`

语义检索。

**参数**:
- `query` (str): 查询文本
- `top_k` (int): 返回数量
- `time_range` (tuple, optional): 时间范围 (start_ts, end_ts)
- `energy_filter` (list, optional): 五行类型过滤 ["金", "木"]

**返回**: List[dict] - 检索结果

**示例**:
```python
results = pro.query("人工智能", top_k=5)
```

#### `query_multihop(query: str, max_hops: int = 3, top_k: int = 5, fusion_mode: str = "vector_first") -> List[dict]`

多跳推理查询。

**参数**:
- `query` (str): 查询文本
- `max_hops` (int): 最大跳数，默认3
- `top_k` (int): 返回数量
- `fusion_mode` (str): 融合模式
  - "vector_first": 向量优先（默认）
  - "hybrid": 混合模式
  - "graph_first": 图优先

**返回**: List[dict] - 多跳推理结果

**示例**:
```python
results = pro.query_multihop("深度学习的影响", max_hops=3)
for r in results:
    print(f"{r['content']} (hops={r['hops']}, score={r['score']:.3f})")
```

#### `search_multihop(query: str, time_range: tuple = None, max_hops: int = 3, top_k: int = 5) -> List[dict]`

时空多跳搜索。

**参数**:
- `query` (str): 查询文本
- `time_range` (tuple, optional): 时间范围
- `max_hops` (int): 最大跳数
- `top_k` (int): 返回数量

**返回**: List[dict] - 时空多跳结果

#### `link_memories(source_id: str, target_id: str, relation_type: str = "cause") -> bool`

建立记忆间的因果链接。

**参数**:
- `source_id` (str): 源记忆ID
- `target_id` (str): 目标记忆ID
- `relation_type` (str): 关系类型
  - "cause": 因果关系
  - "condition": 条件关系
  - "result": 结果关系
  - "sequence": 时序关系

**返回**: bool - 是否成功

**示例**:
```python
pro.link_memories(mem1_id, mem2_id, "cause")
```

#### `explain_query(query: str, results: List[dict]) -> dict`

获取查询解释。

**参数**:
- `query` (str): 查询文本
- `results` (List[dict]): 查询结果

**返回**: dict - 解释详情

**示例**:
```python
results = pro.query("学习")
explanation = pro.explain_query("学习", results)
print(explanation['explanation'])
```

#### `predict(query: str, metric: str = "activity") -> dict`

时序预测。

**参数**:
- `query` (str): 查询文本
- `metric` (str): 预测指标类型

**返回**: dict - 预测结果

### 3.4 会话管理方法

#### `create_session(name: str) -> str`

创建会话。

**返回**: str - 会话ID

#### `get_session(session_id: str) -> dict`

获取会话信息。

#### `list_sessions() -> List[dict]`

列出所有会话。

#### `delete_session(session_id: str) -> bool`

删除会话。

### 3.5 内部组件属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `_memories` | Dict | 记忆存储 |
| `_graph` | MemoryGraph | 因果图谱 |
| `_embedding` | OllamaEmbedding | 向量编码器 |
| `_vector_graph` | VectorGraphRAG | 多跳推理引擎 |
| `_spacetime` | SpacetimeIndex | 时空索引 |
| `_spacetime_engine` | SpacetimeMultihopEngine | 时空多跳融合 |
| `_multimodal` | MultimodalEmbeddingManager | 多模态嵌入 |
| `_spatial` | SpatialRAG | 三维世界模型 |
| `_temporal` | TemporalSystem | 时序系统 |
| `_sessions` | SessionManager | 会话管理 |
| `_prediction` | PredictionModule | 预测模块 |
| `_explainability` | ExplainabilityModule | 可解释性 |

---

## 四、VectorGraphRAG 多跳推理引擎

### 4.1 类定义

```python
class VectorGraphRAG(
    embedding_func: Callable[[str], List[float]],
    dims: int = 1024,
    enable_faiss: bool = True,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 64,
    hnsw_ef_search: int = 64,
    quantization_mode: str = "fp32"
)
```

### 4.2 方法

#### `add_memory(memory_id: str, content: str, embedding: List[float])`

添加记忆节点。

#### `add_edge(source_id: str, target_id: str, relation_type: str = "cause", weight: float = 1.0)`

添加边关系。

#### `multi_hop_query(query_embedding: List[float], max_hops: int = 3, top_k: int = 5) -> List[dict]`

多跳推理查询。

#### `get_memory_stats() -> dict`

获取内存统计。

**返回**:
```python
{
    "n_nodes": 1000,
    "n_edges": 500,
    "memory_usage_mb": 50.5,
    "compression_ratio": 4.0,
    "memory_saved_mb": 150.0
}
```

---

## 五、SpacetimeIndex 时空索引

### 5.1 类定义

```python
class SpacetimeIndex(
    embedding_func: Callable[[str], List[float]],
    dims: int = 1024,
    decay_rate: float = 0.95
)
```

### 5.2 方法

#### `add(memory_id: str, content: str, embedding: List[float], timestamp: int)`

添加时空记忆。

#### `search(query_embedding: List[float], time_range: tuple = None, top_k: int = 5) -> List[dict]`

时空检索。

---

## 六、SpacetimeMultihopEngine 时空多跳融合

### 6.1 类定义

```python
class SpacetimeMultihopEngine(
    vector_graph: VectorGraphRAG,
    spacetime: SpacetimeIndex,
    memory_nodes: Dict[str, Any],
    embedding_func: Callable[[str], List[float]] = None
)
```

### 6.2 方法

#### `search_multihop(query: str, time_range: tuple = None, max_hops: int = 3, top_k: int = 5) -> List[dict]`

时空多跳搜索。

---

## 七、MultimodalEmbedding 多模态嵌入

### 7.1 类定义

```python
class MultimodalEmbeddingManager(
    text_embedding_func: Callable[[str], List[float]],
    enable_image: bool = False,
    enable_audio: bool = False,
    image_weight: float = 0.4,
    audio_weight: float = 0.3,
    text_weight: float = 0.3
)
```

### 7.2 方法

#### `add_multimodal_memory(memory_id: str, content: str, image_path: str = None, audio_path: str = None, **kwargs) -> bool`

添加多模态记忆。

**示例**:
```python
manager.add_multimodal_memory(
    memory_id="img_001",
    content="会议室场景",
    image_path="/path/to/meeting.jpg"
)
```

#### `search(query: str, query_image: str = None, mode: str = "multimodal", top_k: int = 5) -> List[MultimodalSearchResult]`

多模态检索。

**参数**:
- `mode`: "text" | "image" | "audio" | "multimodal"

#### `get_stats() -> dict`

获取统计信息。

---

## 八、SpatialRAG 三维世界模型

### 8.1 类定义

```python
class SpatialRAG(
    embedding_func: Callable[[str], List[float]],
    spacetime: SpacetimeIndex = None,
    dim: int = 3,
    enable_trajectory: bool = True
)
```

### 8.2 方法

#### `add_spatial_memory(memory_id: str, content: str, position: Tuple[float, float, float], timestamp: int = None, entity_id: str = None, **kwargs) -> bool`

添加空间记忆。

**参数**:
- `position`: (x, y, z) 坐标

**示例**:
```python
spatial.add_spatial_memory(
    memory_id="loc_001",
    content="在会议室A发生",
    position=(10.0, 20.0, 0.0),
    timestamp=1704067200
)
```

#### `search_nearby(position: Tuple[float, float, float], radius: float, max_results: int = 10) -> List[SpatialSearchResult]`

空间邻域搜索。

#### `search_3d(query: str, position: Tuple[float, float, float], time_range: tuple = None, max_distance: float = 10.0, max_results: int = 10) -> List[SpatialSearchResult]`

三维检索。

#### `search_path(start_pos: Tuple, end_pos: Tuple, max_distance: float = 5.0) -> List[SpatialSearchResult]`

路径搜索。

#### `get_trajectory(entity_id: str) -> TrajectoryTracker`

获取实体轨迹。

#### `get_stats() -> dict`

获取统计信息。

---

## 九、MemoryGraph 因果图谱

### 9.1 方法

#### `add(memory_id: str, content: str, energy_type: str = "土")`

添加节点。

#### `connect(source_id: str, target_id: str, relation_type: str = "cause")`

连接节点。

#### `get_related(memory_id: str, depth: int = 1) -> List[str]`

获取关联节点。

#### `get_path(start_id: str, end_id: str, max_depth: int = 5) -> List[str]`

获取路径。

---

## 十、TemporalSystem 时序系统

### 10.1 方法

#### `encode_timestamp(timestamp: int) -> List[float]`

编码时间戳。

#### `calculate_decay(timestamp: int, half_life_days: float = 30) -> float`

计算时间衰减。

#### `encode_with_time(content: str, timestamp: int) -> List[float]`

时间增强编码。

---

## 十一、配置常量

### 11.1 五行类型

```python
ENERGY_TYPES = ["金", "木", "水", "火", "土"]
```

### 11.2 因果关系类型

```python
RELATION_TYPES = ["cause", "condition", "result", "sequence"]
```

### 11.3 量化模式

```python
QUANTIZATION_MODES = ["fp32", "fp16", "int8", "binary"]
```

### 11.4 融合模式

```python
FUSION_MODES = ["vector_first", "hybrid", "graph_first"]
```

---

## 十二、异常类

### 12.1 SuMemoryError

基础异常类。

### 12.2 MemoryNotFoundError

记忆未找到异常。

### 12.3 EmbeddingError

向量编码异常。

### 12.4 StorageError

存储异常。

---

**文档版本**: v1.0
**更新日期**: 2026-04-25