# su-memory-sdk 降级矩阵 (Fallback Matrix)

> **版本**: v2.6.0 | **覆盖率**: 7/7 组件 | **保障**: 核心功能任何条件下均可用

---

## 矩阵总览

| 组件 | 主路径 | 降级1 | 降级2 | 降级3 |
|------|--------|-------|-------|-------|
| **嵌入** | Ollama (bge-m3) | MiniMax API | sentence-transformers (all-MiniLM) | TF-IDF (纯统计) |
| **向量索引** | FAISS HNSW | 线性检索 (numpy) | — | — |
| **图谱** | MemoryGraph | 纯向量检索 | — | — |
| **时空** | SpacetimeIndex | TemporalSystem (时序衰减) | — | — |
| **存储** | Qdrant | SQLite (WAL) | 内存 Dict | — |
| **能量推断** | LLM (≥85%) | 关键词规则 (≥60%) | 默认值 | — |
| **会话** | SessionManager | 内存 Session | — | — |

---

## 各组件详情

### 1. 嵌入 (Embedding) — 4 级降级

| 级别 | 方案 | 延迟 (ms) | 相对准确度 | 依赖 |
|------|------|-----------|-----------|------|
| 主路径 | Ollama bge-m3 | ~50 | 1.00 | `ollama serve` |
| 降级1 | MiniMax API | ~200 | 0.95 | `MINIMAX_API_KEY` |
| 降级2 | sentence-transformers all-MiniLM-L6-v2 | ~100 | 0.85 | `pip install sentence-transformers` |
| 降级3 | TF-IDF | ~5 | 0.60 | 无 (内置) |

**触发条件**:
- 主路径 → 降级1: Ollama 服务不可达
- 降级1 → 降级2: MiniMax API Key 未配置或 API 超时
- 降级2 → 降级3: sentence-transformers 未安装
- 降级3 兜底: 纯 TF-IDF，始终可用

**代码入口**: `su_memory.embeddings.base.EmbeddingFactory.auto_detect()`

---

### 2. 向量索引 (Vector Index) — 2 级降级

| 级别 | 方案 | 延迟 (ms) | 相对准确度 | 依赖 |
|------|------|-----------|-----------|------|
| 主路径 | FAISS HNSW | ~5 (O(log n)) | 1.00 | `pip install faiss-cpu` |
| 降级1 | numpy 线性检索 | ~50 (O(n)) | 1.00 | `numpy` (内置) |

**触发条件**:
- 主路径 → 降级1: FAISS 未安装或索引创建失败
- 降级1 兜底: 使用 `numpy.dot` 计算余弦相似度

**代码入口**: `su_memory.sdk.lite_pro._check_and_suggest_faiss()`

---

### 3. 图谱 (Graph) — 2 级降级

| 级别 | 方案 | 延迟 (ms) | 相对准确度 | 依赖 |
|------|------|-----------|-----------|------|
| 主路径 | MemoryGraph 多跳推理 | ~20 | 1.00 | 内置 |
| 降级1 | 纯向量检索 | ~10 | 0.70 | FAISS / numpy |

**触发条件**:
- 主路径 → 降级1: `enable_graph=False` 或图谱操作异常
- 降级1 兜底: 放弃因果推理，返回相似度排名结果

**代码入口**: `su_memory.sdk.lite_pro.SuMemoryLitePro.query_multihop()`

---

### 4. 时空 (Temporal/Spatial) — 2 级降级

| 级别 | 方案 | 延迟 (ms) | 相对准确度 | 依赖 |
|------|------|-----------|-----------|------|
| 主路径 | SpacetimeIndex (时空索引) | ~15 | 1.00 | 内置 |
| 降级1 | TemporalSystem (时序衰减) | ~5 | 0.75 | 内置 |

**触发条件**:
- 主路径 → 降级1: `enable_temporal=True` 但 spacetime 模块不可用
- 降级1 兜底: 使用指数衰减时间权重

**代码入口**: `su_memory.sdk.lite_pro.SuMemoryLitePro.query_multihop_spacetime()`

---

### 5. 存储 (Storage) — 3 级降级

| 级别 | 方案 | 延迟 (ms) | 相对准确度 | 依赖 |
|------|------|-----------|-----------|------|
| 主路径 | Qdrant 向量数据库 | ~100 | 1.00 | Qdrant 服务 |
| 降级1 | SQLite (WAL 模式) | ~10 | 1.00 | `sqlite3` (内置) |
| 降级2 | 内存 Dict | ~1 | 1.00 | 无 |

**触发条件**:
- 主路径 → 降级1: Qdrant 未配置或不可达
- 降级1 → 降级2: SQLite 写入失败 (磁盘满/权限错误)
- 降级2 兜底: 所有数据在内存中，进程退出丢失

**代码入口**: `su_memory.storage.sqlite_backend.SQLiteBackend`, `su_memory.sdk.lite_pro`

---

### 6. 能量推断 (Prediction) — 3 级降级

| 级别 | 方案 | 延迟 (ms) | 相对准确度 | 依赖 |
|------|------|-----------|-----------|------|
| 主路径 | LLM 能量推断 | ~500 | 1.00 (≥85%) | LLM 服务 |
| 降级1 | 关键词规则推断 | ~10 | 0.71 (≥60%) | 内置规则库 |
| 降级2 | 默认值 | ~1 | 0.40 | 无 |

**触发条件**:
- 主路径 → 降级1: LLM 不可用或超时
- 降级1 → 降级2: 规则库未加载
- 降级2 兜底: 返回预设默认能量值

**代码入口**: `su_memory.sdk.lite_pro.SuMemoryLitePro.predict()`

---

### 7. 会话 (Session) — 2 级降级

| 级别 | 方案 | 延迟 (ms) | 相对准确度 | 依赖 |
|------|------|-----------|-----------|------|
| 主路径 | SessionManager (持久化) | ~10 | 1.00 | SQLite |
| 降级1 | 内存 Session | ~1 | 0.90 | 无 |

**触发条件**:
- 主路径 → 降级1: `enable_session=True` 但持久化存储不可用
- 降级1 兜底: 会话数据仅在内存中保留

**代码入口**: `su_memory.sdk.lite_pro.SuMemoryLitePro._session_manager`

---

## 使用方式

### 编程式使用 FallbackChain

```python
from su_memory._sys.fallback import (
    FallbackChain, FallbackLevel, FallbackManager,
    create_embedding_fallback_chain
)

# 方式1: 使用预定义链
chain = create_embedding_fallback_chain(
    ollama_func=lambda text: ollama_embed(text),
    minimax_func=lambda text: minimax_embed(text),
    tfidf_func=lambda text: tfidf_embed(text),
)

result = chain.try_execute("hello world")
print(f"使用: {result.step_name}, 级别: {result.level}")

# 方式2: 全局管理器
fm = FallbackManager()
fm.register("embed", chain)
result = fm.execute("embed", "hello world")

# 查看统计
print(fm.get_all_stats())
# {'embed': {'primary': 1, 'fallback': 0, 'guaranteed': 0, 'failed': 0}}
```

### 自定义降级回调

```python
def on_degrade(step, error):
    print(f"⚠️ 降级触发: {step.name} → 原因: {error}")

chain.on_fallback(on_degrade)
```

---

## 性能对比

| 降级路径 | 主路径延迟 | 降级后延迟 | 延迟变化 |
|----------|-----------|-----------|---------|
| 嵌入 (Ollama → TF-IDF) | 50ms | 5ms | ↓ 90% |
| 向量索引 (FAISS → numpy) | 5ms | 50ms | ↑ 900% |
| 图谱 (Graph → Vector) | 20ms | 10ms | ↓ 50% |
| 时空 (Spacetime → Temporal) | 15ms | 5ms | ↓ 67% |
| 存储 (Qdrant → SQLite) | 100ms | 10ms | ↓ 90% |
| 能量推断 (LLM → Default) | 500ms | 1ms | ↓ 99.8% |
| 会话 (Manager → Memory) | 10ms | 1ms | ↓ 90% |

---

## 异常与监控

所有降级操作通过 `logging.warning` 记录，并可通过以下方式监控：

```python
import logging
logging.getLogger("su_memory._sys.fallback").setLevel(logging.WARNING)
```

降级链耗尽时会抛出 `SuMemoryError(ErrorCode.EMBED_FALLBACK_EXHAUSTED)`。
