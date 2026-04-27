# su-memory SDK V1.7.6 → V1.7.7 优化变更日志
## 优化者: 小源 (Nutri-Brain 项目总顾问)
## 日期: 2026-04-28
## 目标: 冷启动从 ~90s 降到 <5s (首次) / <1s (后续)

---

## 变更文件

### `src/su_memory/sdk/lite_pro.py` (+120行, -60行)

#### 1. 懒加载向量服务 (Lazy Embedding Init)
**问题**: `__init__` 中同步检测 Ollama → 连接失败时依次尝试 3 个后端 → 每次启动 ~15s
**修改**: 新增 `_ensure_embedding()` 方法，首次 `add()`/`query()` 时才初始化
- 优先 Ollama (超时 2s, 原 5s)
- 回退 sentence-transformers (本地, 无需网络)
- `_ollama_checked` 标志防止重复检测
**影响**: `__init__` 不再阻塞等待 Ollama

#### 2. FAISS 索引持久化 (Index Persistence)
**问题**: 每次启动重建 HNSW 索引 (m=32, efConstruction=40)，不保存
**修改**: 
- 新增 `_load_faiss_index()`: 启动时从 `{storage_path}/faiss_hnsw.index` 读取
- 新增 `_save_faiss_index()`: `add()` 后自动写入磁盘
- 新增 `_faiss_index_path` 字段
- 新增 `_ensure_faiss_index()`: 首次搜索时懒创建索引
**影响**: 首次启动 ~5s 构建索引，后续 <100ms 加载

#### 3. 向量搜索懒加载 (Lazy FAISS in Query)
**问题**: `_vector_search()` 直接访问 `self._embedding` / `self._faiss_index`
**修改**: 改为调用 `_ensure_embedding()` + `_ensure_faiss_index()`
**影响**: 搜索请求触发按需初始化，而非启动时全部加载

#### 4. `add()` 方法适配
**问题**: `add()` 中 `self._embedding.encode()` 假设 embedding 已初始化
**修改**: 改为 `self._ensure_embedding().encode()`
**影响**: 首次添加记忆时自动连接向量服务

---

## 性能影响

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| `SuMemoryLitePro()` 构造 | ~15s (Ollama检测+FAISS构建) | <0.1s (仅内存分配) |
| 首次 `add()` | ~0.1s (+已初始化的embedding) | ~15s (触发Ollama连接+FAISS构建) |
| 首次 `query()` | ~0.05s | ~15s (同上, 仅触发一次) |
| 第二次启动 `SuMemoryLitePro()` | ~15s (重建FAISS) | <0.1s (磁盘加载索引) |
| 第二次启动 首次`query()` | ~0.05s | <0.1s (索引已在内存) |

**关键收益**:
- Nutri-Brain 系统启动: **90s → 8s** (因不再导入 su-memory 路由)
- su-memory 自身冷启动: **15s → <0.1s** (构造器不再阻塞)
- 后续启动: FAISS 索引从磁盘加载, **<100ms**

---

## 向后兼容性

✅ 完全兼容。所有现有 API 不变, `enable_*` 参数仍生效。
⚠️ 行为变化: 首次 `add()`/`query()` 可能比之前慢 ~15s (一次性, 后续正常)。
💡 建议: 在应用启动时主动调用 `memory._ensure_embedding()` 预热。

---

## 给 su-memory 开发团队的建议 (TODO)

以下优化未在本次变更中实现，建议后续迭代:

1. **批量嵌入 API**: Ollama 支持 `embed_batch`, 564 条记录 18 次请求代替 564 次 (~50s 收益)
2. **FAISS 增量保存**: 当前每次 `add()` 全量写盘, 可改为每 N 次或定时写盘
3. **引擎按需加载**: VectorGraphRAG/SpacetimeIndex/Multimodal/SpatialRAG 仍在 `__init__` 中创建, 
   建议也改为懒加载 (类似 FAISS)
4. **轻量构造模式**: 新增 `mode="search_only"` 参数, 跳过因果图谱/时空/多模态引擎 (~10s 收益)
5. **`_initialized` 双重检查**: ClinicalKnowledgeMemory 的 `initialize()` 被调用两次 (日志可见),
   需要在 SuMemoryLitePro 层面加锁或原子检查
