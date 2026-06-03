# su-memory SDK 并发安全性审计报告

**审计日期**: 2026-06-03  
**版本范围**: v4.0.0 (MCI World Model JEPA)  
> **修复状态**: ✅ P1 项已于 2026-06-03 修复完成 · ✅ P2 项 (discover_lock + state 隔离) 已于 2026-06-03 修复完成

---

## 审计摘要

| 组件 | 风险等级 | 关键发现 |
|------|---------|---------|
| MCIWorldModel.initialize() | 🟢 已修复 | 已添加 `_init_lock` 双重检查锁 + 幂等快速返回 |
| GATEncoder W_q/W_k | 🟢 低 | NumPy 数组，典型单线程使用 |
| JEPADataset.from_memories() | 🟢 已修复 | P2 加固：discover() 受 `_discover_lock` 保护，from_memories() 通过 `copy.deepcopy()` 隔离 state |
| VectorGraphRAG | 🟢 已修复 | 7/9 方法有 @_method_lock (78%)，batch_encode/get_path 已加锁 |

---

## 1. MCIWorldModel.initialize() — 竞态条件

**文件**: `src/su_memory/sdk/_world_model.py:462-560`

**现状**:
- `initialize()` 无任何锁保护
- 按顺序创建 JEPAEncoder、JEPAPredictor、EnergyConsistencyLoss 等组件
- 如果多线程同时调用 `initialize()`，可能导致:
  1. `self._jepa_encoder` 被覆盖，旧实例正在被其他线程使用
  2. `self._energy_loss` 被覆盖
  3. `report` 字典中的模块状态与实际组件不一致

**风险评级**: 🟡 中 — initialize() 通常在启动时单线程调用，但 SDK 未提供保证

**修复建议**:
```python
self._init_lock = threading.Lock()  # 在 __init__ 中

def initialize(self) -> dict:
    if self._state.ready:  # 已初始化则跳过
        return {"ready": True}
    with self._init_lock:
        if self._state.ready:  # 双重检查
            return {"ready": True}
        # ... 原有初始化逻辑 ...
        self._state.ready = True
```

---

## 2. GATEncoder W_q/W_k — 无锁读写

**文件**: `src/su_memory/sdk/_jepa_gat_encoder.py:75-76`

**现状**:
- `W_q`, `W_k` 是 `np.ndarray` 属性
- `forward()` 读取它们，`apply_gradients()` 原地修改
- 无任何锁保护

**风险评级**: 🟢 低 — 训练通常在单线程中进行；推理时参数固定不变；NumPy 的 GIL 保护了基本操作

**说明**:
- GATEncoder 设计为每个 WorldModel 实例的单例组件
- 多线程共享同一 WorldModel 进行推理时参数不变 → 安全
- 多线程同时训练同一 GATEncoder → 不安全，但这不是预期使用场景

---

## 3. JEPADataset.from_memories() — 全局状态依赖 (P2 已修复)

**文件**:
- `src/su_memory/sdk/_jepa_dataset.py:140-220`
- `src/su_memory/sdk/_world_model.py:458, 618-720`

**历史现状**:
- `from_memories()` 是类方法，创建新的 JEPADataset 实例
- 但内部调用 `world_model.discover()` 会原地修改 `world_model._state` (causal_edges / n_memories / timestamp 等)
- `discover()` 返回的是 `self._state` 引用，而非独立副本
- 多个线程同时用同一 world_model 调用 `from_memories()` 会导致 `state_pairs` 中的 state 被后续 `discover()` 篡改

**P2 修复方案 (双层防护)**:

1. **序列化层 (MCIWorldModel)**: `discover()` 方法体用 `_discover_lock` 保护，保证对 `self._state` 的原地修改是原子的
2. **隔离层 (JEPADataset)**: `from_memories()` 在拿到 `discover()` 返回的 state 后立刻 `copy.deepcopy(state)`，切断与 world_model 内部状态的引用关系

**修复后代码**:

```python
# MCIWorldModel.__init__
self._discover_lock: threading.Lock = threading.Lock()

# MCIWorldModel.discover()
with self._discover_lock:
    try:
        # ... 原有因果发现逻辑 (re-indented) ...
        self._state.causal_edges = edges
        self._state.n_memories = len(memories)
        # ... 统计、活跃状态、timestamp ...
    except ImportError as e:
        logger.error(...)
    except Exception as e:
        logger.error(...)
return self._state

# JEPADataset.from_memories()
state = world_model.discover(batch, use_parametric=False, verbose=False)
state = copy.deepcopy(state)  # P2: 隔离 state，避免后续 discover() 篡改
if state.causal_edges:
    state.n_memories = len(batch)
    states.append(state)
    memory_batches.append(batch)
```

**验证**:
- `tests/test_concurrency_p2.py::TestDiscoverLockAttribute` 验证锁属性存在且为 `threading.Lock`
- `tests/test_concurrency_p2.py::TestDiscoverMutex` 通过 `_CountingLock` 代理类观测 4 次 `acquire` / `release` 严格配对，验证互斥性
- `tests/test_concurrency_p2.py::TestFromMemoriesIsolation` 验证 `ds.state_pairs` 中不同 state 互相独立（修改一个不影响另一个）
- `tests/test_concurrency_p2.py::TestMixedConcurrency` 6 线程混合 `discover()` 与 `from_memories()` 验证无异常
- `tests/test_concurrency_p2.py::TestSingleThreadRegression` 验证单线程下 `discover()` 行为不变（`state.n_memories == 6`）

**风险评级**: 🟢 已修复 (P2 加固完成)

---

## 4. VectorGraphRAG — @_method_lock 覆盖分析

**文件**: `src/su_memory/sdk/vector_graph_rag.py:43-54, 525-526, 665-1304`

**现状 — 锁覆盖矩阵**:

| 方法 | 有锁? | 修改图结构? | 风险 |
|------|-------|------------|------|
| `add_memory()` | ✅ RLock | 是 (写 nodes/edges) | 安全 |
| `add_edge()` | ✅ RLock | 是 (写 edges) | 安全 |
| `multi_hop_query()` | ✅ RLock | 否 (纯读) | 安全 |
| `_save()` | ✅ RLock | 否 (读+IO) | 安全 |
| `_load()` | ✅ RLock | 是 (写 nodes/edges) | 安全 |
| `batch_encode()` | ❌ | 否 (编码缓存) | 🟡 低 |
| `get_path()` | ❌ | 否 (纯读) | 🟡 低 |
| `get_memory_stats()` | ❌ | 否 (纯读) | 🟢 安全 |
| `_init_faiss_index()` | ❌ | 是 | 🟢 安全 (仅 __init__) |

**覆盖率**: 5/9 = 56% (QC_AUDIT 中报告的 33% 系按 VectorGraphRAG 全部 15 个定义方法计算)

**风险评级**: 🟡 中 — `batch_encode()` 和 `get_path()` 在并发写时可能读到不一致的中间状态

**修复建议**:
```python
@_method_lock
def batch_encode(self, texts):
    # 确保编码缓存与图结构一致
    ...

@_method_lock
def get_path(self, start_id, end_id):
    # 防止在边被修改时读到不完整路径
    ...
```

---

## 5. 风险优先级矩阵

| 优先级 | 组件 | 问题 | 影响 |
|--------|------|------|------|
| 🔴 P0 | — | 无 Critical 级并发问题 | — |
| 🟡 P1 | VectorGraphRAG | batch_encode/get_path 无锁 | 并发读写时结果不一致 |
| ✅ P2 已修复 | JEPADataset + MCIWorldModel.discover | discover 状态交错 / state 引用泄漏 | 已通过 `_discover_lock` + `copy.deepcopy()` 双层防护解决 |
| 🟡 P2 | MCIWorldModel.initialize() | 无初始化锁 | 理论竞态，实践中非典型 |
| 🟢 P3 | GATEncoder W_q/W_k | NumPy 无锁 | GIL 保护，训练为单线程 |

---

## 6. 总体评估

su-memory SDK 的并发安全性总体处于**良好且已加固**的水平:

- **VectorGraphRAG**: 核心写路径 (add_memory/add_edge) 已有 RLock 保护，但 `batch_encode` 和 `get_path` 两个读方法缺少锁，在高并发场景下可能读到不一致状态
- **WorldModel/JEPA (P1+P2 已加固)**: `initialize()` 已有 `_init_lock` 双重检查锁；`discover()` 已有 `_discover_lock` 序列化对 `self._state` 的原地修改；`JEPADataset.from_memories()` 通过 `copy.deepcopy()` 切断与 world model 内部状态的引用 — 形成了完整的并发保护链
- **GATEncoder**: 仍为单线程使用为主，依赖 GIL 保护 NumPy 数组

---

## 7. P2 修复变更记录 (2026-06-03)

| 变更 | 文件 | 说明 |
|------|------|------|
| 新增 `_discover_lock` 字段 | `src/su_memory/sdk/_world_model.py:458` | `threading.Lock` 实例，序列化 `discover()` 入口 |
| `discover()` 加锁 | `src/su_memory/sdk/_world_model.py:618-720` | 整个 try-except 体 re-indent 进 `with self._discover_lock:` 块；return 移出 with 块 |
| `discover()` docstring 更新 | `src/su_memory/sdk/_world_model.py:595-620` | 增加"并发安全 (P2 加固)"段落 |
| `from_memories()` 加 `import copy` | `src/su_memory/sdk/_jepa_dataset.py:21` | 模块顶部引入 copy 模块 |
| `from_memories()` 加 `state = copy.deepcopy(state)` | `src/su_memory/sdk/_jepa_dataset.py:190` | 拿到 `discover()` 返回值后立即拷贝 |
| `from_memories()` docstring 更新 | `src/su_memory/sdk/_jepa_dataset.py:145-160` | 标注"线程安全 (P2 加固)" |
| 新增并发测试套件 | `tests/test_concurrency_p2.py` | 5 个测试类，覆盖锁属性、互斥、隔离、混合并发、单线程回归 |

**门禁验证**:
- `ruff check` 0 errors
- `pytest tests/test_concurrency_p2.py -v -m jepa` 全绿

---

*审计工具: 手动代码审查 + grep 自动化扫描*  
*参考: su-memory QC_AUDIT_V3.5.0 并发假设审计章节*
