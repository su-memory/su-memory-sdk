# 迁移指南: v2.7 → v3.0.0

> **插件化体系 + 分布式存储 — 架构升级**

v3.0.0 是 su-memory 的重大架构升级。本文档指导从 v2.7.x 迁移到 v3.0.0。

---

## 目录

1. [五元素命名 → 标准英文](#1-五元素命名--标准英文)
2. [MemoryProtocol 接口变更](#2-memoryprotocol-接口变更)
3. [插件化接入指南](#3-插件化接入指南)
4. [分布式存储后端](#4-分布式存储后端)
5. [API 变更汇总](#5-api-变更汇总)
6. [常见问题](#6-常见问题)

---

## 1. 五元素命名 → 标准英文

### 变更对照表

| v2.x 旧名 | v3.0.0 新名 | 含义 |
|-----------|------------|------|
| `wood` | `semantic` | 语义维度 |
| `fire` | `causal` | 因果维度 |
| `earth` | `spacetime` | 时空维度 |
| `metal` | `generative` | 生成性/创造性 |
| `water` | `trust` | 信任/置信度 |

### 向后兼容

v3.0.0 **完全向后兼容**旧命名。所有接受能量类型参数的 API 会自动映射：

```python
# v2.x 代码在 v3.0.0 中仍正常工作
from su_memory._sys._energy_core import EnergyCore
core = EnergyCore()
state = core.get_energy_state("wood")   # 自动映射为 "semantic"
state = core.get_energy_state("fire")   # 自动映射为 "causal"
```

### 建议迁移

```python
# 旧代码
energy_type = "wood"

# 新代码（推荐）
energy_type = "semantic"
```

如果直接访问 `_sys/` 模块的常量和枚举：

```python
# 旧代码
from su_memory._sys._terms import ELEM_WOOD, ENERGY_ENHANCE

# 新代码 — 常量已更新，但旧名仍可作为别名
from su_memory._sys._terms import ELEM_SEMANTIC  # 新常量
from su_memory._sys._terms import ELEM_WOOD       # 仍可用 (别名)

# 推荐使用新名称
from su_memory._sys._energy_core import EnergyType
EnergyType.SEMANTIC  # 新名
EnergyType.WOOD      # 仍可用 (DeprecationWarning)
```

---

## 2. MemoryProtocol 接口变更

### 变更说明

`SuMemoryClient` / `SuMemoryLite` / `SuMemoryLitePro` 现在统一实现 `MemoryProtocol` 接口。

```python
from su_memory.sdk import MemoryProtocol, SuMemoryLite, SuMemoryLitePro

lite = SuMemoryLite()
assert isinstance(lite, MemoryProtocol)  # v3.0.0: True

pro = SuMemoryLitePro()
assert isinstance(pro, MemoryProtocol)  # v3.0.0: True
```

### 新增 `count()` 方法

所有 SDK 客户端现在都支持 `count()` 方法：

```python
client = SuMemoryLite()
client.add("记忆内容")
print(client.count())  # 1
```

### 统一的 `health_check()` 和 `integration_health()`

```python
# 所有客户端都支持这两个方法
health = client.integration_health()
# {"status": "healthy", "detail": "SuMemoryLite running normally", "count": 42}

health = client.health_check()
# {"status": "ok", "detail": "...", "count": 42}
```

---

## 3. 插件化接入指南

### PluginManager 使用

v3.0.0 引入了 `PluginManager`，将 `_sys/` 下 53 个模块统一管理：

```python
from su_memory.sdk.plugin_manager import PluginManager

pm = PluginManager()

# 自动发现所有插件
count = pm.auto_discover()
print(f"发现 {count} 个插件")  # 53

# 初始化所有插件
results = pm.initialize_all()
print(f"成功: {sum(1 for v in results.values() if v)}")  # 52/53

# 健康报告
report = pm.health_report()
# {"total": 53, "healthy": 52, "unhealthy": 1, "plugins": {...}}

# 热重载插件
pm.hot_reload("codec", config={})

# 获取核心插件
core = pm.get_core_plugins()
```

### 插件分类

| 类别 | PluginType | 数量 | 示例 |
|------|-----------|:---:|------|
| 核心引擎 | `EMBEDDING` | 8 | energy_bus, causal_engine, temporal_core |
| 处理管线 | `PROCESSOR` | 12 | pattern_inference, adaptive_engine, dimension_map |
| 推理/分析 | `REASONING` | 8 | bayesian, causal, multi_hop |
| 工具/编解码 | `UTILITY` | 18 | embedder, codec, migrator, fallback |
| 基础设施 | `MONITOR` | 7 | recall_trigger, priority_boost, recency_feedback |

### 不需要迁移的代码

如果你只使用 SDK 公共 API (`SuMemoryClient` / `SuMemoryLite` / `SuMemoryLitePro`)，**无需任何更改**。插件化是内部架构升级，对外 API 保持不变。

---

## 4. 分布式存储后端

### 新增后端类型

v3.0.0 支持 4 种存储后端：

| 后端 | 标识 | 依赖 | 适用场景 |
|------|------|------|---------|
| JSON 文件 (默认) | `"default"` | 零依赖 | 开发/嵌入式 |
| SQLite | `"sqlite"` | 零依赖 (标准库) | 单机/小规模 |
| PostgreSQL + pgvector | `"postgresql"` | `asyncpg`, `pgvector` | 生产级向量检索 |
| Redis | `"redis"` | `redis[hiredis]` | 超低延迟缓存 |
| 自动检测 | `"auto"` | — | 自适应 |

### 使用方式

```python
from su_memory.sdk import SuMemoryLite, SuMemoryLitePro

# 默认: JSON 文件持久化 (行为不变)
lite = SuMemoryLite()

# SQLite 后端 (零额外依赖)
lite = SuMemoryLite(storage_path="/data/memory", storage_backend="sqlite")

# 自动检测 (PG → Redis → SQLite)
lite = SuMemoryLite(storage_backend="auto")

# SuMemoryLitePro 同样支持
pro = SuMemoryLitePro(storage_path="/data/memory", storage_backend="sqlite")
```

### 安装 PostgreSQL 后端依赖

```bash
pip install su-memory[pgvector]
```

### 安装 Redis 后端依赖

```bash
pip install su-memory[redis]
```

### 完整安装

```bash
pip install su-memory[full]
```

### 配置 PostgreSQL 后端

```python
from su_memory._sys._storage_backend import StorageConfig, BackendType, create_backend

config = StorageConfig(
    pg_host="localhost",
    pg_port=5432,
    pg_database="su_memory",
    pg_user="postgres",
    pg_password="your_password",
    pg_pool_min=5,
    pg_pool_max=20,
    embedding_dim=1536,
    backend_type=BackendType.POSTGRESQL,
)

backend = await create_backend(BackendType.POSTGRESQL, config)
await backend.health_check()
```

### 配置 Redis 后端

```python
from su_memory._sys._storage_backend import StorageConfig, BackendType, create_backend

config = StorageConfig(
    redis_host="localhost",
    redis_port=6379,
    redis_db=0,
    redis_password="",
    redis_ttl=86400,  # 记忆 24 小时后过期
    embedding_dim=1536,
    backend_type=BackendType.REDIS,
)

backend = await create_backend(BackendType.REDIS, config)
await backend.health_check()
```

### 直接使用存储后端 API

```python
import asyncio
from su_memory._sys._storage_backend import StorageConfig, StorageMemory
from su_memory._sys._sqlite_storage import SqliteStorageBackend

async def main():
    config = StorageConfig(sqlite_path="/data/su_memory/storage.db")
    backend = SqliteStorageBackend(config)
    await backend.initialize()

    # 添加记忆
    await backend.add("mem_001", "今天天气很好", embedding=[0.1, 0.2, ...])

    # 向量检索
    results = await backend.query([0.1, 0.2, ...], top_k=5)

    # 健康检查
    health = await backend.health_check()

    await backend.close()

asyncio.run(main())
```

---

## 5. API 变更汇总

### 新增 API

| API | 位置 | 说明 |
|-----|------|------|
| `MemoryProtocol` | `sdk/_memory_protocol.py` | 统一接口协议 |
| `PluginManager` | `sdk/plugin_manager.py` | 插件管理器 |
| `PluginType.REASONING` | `_sys/_plugin_interface.py` | 推理类插件 |
| `PluginType.UTILITY` | `_sys/_plugin_interface.py` | 工具类插件 |
| `StorageBackend` | `_sys/_storage_backend.py` | 存储后端 ABC |
| `StorageConfig` | `_sys/_storage_backend.py` | 统一存储配置 |
| `StorageMemory` | `_sys/_storage_backend.py` | 记忆数据模型 |
| `BackendType` | `_sys/_storage_backend.py` | 后端类型枚举 |
| `BackendHealth` | `_sys/_storage_backend.py` | 健康检查模型 |
| `create_backend()` | `_sys/_storage_backend.py` | 后端工厂函数 |
| `SqliteStorageBackend` | `_sys/_sqlite_storage.py` | SQLite 后端 |
| `PgStorageBackend` | `_sys/_pg_storage.py` | PostgreSQL 后端 |
| `RedisStorageBackend` | `_sys/_redis_storage.py` | Redis 后端 |
| `SuMemoryLite(storage_backend=...)` | `sdk/lite.py` | 后端选择参数 |
| `SuMemoryLitePro(storage_backend=...)` | `sdk/lite_pro.py` | 后端选择参数 |

### 无破坏性变更

- 所有现有 API 行为不变
- 旧命名 (wood/fire/...) 仍可使用
- 默认存储方式 (JSON 文件) 不变

---

## 6. 常见问题

### Q: 升级后我的代码会崩溃吗？

**不会。** v3.0.0 完全向后兼容。除非你直接修改了 `_sys/` 内部模块的私有 API，否则无需任何更改。

### Q: 我需要迁移旧命名的代码吗？

**不需要立即迁移。** 旧命名 (wood/fire/earth/metal/water) 通过 `_normalize_energy()` 自动映射。建议在新代码中使用新命名 (semantic/causal/spacetime/generative/trust)。

### Q: 插件化会影响性能吗？

**不会。** `PluginManager` 是懒加载的 — 只有在你显式调用 `auto_discover()` 时才会初始化插件。SDK 公共 API 不经过 PluginManager。

### Q: PostgreSQL 后端比 SQLite 快多少？

PostgreSQL + pgvector 使用 IVFFlat/HNSW 向量索引，在大规模数据 (10万+) 下可实现 O(log n) 检索，而 SQLite 使用线性扫描。对于小于 1 万的记忆量，SQLite 性能足够。

### Q: 如何选择存储后端？

- **开发/测试**: 默认 JSON 或 SQLite
- **生产环境 (大规模)**: PostgreSQL + pgvector
- **超低延迟缓存**: Redis
- **不确定**: 使用 `"auto"` 自动选择

### Q: 可以混合使用多个后端吗？

可以。不同的 `SuMemoryLite` 实例可以使用不同的后端：

```python
lite_default = SuMemoryLite()  # JSON 文件
lite_sqlite  = SuMemoryLite(storage_backend="sqlite")
lite_auto    = SuMemoryLite(storage_backend="auto")  # 自动选择
```

---

## 从 v2.7.0 升级检查清单

- [ ] 阅读本迁移指南
- [ ] 验证 `isinstance(client, MemoryProtocol)` → True
- [ ] 检查是否有直接使用 `_sys/` 模块的能量类型常量（旧名仍可用）
- [ ] 探索 PluginManager: `python -c "from su_memory.sdk.plugin_manager import PluginManager; pm = PluginManager(); print(pm.auto_discover())"`
- [ ] 评估存储需求，决定是否使用新后端
- [ ] (可选) 安装 PostgreSQL/Redis 依赖
- [ ] 运行测试: `python -m pytest tests/`
- [ ] 检查 CHANGELOG 中的完整变更列表
