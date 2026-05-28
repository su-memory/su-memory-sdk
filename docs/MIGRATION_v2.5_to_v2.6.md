# 迁移指南: v2.5.x → v2.6.0

> **目标读者**: 已使用 su-memory v2.5.x 的开发者
>
> **概述**: v2.6.0 是稳定性版本，**零 Breaking Change**。所有 v2.5.x API 完全兼容，仅增强异常处理和性能。

---

## 目录

1. [变更总览](#1-变更总览)
2. [异常处理迁移](#2-异常处理迁移)
3. [降级行为变更](#3-降级行为变更)
4. [启动性能变更](#4-启动性能变更)
5. [新增可用模块](#5-新增可用模块)
6. [数据兼容性](#6-数据兼容性)
7. [常见问题](#7-常见问题)

---

## 1. 变更总览

| 维度 | v2.5.x | v2.6.0 | 影响 |
|------|--------|--------|:--:|
| 异常类型 | 裸 `RuntimeError`/`ImportError` | `SuMemoryError` + 42 `ErrorCode` | 🔄 增强 |
| 异常链 | 部分 `raise ... from e` | 100% `raise ... from e` | 🔄 增强 |
| 降级行为 | 无显式降级 | 7 组件自动降级 | ✅ 新增 |
| 启动方式 | 全部 eager import | 14 模块懒加载 | 🔄 增强 |
| 启动时间 | ~2s | 154ms | ✅ 改善 |
| FAISS 索引 | 固定参数 | HNSW/IVF 自动调参 | ✅ 新增 |
| 嵌入缓存 | 无 | LFU+TTL 线程安全 | ✅ 新增 |
| CI 门禁 | 无 | 7 大门禁 GitHub Actions | ✅ 新增 |
| API 文档 | 手动 Markdown | Sphinx + autodoc | ✅ 新增 |
| 公开 API | 不变 | 不变 | ✅ 兼容 |

---

## 2. 异常处理迁移

### 2.1 捕获异常的类型变化

**v2.5.x 写法**（仍可使用，但不推荐）:

```python
from su_memory import SuMemory

client = SuMemory()
try:
    client.add("某条记忆")
except RuntimeError as e:
    print(f"发生错误: {e}")
except ImportError as e:
    print(f"依赖缺失: {e}")
```

**v2.6.0 推荐写法**:

```python
from su_memory import SuMemory
from su_memory.exceptions import SuMemoryError, ErrorCode

client = SuMemory()
try:
    client.add("某条记忆")
except SuMemoryError as e:
    # e.code → ErrorCode.STORAGE_WRITE_FAILED
    # e.detail → "存储写入失败。磁盘空间不足或路径权限问题: /data/storage"
    # e.hint → {"path": "/data/storage"}
    # e.context → {"path": "/data/storage"}
    
    if e.code == ErrorCode.STORAGE_WRITE_FAILED:
        print(f"存储故障: {e.detail}")
        # 可根据 e.code 做精确处理
    elif e.code == ErrorCode.MEMORY_OVERFLOW:
        print(f"容量超限: 当前 {e.context.get('current')}/{e.context.get('max_memories')}")
    else:
        print(f"[{e.code.code}] {e.detail}")
```

### 2.2 ErrorCode 速查

所有 42 个错误码详见 [`src/su_memory/exceptions.py`](../src/su_memory/exceptions.py)。常用：

| ErrorCode | 含义 |
|-----------|------|
| `FAISS_DIMENSION_MISMATCH` | 向量维度不匹配 |
| `EMBED_UNAVAILABLE` | 所有嵌入后端不可用 |
| `STORAGE_READ_FAILED` | 存储读取失败 |
| `QUERY_EMPTY` | 查询文本为空 |
| `MEMORY_OVERFLOW` | 记忆数量超限 |

### 2.3 向后兼容

以下 v2.5.x 异常类 **完全保留**，但内部已改为继承 `SuMemoryError`:

```python
from su_memory.exceptions import (
    MemoryNotFoundError,   # → SuMemoryError(ErrorCode.MEMORY_NOT_FOUND)
    EncodingError,         # → SuMemoryError(ErrorCode.EMBED_CONVERSION_ERROR)
    StorageError,          # → SuMemoryError(ErrorCode.STORAGE_WRITE_FAILED)
    ConfigurationError,    # → SuMemoryError(ErrorCode.CONFIG_INVALID_PARAM)
    APIError,              # → SuMemoryError(ErrorCode.EMBED_TIMEOUT)
)

# 旧代码完全可以正常运行
try:
    ...
except StorageError as e:
    print(e)  # 输出包含 [STO_E001] 前缀
```

---

## 3. 降级行为变更

### 3.1 自动降级

v2.6.0 引入了 `FallbackChain` 降级引擎。当主路径组件不可用时，SDK 会自动尝试降级路径，无需手动干预。

**例如 — 嵌入降级**:

```
Ollama 服务不可用
  → 自动切换到 MiniMax API
    → 如仍失败，切换到 sentence-transformers (本地)
      → 如仍失败，切换到 TF-IDF (纯文本)
```

**例如 — 存储降级**:

```
Qdrant 连接失败
  → 自动切换到 SQLite (WAL 模式)
    → 如仍失败，切换到内存 Dict (会话内有效)
```

### 3.2 禁止显式降级（高级用法）

如需手动触发或禁用降级：

```python
from su_memory._sys.fallback import FallbackChain, FallbackLevel

# 自定义降级链
chain = FallbackChain("custom-embed")
chain.add_step("ollama", level=FallbackLevel.PRIMARY, func=embed_ollama)
chain.add_step("minimax", level=FallbackLevel.FALLBACK_1, func=embed_minimax)
chain.add_step("tfidf", level=FallbackLevel.GUARANTEED, func=embed_tfidf)

result = chain.try_execute(query="hello")
print(result.level)     # FallbackLevel.PRIMARY
print(result.attempts)  # 1
```

### 3.3 日志输出

降级切换时会输出日志（`logging.WARNING` 级别），建议在应用中配置日志监听：

```python
import logging
logging.basicConfig(level=logging.WARNING)
# 当触发降级时:
# WARNING:su_memory.embeddings:Ollama 不可用，降级至 MiniMax API
```

---

## 4. 启动性能变更

### 4.1 懒加载

v2.6.0 将 14 个内部模块改为懒加载（按需 `import`），启动时间从 ~2s 降至 ~154ms。

**对用户影响**: 无。所有公开 API 符号均可正常 `from su_memory import X`。

**高级用法 — 预热加载**（如果需要确保所有模块在启动时加载）:

```python
# 显式触发所有懒加载模块的实际导入
from su_memory._sys._lazy import warm_up_all
warm_up_all()  # 等同于 v2.5.x 的 eager import 行为
```

### 4.2 模块导入方式

```python
# ✅ 仍然可用（公开 API）
from su_memory import SuMemory, SuMemoryLitePro, get_version

# ✅ 仍然可用（懒加载后首次访问触发 import）
from su_memory import BackupManager, MemoryMigrator, EmbeddingFactory

# ⚠️ 不推荐（内部模块，无兼容保证）
from su_memory._sys.fallback import FallbackChain  # 可用但非公开 API
```

---

## 5. 新增可用模块

### 5.1 FAISS 自动调参 (`_sys/_faiss_tuner.py`)

```python
from su_memory._sys._faiss_tuner import FAISSAutoTuner

tuner = FAISSAutoTuner()
recommendation = tuner.recommend(n_vectors=50000)
# → {"strategy": "IVF", "nlist": 894, "nprobe": 29, "M": 32, "quantizer": None}

# 构建优化索引
index = tuner.build_index(n_vectors=50000, train_vectors=vectors)
```

### 5.2 嵌入缓存 (`_sys/_embedding_cache.py`)

```python
from su_memory._sys._embedding_cache import EmbeddingCache

cache = EmbeddingCache(max_entries=10000, ttl_seconds=3600)
vec = cache.get_or_compute("hello world", lambda: embed("hello world"))
stats = cache.get_stats()
# → {"hits": 95, "misses": 5, "hit_rate": "95.0%", "size": 5}
```

### 5.3 性能基准

```bash
# 运行性能基准（验证门禁）
python benchmarks/bench_add.py         # 写入吞吐
python benchmarks/bench_query.py       # 查询延迟
python benchmarks/bench_multihop.py    # 多跳推理
python benchmarks/stress_test.py       # 全功能压测

# CI 性能门禁检查
python scripts/check_perf_gate.py results.json
```

---

## 6. 数据兼容性

### 6.1 存储格式

**完全向后兼容**。v2.5.x 创建的 FAISS 索引、SQLite 数据库、JSON 备份均可在 v2.6.0 中直接使用，无需任何迁移操作。

### 6.2 验证方法

```bash
# 升级后验证数据完整性
python -c "
from su_memory import SuMemory
client = SuMemory(storage_path='./existing_data')
results = client.query('测试查询')
print(f'成功加载 {len(results)} 条记忆')
"
```

### 6.3 升级步骤

```bash
# 1. 升级包
pip install --upgrade su-memory==2.6.0

# 2. 验证安装
python -c "from su_memory import get_version; print(get_version())"
# → 2.6.0

# 3. 运行冒烟测试
python -c "
from su_memory import SuMemory
from su_memory.exceptions import SuMemoryError
c = SuMemory()
c.add('test')
r = c.query('test')
assert len(r) > 0, '查询失败'
print('✅ 升级成功')
"
```

---

## 7. 常见问题

### Q: 旧代码中 `except Exception as e` 捕获的行为会变吗？

**A**: 不会变。所有 `SuMemoryError` 都继承自 `Exception`，`except Exception` 仍然能捕获。但建议将关键路径升级为 `except SuMemoryError as e` 以获取结构化错误码。

### Q: `import su_memory` 变慢了怎么办？

**A**: 不会。v2.6.0 启动更快（154ms vs ~2s）。如果首次调用某模块时感觉慢，是因为该模块被懒加载了——这正是设计意图。

### Q: 降级后性能会下降吗？

**A**: 会。降级路径本身是保底方案：
- TF-IDF 降级: 准确率下降但响应极快
- numpy 线性检索: 比 FAISS 慢 ~100x（仅小数据量时触发）
- 内存 Dict 存储: 数据不持久化

降级是临时状态，建议尽快修复主路径。

### Q: 文档在哪里？

**A**:
- [Sphinx API 文档](https://su-memory.readthedocs.io) — 完整 API 参考
- [README.md](../README.md) — 快速入门
- [CHANGELOG.md](../CHANGELOG.md) — 完整更新日志
- [降级矩阵](fallback-matrix.md) — 降级路径全景

---

> **如果遇到任何迁移问题，请提交 Issue**: https://github.com/su-memory/su-memory-sdk/issues
