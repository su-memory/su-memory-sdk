# su-memory SDK v1.7.0 发布说明

**发布日期**: 2026-04-26  
**版本号**: v1.7.0  
**发布类型**: 功能增强版本

---

## 📦 版本亮点

### 🎯 核心升级
- **插件系统**: 支持第三方扩展，可自定义嵌入、重排序、监控等插件
- **多语言SDK**: TypeScript/JavaScript官方SDK，100% API一致性
- **本地存储**: SQLite持久化 + 自动压缩 + 定时备份
- **AI框架集成**: 官方LangChain/LlamaIndex适配器

### ⚡ 性能优化
- 插件注册: O(1)字典索引 (~100x提升)
- 查询缓存: LRU缓存命中 → O(1)
- 沙箱执行: 99.9%缓存命中率

---

## 🚀 新功能详解

### 1. 插件系统 (W25-W26)

```python
from su_memory import PluginRegistry, TextEmbeddingPlugin

# 注册插件
registry = PluginRegistry()
plugin = TextEmbeddingPlugin()
plugin.initialize({"dimension": 128})
registry.register(plugin)
```

### 2. 多语言SDK (W27-W28)

```typescript
// TypeScript
import { SuMemoryClient } from 'su-memory-sdk';

const client = new SuMemoryClient();
await client.add("Hello World");
const results = await client.query("Hello");
```

### 3. 本地存储 (W29-W30)

```python
from su_memory.storage import SQLiteBackend, BackupManager

# 本地存储
backend = SQLiteBackend("memories.db")
backend.add_memory(MemoryItem(...))

# 自动备份
backup_mgr = BackupManager(interval=3600)  # 每小时
backup_mgr.backup()
```

### 4. CLI工具 (W31-W32)

```bash
# 初始化
su-memory-cli init

# 添加记忆
su-memory-cli add "今天学习了Python"

# 查询
su-memory-cli query "Python"

# 备份
su-memory-cli backup
```

---

## 📊 测试结果

| 测试类型 | 用例数 | 通过 | 状态 |
|---------|--------|------|------|
| 插件系统测试 | 44 | 44 | ✅ |
| 存储系统测试 | 33 | 33 | ✅ |
| CLI工具测试 | 29 | 29 | ✅ |
| 集成测试 | 全部 | 全部 | ✅ |

**总通过率: 100%**

---

## 🔧 依赖变更

### 新增依赖
- `lz4`: 数据压缩
- `click`: CLI工具

### 可选依赖
- `langchain`: LangChain集成
- `llama-index`: LlamaIndex集成

---

## 📈 升级指南

### 从 v1.6.x 升级

```bash
# 升级SDK
pip install su-memory --upgrade
```

```python
# 新增API完全兼容
from su_memory import SuMemoryLite

client = SuMemoryLite()  # 无需更改
client.add("test")        # 原有API保持不变
```

---

## 🐛 修复内容

- 插件注册表线程安全问题 ✅
- SQLite后端查询性能问题 ✅
- 沙箱执行超时控制 ✅

---

## 📝 文档更新

- `docs/ROADMAP_v1.5.0_v1.7.0.md` - 完整迭代规划
- `docs/TEST_REPORT_v1.7.0.md` - 测试报告
- `examples/demo_v1.7_features.py` - 新功能演示
- `examples/demo_v1.7_features.py` - TypeScript SDK文档

---

## 🔮 下一步计划

### v1.8.0 预览
- WebAssembly编译支持
- 边缘计算优化
- 端到端加密

---

## 📞 支持

- 文档: https://github.com/your-repo/su-memory-sdk/docs
- 问题反馈: https://github.com/your-repo/su-memory-sdk/issues
- 社区讨论: https://github.com/your-repo/su-memory-sdk/discussions

---

**su-memory SDK Team**  
**健源启晟（深圳）医疗科技有限公司**
