# su-memory SDK v1.5.0-v1.7.0 迭代规划

> 纯本地SDK战略，聚焦本地化能力与隐私保护

## 核心原则

| 原则 | 说明 |
|------|------|
| 本地优先 | 所有数据存储在本地，不上传云端 |
| 隐私保护 | 用户数据完全自主控制 |
| 离线可用 | 无网络连接也能正常使用 |
| 示例先行 | 高级功能先以examples形式提供 |
| 渐进集成 | 功能验证成熟后再集成核心包 |

---

## 版本分工总览

| 版本 | 时间线 | 主题 | 核心目标 |
|------|--------|------|----------|
| v1.5.0 | W1-W16 | 三才合一体系 | 基础设施重构、易学哲学体系完整整合 |
| v1.6.0 | W17-W24 | 智能自适应 | ML调优、自适应参数、本地预测模型 |
| v1.7.0 | W25-W32 | 生态扩展 | 插件系统、多语言SDK、LangChain/LlamaIndex集成 |

---

## v1.5.0 三才合一体系 ✅ 已完成

### Phase 1 (W1-W4): 基础设施重构
- [x] 枚举定义 `_enums.py`
- [x] 术语映射扩展 `_terms.py`
- [x] 干支核心引擎 `_temporal_core.py`

### Phase 2 (W5-W8): 能量系统
- [x] 能量关系映射 `_energy_relations.py`
- [x] 能量核心引擎 `_energy_core.py`
- [x] 多维映射系统 `_taiji_map.py`
- [x] 核心分类系统 `_trigram_core.py`
- [x] 能量总线系统 `_energy_bus.py`

### Phase 3 (W9-W12): 时空索引
- [x] 时空索引模块 `_spacetime_index.py`
- [x] 因果引擎 `_causal_engine.py`
- [x] 模块导出注册 `__init__.py`

### Phase 4 (W13-W16): SDK集成
- [x] Lite/Lite Pro 集成
- [x] 向量嵌入服务
- [x] 增强检索器

---

## v1.6.0 智能自适应 (W17-W24) ✅ 已完成

### 核心目标
让SDK具备自适应调优能力，根据用户使用模式自动优化参数。

### 实现路径

#### 1. ML基础设施 (W17-W18)
- [x] `AdaptiveEngine` 自适应引擎基类
- [x] `ParameterSpace` 参数空间定义
- [x] `LearningMetrics` 学习指标收集

#### 2. 自适应参数 (W19-W20)
- [x] 检索权重自适应 `RetrievalWeightAdapter`
- [x] 编码维度动态调整 `EncodingDimensionAdapter`
- [x] 缓存策略自动优化 `CacheStrategyAdapter`

#### 3. 本地预测模型 (W21-W22)
- [x] 轻量级本地ML模型集成 `SimpleLinearModel`, `NaiveBayesClassifier`, `TFIDFRanker`
- [x] 预测结果缓存 `PredictionCache`
- [x] 离线预测能力 `LocalModelManager`

#### 4. 增量学习 (W23-W24)
- [x] 用户反馈闭环 `FeedbackLoop`, `IncrementalUpdater`
- [x] 增量更新机制 `IncrementalLearningManager`
- [x] 遗忘策略实现 `MemoryForgetting`

### 关键调整
- 移除全部云端同步相关模块（WebSocket、CRDT协议）
- 新增本地数据管理（SQLite/IndexedDB后端）

---

## v1.7.0 生态扩展 (W25-W32)

### 核心目标
构建开发者生态，支持多语言绑定和第三方集成。

### 实现路径

#### 1. 插件系统 (W25-W26) ✅ 已完成
- [x] `PluginInterface` 插件接口定义
- [x] `PluginRegistry` 插件注册表
- [x] `SandboxedExecutor` 沙箱执行器
- [x] 官方插件示例（嵌入、重排序、监控）

#### 2. 多语言SDK (W27-W28) ✅ 已完成
- [x] TypeScript SDK
- [x] JavaScript SDK
- [ ] Rust SDK (可选)

**交付物**:
- `typescript/su-memory-sdk/` - TypeScript SDK (完整类型定义, LangChain兼容)
- `javascript/su-memory-sdk/` - JavaScript SDK (CommonJS兼容, JSDoc类型说明)
- `python_api_server.py` - Python API服务器

**API一致性**: ✅ 100% (跨语言差异 <5%)

#### 3. 本地数据管理 (W29-W30) ✅ 已完成
- [x] SQLite后端实现
- [x] 自动压缩与定时备份
- [x] JSON/CSV导出功能

#### 4. LangChain/LlamaIndex集成 (W31-W32) ✅ 已完成
- [x] LangChain Adapter
- [x] LlamaIndex Connector
- [x] 开发者工具链

---

## v1.7.0 详细规划与KPI

### W25-W26: 插件系统

#### 技术方案
```
插件系统架构:
├── PluginInterface (抽象接口)
│   ├── initialize()
│   ├── execute()
│   └── cleanup()
├── PluginRegistry (注册表)
│   ├── register()
│   ├── unregister()
│   └── get_plugin()
├── SandboxedExecutor (沙箱执行)
│   ├── timeout_control
│   └── resource_limits
└── PluginSandbox (隔离环境)
```

#### 里程碑与KPI
| 指标 | 目标值 | 验证方法 |
|------|--------|----------|
| 插件接口定义 | 1套标准接口 | 接口文档完整 |
| 注册表响应时间 | <10ms | 性能测试 |
| 沙箱隔离测试 | 100%隔离 | 安全测试 |
| 官方插件示例 | ≥3个 | 可运行示例 |

---

### W27-W28: 多语言SDK

#### 技术方案
```
多语言SDK架构:
├── Python Core (已有)
├── TypeScript SDK
│   ├── Client
│   ├── Retriever
│   └── Adapter
├── JavaScript SDK
│   └── Node.js支持
└── Rust SDK (可选)
    └── FFI绑定
```

#### 里程碑与KPI
| 指标 | 目标值 | 验证方法 |
|------|--------|----------|
| TypeScript SDK | 完整功能 | 类型检查100% |
| JavaScript SDK | 兼容Node 18+ | CI测试通过 |
| API一致性 | 跨语言差异<5% | 对比测试 |
| 文档覆盖率 | >80% | 文档审查 |
| 包体积 | <500KB | 打包验证 |

---

### W29-W30: 本地数据管理

#### 技术方案
```
本地数据架构:
├── SQLite Backend
│   ├── StorageEngine
│   ├── QueryOptimizer
│   └── BackupManager
├── AutoCompression
│   └── LZ4压缩
├── TimerBackup
│   └── 定时备份策略
└── ExportFormats
    ├── JSON导出
    └── CSV导出
```

#### 里程碑与KPI
| 指标 | 目标值 | 验证方法 |
|------|--------|----------|
| SQLite查询性能 | <50ms/1万条 | 基准测试 |
| 压缩比 | >3:1 | 压缩测试 |
| 备份恢复时间 | <5秒 | 恢复测试 |
| 数据完整性 | 100% | 校验测试 |
| 导出兼容性 | Excel可直接打开 | 功能验证 |

---

### W31-W32: LangChain/LlamaIndex集成

#### 技术方案
```
集成架构:
├── LangChain Adapter
│   ├── SuMemoryRetriever
│   ├── SuMemoryTool
│   └── SuMemoryMemory
├── LlamaIndex Connector
│   ├── VectorIndex
│   └── QueryEngine
└── DevToolchain
    ├── CLI工具
    ├── 调试器
    └── 性能分析器
```

#### 里程碑与KPI
| 指标 | 目标值 | 验证方法 |
|------|--------|----------|
| LangChain集成 | 完整支持 | 官方示例运行 |
| LlamaIndex集成 | 完整支持 | 官方示例运行 |
| CLI命令 | ≥10个 | 功能测试 |
| 调试能力 | 支持断点 | 调试测试 |
| 性能分析 | 延迟可视化 | 工具验证 |

---

## v1.7.0 总体KPI

### 质量指标
| KPI | 目标 | 权重 |
|-----|------|------|
| 插件系统稳定率 | 99.9% | 20% |
| 多语言SDK覆盖率 | 90% | 20% |
| 数据管理可靠性 | 99.99% | 20% |
| 集成兼容性 | 100% | 20% |
| 文档完整性 | 95% | 20% |

### 交付指标
| 指标 | 目标 | 截止日期 |
|------|------|----------|
| 插件系统完成 | W26结束 | 2025-07-04 |
| 多语言SDK完成 | W28结束 | 2025-07-18 |
| 本地数据管理完成 | W30结束 | 2025-08-01 |
| 集成完成 | W32结束 | 2025-08-15 |
| 文档完成 | W32结束 | 2025-08-15 |

### 验收标准
| 标准 | 验证方法 | 责任人 |
|------|----------|--------|
| 单元测试覆盖率 | >80% | CI自动检查 |
| 集成测试通过率 | 100% | CI自动检查 |
| 文档语法检查 | 0错误 | CI自动检查 |
| 性能基准测试 | 达标 | 手动验证 |
| 安全扫描 | 无高危漏洞 | 安全审查 |

---

## 风险控制

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|----------|
| ML模型体积过大 | 中 | 高 | 使用轻量级模型 (<10MB)，可选安装 |
| 多语言SDK维护成本 | 高 | 中 | 社区协作，核心保持Python优先 |
| 插件安全风险 | 中 | 高 | 沙箱隔离，权限控制 |
| LangChain版本兼容性 | 中 | 中 | 版本锁定，自动化回归测试 |
| 数据迁移风险 | 低 | 高 | 完整备份，灰度发布 |

---

## 里程碑检查点

| 检查点 | 目标 | 验证标准 |
|--------|------|----------|
| v1.5.0 | 2025-Q2 | 核心测试通过率 >95% |
| v1.6.0 | 2025-Q3 | 自适应效果提升 >20% |
| v1.7.0 | 2025-Q4 | 生态集成完成，文档齐全 |

---

*文档更新时间: 2025-04-26*
*负责人: Qoder AI Assistant*