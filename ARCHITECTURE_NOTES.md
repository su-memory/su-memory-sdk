# 架构改进建议 (未来重构, 非紧急)

> 本文件记录评估中发现的架构改进点, 当前不执行, 供未来迭代参考。

## 1. lite_pro.py 拆分 (4168 行 → 6 个模块)

`lite_pro.py` 承载了 6 个类, 建议拆分为独立模块:

```
sdk/
  _memory_graph.py        # MemoryGraph, Edge, MemoryNode (边缘置信度, 因果图)
  _temporal_system.py     # TemporalSystem (时间码, 能量强度, 衰减)
  _prediction.py          # PredictionModule (贝叶斯预测, 事件序列)
  _explainability.py      # ExplainabilityModule (推理树, 可解释性)
  _session.py             # SessionManager (会话管理, 主题交叉)
  lite_pro.py             # SuMemoryLitePro (核心 API, 组合上述模块)
```

**不紧急的原因**: 当前单文件可工作, 1020 测试通过, 拆分有引入回归的风险。
建议在下一个大版本 (v5.0) 时执行, 配合完整的集成测试。

## 2. vector_graph_rag.py vs spatial_rag.py 功能重叠

- `vector_graph_rag.py` (1447 行): 向量+图 RAG
- `spatial_rag.py` (674 行): 空间 RAG

两者功能重叠, 建议合并或明确职责边界。

## 3. benchmark 结果归档策略

- `benchmarks/results/archive/` 已归档 74 个旧文件
- 建议建立 CI 策略: 每次评测只保留最新结果 + MD 报告
- 旧 JSON 自动归档到 `archive/`

## 4. 测试改进

- `test_bayesian_system.py` 有 `PytestReturnNotNoneWarning` (测试函数返回 bool 而非 assert)
- `test_pg_redis_integration.py` 需要外部服务, 建议标记 `@pytest.mark.integration`
