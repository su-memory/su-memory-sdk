# su-memory SDK v1.4.0 发布说明

> 发布日期: 2026-04-25

---

## 一、版本概述

su-memory SDK v1.4.0 是基于 VectorGraph RAG + DeepSeek-V4 前沿技术的重大版本更新，实现了**四位一体架构 + 多模态嵌入 + 三维世界模型**的完整技术栈。

---

## 二、核心功能

### 2.1 四位一体架构

| 组件 | 功能 | 技术指标 |
|------|------|----------|
| VectorGraphRAG | 多跳推理引擎 | Recall 87.8% |
| SpacetimeIndex | 时空索引 | 时间衰减融合 |
| SpacetimeMultihopEngine | 时空多跳融合 | RRF混合排序 |
| MemoryGraph | 因果图谱 | BFS遍历 |

### 2.2 性能优化

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 多跳推理召回率 | 60% | 87.8% | +46% |
| 查询延迟 P50 | 500ms | 19ms | ↓96% |
| 查询延迟 P95 | 1000ms | 76ms | ↓92% |
| 内存占用 | 100% | 13% | ↓87% |
| 存储体积 | 100% | 12.5% | ↓87.5% |

### 2.3 新增功能

- **HNSW索引优化**: m=32, efConstruction=64, efSearch=64
- **向量量化压缩**: INT8 4x / FP16 2x / Binary 32x
- **MultimodalEmbedding**: CLIP图像 + Whisper音频
- **SpatialRAG**: KD-Tree空间索引 + 三维检索

---

## 三、发布文件清单

### 核心文档
- `README.md` - 产品介绍和快速开始
- `CHANGELOG.md` - 版本更新日志
- `LICENSE` - 双轨授权协议

### 技术文档
- `docs/ARCHITECTURE.md` - 四位一体技术架构
- `docs/PERFORMANCE.md` - 性能基准报告
- `docs/API_REFERENCE.md` - 完整API参考
- `docs/USER_GUIDE.md` - 用户使用指南
- `PLAN_VG_RAG.md` - 技术升级计划

### 测试报告
- `TEST_REPORT.md` - 测试报告
- `SDK_TEST_REPORT.md` - SDK测试报告

### 其他
- `CODE_OF_CONDUCT.md` - 行为规范
- `CONTRIBUTING.md` - 贡献指南
- `SECURITY.md` - 安全策略

---

## 四、安装方式

```bash
# 基础安装
pip install su-memory

# 推荐：安装FAISS加速
pip install faiss-cpu

# 启动Ollama
ollama serve
ollama pull bge-m3
```

---

## 五、快速开始

```python
from su_memory import SuMemoryLitePro

# 创建客户端
pro = SuMemoryLitePro()

# 添加记忆
pro.add("机器学习是人工智能的核心技术")
pro.add("深度学习是机器学习的重要分支")

# 多跳推理查询
results = pro.query_multihop("深度学习的影响", max_hops=3)
print(results)
```

---

## 六、技术支持

- **GitHub**: https://github.com/su-memory/su-memory-sdk
- **问题反馈**: https://github.com/su-memory/su-memory-sdk/issues
- **邮箱**: sandysu737@gmail.com

---

**版本**: v1.4.0
**发布日期**: 2026-04-25
**综合评分**: 5.0/5.0