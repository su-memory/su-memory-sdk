## 🚀 v3.5.0 — 能量中心公开API完整释放 + 代码质量加固

### ✨ 核心特性
- **能量中心公开 API 释放**：EnergyCenter、EnergyLayer、EnergyMemoryNode 完整导出
- **三重导出链验证**：sdk/__init__.py → 顶层 __init__.py → __all__ 全路径通过
- **QC 审计修复**：EnergyLayer/EnergyMemoryNode 补入 SDK __all__

### 🔧 代码质量加固
- 噪声鲁棒性验证通过
- Reflection QA 合成系统
- Entity Surfacing 实体浮现
- SIGReg 嵌入正则化

### 📊 性能指标
- 多跳推理召回率：87.8% (HotpotQA SOTA)
- 查询延迟 P50：19ms
- 启动时间：154ms（懒加载优化）

### 📦 安装
```bash
pip install su-memory==3.5.0
```

### 📋 完整更新日志
详见 CHANGELOG.md
