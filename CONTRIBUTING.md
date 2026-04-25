# Contributing to su-memory SDK

感谢您对 su-memory SDK 的关注！本文档将帮助您了解如何参与项目贡献。

---

## 📋 贡献者指南

### 1. 行为准则

请尊重所有社区成员，营造一个包容、友善的贡献环境。

### 2. 如何贡献

#### 2.1 报告问题

- 使用 GitHub Issues 报告Bug
- 提交功能请求
- 描述问题的复现步骤
- 提供环境信息（Python版本、操作系统等）

#### 2.2 提交代码

1. **Fork 仓库**
2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **提交更改**
   ```bash
   git commit -m "Add: 简短描述您的更改"
   ```
4. **推送分支**
   ```bash
   git push origin feature/your-feature-name
   ```
5. **创建 Pull Request**

#### 2.3 代码规范

- 遵循 PEP 8 代码风格
- 添加适当的注释和文档
- 确保通过所有测试
- 更新相关文档

### 3. 开发环境

```bash
# 克隆仓库
git clone https://github.com/su-memory/su-memory-sdk.git
cd su-memory-sdk

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v
```

### 4. 测试要求

- 所有新功能必须包含测试
- 所有测试必须通过
- 保持测试覆盖率

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_lite.py -v

# 生成覆盖率报告
pytest --cov=su_memory tests/
```

### 5. 文档要求

- 更新 README.md（如有必要）
- 为新功能添加文档
- 添加示例代码
- 更新 CHANGELOG.md

---

## 📝 Pull Request 模板

```markdown
## 描述
简要描述您的更改

## 更改类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 文档更新
- [ ] 代码重构

## 测试
- [ ] 测试已添加
- [ ] 测试已通过

## 检查清单
- [ ] 代码遵循PEP 8
- [ ] 文档已更新
- [ ] 无敏感信息泄露
```

---

## 🔒 授权说明

**重要**：提交代码即表示您同意您的代码将遵循项目的 LICENSE 协议。

个人用户可免费使用，但商业使用需付费授权。如有疑问，请联系 sandysu737@gmail.com。

---

## 📞 联系

- 邮箱：sandysu737@gmail.com
- GitHub Issues：https://github.com/su-memory/su-memory-sdk/issues

感谢您的贡献！
