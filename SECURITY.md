# 安全策略

## 报告安全漏洞

我们非常重视项目安全。如果您发现任何安全漏洞，请通过以下方式报告：

**邮箱**：sandysu737@gmail.com

**注意**：请勿在 GitHub Issues 中公开报告安全问题。

---

## 安全最佳实践

### 1. API密钥管理

```python
# ✅ 推荐：使用环境变量
import os
api_key = os.getenv("MINIMAX_API_KEY")

# ✅ 推荐：使用 .env 文件（已加入 .gitignore）
# .env 内容：
# MINIMAX_API_KEY=your-key-here

# ❌ 禁止：硬编码密钥
api_key = "sk-xxxxxx"  # 绝对禁止！
```

### 2. 本地存储

- 数据默认存储在本地 `./storage/` 目录
- 生产环境请定期备份
- 敏感数据请自行加密

### 3. 依赖安全

```bash
# 定期更新依赖
pip install --upgrade su-memory

# 检查已知漏洞
pip audit
```

---

## 漏洞响应流程

1. **接收报告**：24小时内确认收到报告
2. **评估**：3天内完成漏洞评估
3. **修复**：尽快发布安全更新
4. **公告**：同步安全公告

---

## 版本支持

| 版本 | 安全更新 | 状态 |
|------|----------|------|
| v1.3.0 | ✅ | 当前稳定版 |
| v1.2.x | ✅ | 维护中 |
| v1.1.x | ⚠️ | 仅关键修复 |

建议始终使用最新版本。

---

## 联系

- 邮箱：sandysu737@gmail.com
- GitHub：https://github.com/su-memory/su-memory-sdk

感谢您帮助我们保持项目安全！
