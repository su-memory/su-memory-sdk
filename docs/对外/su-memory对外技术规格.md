# su-memory 企业级AI记忆中台
## 对外技术规格说明 · 2026
## ⚠️ 内部使用 · 对外不可分发

---

## 一、产品定位

su-memory是企业级AI长期记忆中台，为私有化大模型提供持久记忆、智能检索、冲突消解和主动遗忘能力。

---

## 二、核心效果指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **长期记住率** | >96% | LongMemEval基准测试 |
| **冲突检测率** | >97% | 自建冲突测试集 |
| **响应延迟P95** | <200ms | 100并发压测 |
| **召回率** | >95% | 多场景召回测试 |
| **存储节省** | >80% | 对比无压缩方案 |
| **可用性** | 99.95% | 7×24小时运行 |

---

## 三、接口规范

### 3.1 基础接口

```yaml
POST /v1/tenant/create           # 创建租户
POST /v1/memory/add               # 写入记忆
POST /v1/memory/query             # 检索记忆
POST /v1/memory/update            # 更新记忆
POST /v1/memory/delete            # 删除记忆
POST /v1/chat/completions         # 带记忆的对话
GET  /v1/memory/stats             # 记忆统计
GET  /v1/health                   # 健康检查
```

### 3.2 请求/响应格式

**POST /v1/memory/add**
```json
{
  "user_id": "string",
  "content": "string",
  "metadata": {
    "type": "fact|preference|event|belief",
    "priority": 0-10
  }
}
```

**POST /v1/memory/query**
```json
{
  "user_id": "string",
  "query": "string",
  "limit": 8
}
```

**响应**
```json
{
  "memories": [
    {
      "id": "string",
      "content": "string",
      "relevance": 0.95,
      "timestamp": "2026-04-01"
    }
  ]
}
```

---

## 四、部署要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| CPU | 4核 | 8核+ |
| 内存 | 8GB | 16GB+ |
| 存储 | 50GB SSD | 200GB SSD |
| GPU | 可选 | NVIDIA GPU（加速检索） |
| Docker | 20.10+ | 20.10+ |
| 网络 | 千兆 | 千兆 |

---

## 五、支持的模型

| 类别 | 支持的模型 |
|------|----------|
| 本地Ollama | qwen、llama、deepseek等全部Ollama模型 |
| vLLM | 所有vLLM支持的模型 |
| 国产私有大模型 | 通义千问、文心、Qwen-any |
| OpenAI兼容 | 所有OpenAI API兼容接口 |

---

## 六、安全与合规

| 项目 | 支持情况 |
|------|---------|
| 多租户数据隔离 | ✅ 完全隔离 |
| 私有网络部署 | ✅ 支持 |
| 审计日志 | ✅ 完整记录 |
| 等保三级 | ✅ 支持 |
| 医疗数据合规 | ✅ 支持 |
| 数据不出本地 | ✅ 完全私有 |

---

## 七、部署流程

```bash
# 1. 拉取镜像
docker pull su-memory:latest

# 2. 配置环境变量
cp .env.example .env
vim .env

# 3. 启动服务
docker-compose up -d

# 4. 验证
curl http://localhost:8000/v1/health
```

---

## 八、技术支持

| 项目 | 标准版 | 企业版 |
|------|-------|-------|
| 部署支持 | 远程1次 | 现场部署 |
| 故障响应 | 工作日 | 7×24 |
| 版本更新 | 每年4次 | 按需 |
| 定制开发 | ❌ | ✅ |

---

## 九、已知限制

| 限制 | 说明 | 解决方案 |
|------|------|---------|
| 单用户记忆上限 | 100万条 | 分桶+归档 |
| 单条记忆长度 | 10万字符 | 自动摘要 |
| 向量模型依赖 | 需匹配 Embedding 模型 | 可配置 |

---

## 十、版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| 1.0.0 | 2026-04 | 初始商用版本 |
