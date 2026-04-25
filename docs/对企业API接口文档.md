# su-memory API 接口文档

---

## 一、认证

所有API调用需要在Header中携带API Key：

```
Authorization: Bearer sk_your_api_key_here
```

---

## 二、租户接口

### POST /v1/tenant/create

创建新租户

**请求**：
```json
{
  "name": "company-name",
  "plan": "standard"  // standard | enterprise
}
```

**响应**：
```json
{
  "tenant_id": "uuid",
  "name": "company-name",
  "api_key": "sk_xxxx...",
  "created_at": "2026-04-22T00:00:00Z"
}
```

---

## 三、记忆接口

### POST /v1/memory/add

写入记忆

**请求**：
```json
{
  "user_id": "user_001",
  "content": "用户有高血压病史10年",
  "metadata": {
    "type": "fact",
    "priority": 7
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_id | string | 是 | 用户唯一标识 |
| content | string | 是 | 记忆内容（最长10万字） |
| metadata.type | string | 否 | fact/preference/event/belief |
| metadata.priority | int | 否 | 优先级0-10 |

**响应**：
```json
{
  "memory_id": "uuid",
  "status": "stored"
}
```

---

### POST /v1/memory/query

检索记忆

**请求**：
```json
{
  "user_id": "user_001",
  "query": "用户有什么慢性病",
  "limit": 8
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_id | string | 是 | 用户标识 |
| query | string | 是 | 查询内容 |
| limit | int | 否 | 返回数量，默认8 |

**响应**：
```json
{
  "memories": [
    {
      "id": "uuid",
      "content": "用户有高血压病史10年",
      "relevance": 0.95,
      "timestamp": 1713724800,
      "metadata": {}
    }
  ],
  "query_time_ms": 45.2
}
```

---

### POST /v1/memory/delete

删除记忆

**请求**：
```json
{
  "user_id": "user_001",
  "memory_id": "uuid-of-memory"
}
```

**响应**：
```json
{
  "status": "deleted"
}
```

---

### GET /v1/memory/stats/{user_id}

获取用户记忆统计

**响应**：
```json
{
  "user_id": "user_001",
  "total_memories": 156,
  "active_memories": 142,
  "archived_memories": 14,
  "storage_bytes": 245760
}
```

---

## 四、对话接口

### POST /v1/chat/completions

带记忆的对话（替代OpenAI接口）

**请求**：
```json
{
  "model": "qwen",
  "messages": [
    {"role": "system", "content": "你是专业助手"},
    {"role": "user", "content": "这个患者该怎么用药"}
  ],
  "user_id": "user_001",
  "temperature": 0.7,
  "max_tokens": 2048
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model | string | 是 | 模型名称 |
| messages | array | 是 | 对话消息 |
| user_id | string | 是 | 用户标识 |
| temperature | float | 否 | 温度参数 |
| max_tokens | int | 否 | 最大生成长度 |

**响应**：
```json
{
  "id": "chatcmpl-xxx",
  "model": "qwen",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "根据记忆，患者有高血压病史..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 120,
    "completion_tokens": 85,
    "total_tokens": 205
  }
}
```

---

## 五、健康检查

### GET /health

**响应**：
```json
{
  "status": "healthy",
  "service": "su-memory",
  "version": "1.0.0"
}
```

---

## 六、错误码

| 错误码 | 说明 |
|--------|------|
| 401 | 认证失败，检查API Key |
| 404 | 资源不存在 |
| 422 | 请求参数错误 |
| 429 | 请求频率超限 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用（检查依赖服务） |
