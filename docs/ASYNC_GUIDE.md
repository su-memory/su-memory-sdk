# su-memory 异步 API 指南 (v2.7.0)

su-memory v2.7.0 新增完整的异步客户端 `AsyncSuMemory`，支持异步写入、查询、流式输出和并发操作。

## 快速开始

```bash
pip install su-memory[async]  # 含 httpx + openai 异步依赖
```

```python
import asyncio
from su_memory.async_client import AsyncSuMemory

async def main():
    # 创建异步客户端
    client = await AsyncSuMemory.create()

    # 异步写入
    mid = await client.aadd("项目ROI增长了25%", metadata={"source": "finance"})

    # 异步查询
    results = await client.aquery("投资回报", top_k=5)
    for r in results:
        print(f"[{r.score:.2f}] {r.memory.get('content')}")

    # 流式查询
    async for chunk in client.astream_query("项目进展"):
        print(f"{chunk.type}: {chunk.data}")

    await client.aclose()

asyncio.run(main())
```

## AsyncSuMemory API 参考

### 创建客户端

```python
client = await AsyncSuMemory.create(
    mode="local",           # "local" / "server"
    storage="sqlite",       # "sqlite" / "memory" / "pgvector"
    persist_dir="/path",    # 持久化目录
    embedder=None,          # 自定义嵌入服务
)
```

或使用上下文管理器：

```python
async with await AsyncSuMemory.create() as client:
    await client.aadd("自动关闭的资源管理")
```

### 核心方法

| 方法 | 类型 | 说明 |
|------|------|------|
| `aadd(content, metadata)` | I/O+CPU | 添加单条记忆 |
| `aadd_batch(items)` | I/O+CPU | 批量添加 |
| `aquery(text, top_k)` | I/O+CPU | 语义检索 |
| `aquery_multihop(text, max_hops)` | I/O+CPU | 多跳推理 |
| `astream_query(text, top_k)` | 流式 | 流式查询 (SSE) |
| `apredict(query)` | I/O+CPU | 预测 |
| `aforget(memory_id)` | CPU | 删除记忆 |
| `adecay(days)` | CPU | 时间衰减 |
| `aclear()` | CPU | 清空 |
| `aclose()` | I/O | 关闭资源 |

### CPU vs I/O 分离

v2.7.0 异步架构的核心设计原则：

- **I/O 密集型**（嵌入API、数据库）：原生 `async/await`
- **CPU 密集型**（FAISS、编码器、因果推理）：`asyncio.to_thread()`

这意味着异步客户端在高并发场景下不会阻塞事件循环。

## 流式查询 (SSE)

`astream_query()` 返回 `AsyncIterator[StreamChunk]`，支持逐条接收结果：

```python
async for chunk in client.astream_query("项目进展", top_k=10):
    if chunk.type == "partial":
        print(f"部分结果: {chunk.data}")
        print(f"进度: {chunk.progress:.0%}")
    elif chunk.type == "complete":
        print(f"查询完成: {chunk.data}")
    elif chunk.type == "error":
        print(f"错误: {chunk.data}")
```

### FastAPI SSE 端点

```python
from su_memory._sys._stream import to_sse
from fastapi.responses import StreamingResponse

@app.get("/query/stream")
async def stream_query(q: str, top_k: int = 5):
    async def _stream():
        async for event in to_sse(client.astream_query(q, top_k)):
            yield event
    return StreamingResponse(_stream(), media_type="text/event-stream")
```

### CLI 流式查询

```bash
su-memory stream-query "投资回报" --top-k 10
```

## 异步嵌入后端

`AsyncEmbeddingFactory` 自动检测可用后端（优先级：Ollama → OpenAI → MiniMax → s-t → TF-IDF）：

```python
from su_memory._sys._async_embedder import AsyncEmbeddingFactory

embedder = await AsyncEmbeddingFactory.create("auto")
vecs = await embedder.aembed(["文本1", "文本2"])
```

支持的异步嵌入后端：

| 后端 | 类型 | 默认维度 | 依赖 |
|------|------|:---:|------|
| `OllamaAsyncEmbedder` | I/O (HTTP) | 768 | httpx |
| `OpenAIAsyncEmbedder` | I/O (API) | 1536 | openai |
| `MiniMaxAsyncEmbedder` | I/O (API) | 1536 | openai |
| `SentenceTransformersAsyncEmbedder` | CPU | 384 | s-t |
| `TfidfAsyncEmbedder` | CPU | 256 | sklearn |

## 并发查询最佳实践

```python
async def concurrent_queries(client, queries):
    tasks = [client.aquery(q, top_k=5) for q in queries]
    results = await asyncio.gather(*tasks)
    return results

# 100 个并发查询
results = await concurrent_queries(client, ["查询"] * 100)
```

## 从同步 API 迁移

```python
# v2.6.0 (同步)
client = SuMemory()
mid = client.add("内容")
results = client.query("查询")

# v2.7.0 (异步) — 只需加 'a' 前缀 + await
client = await AsyncSuMemory.create()
mid = await client.aadd("内容")
results = await client.aquery("查询")
```

所有方法命名规则：同步 `method()` → 异步 `amethod()`。同步 `SuMemory` API 完全保留，零破坏性变更。
