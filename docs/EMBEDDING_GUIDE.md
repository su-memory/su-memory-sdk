# 向量嵌入服务配置指南

su-memory SDK 支持多种向量嵌入服务，可根据环境和需求选择最适合的方案。

## 快速开始

### 方式一：自动检测（推荐）

```python
from su_memory import SuMemoryLitePro

# SDK 自动检测可用嵌入服务
pro = SuMemoryLitePro(enable_vector=True)
```

SDK 会按优先级自动选择：
1. **Ollama**（本地，推荐）
2. **OpenAI**（云端）
3. **MiniMax**（云端）
4. **本地模型**（sentence-transformers）
5. **Hash Fallback**（无服务时）

### 方式二：手动指定

```python
from su_memory.sdk.embedding import EmbeddingManager

# 指定使用 Ollama
manager = EmbeddingManager(backend="ollama", model="nomic-embed-text")

# 指定使用 OpenAI
manager = EmbeddingManager(backend="openai", model="text-embedding-3-small")

# 指定使用 MiniMax
manager = EmbeddingManager(backend="minimax", model="embo-01")
```

---

## 各服务配置详解

### 1. Ollama（推荐）

**优点**：本地运行，完全免费，保护隐私

```bash
# 1. 安装 Ollama
brew install ollama        # macOS
# 或从 https://ollama.ai 下载

# 2. 启动服务
ollama serve

# 3. 拉取嵌入模型
ollama pull nomic-embed-text    # 推荐，768维
ollama pull bge-m3              # 可选，1024维

# 4. 验证
curl http://localhost:11434/api/tags
```

```python
from su_memory.sdk.embedding import EmbeddingManager

manager = EmbeddingManager(
    backend="ollama",
    model="nomic-embed-text",        # 可选，默认 nomic-embed-text
    base_url="http://localhost:11434" # 可选
)

# 测试
vector = manager.encode("你好，世界")
print(f"向量维度: {manager.dims}")
```

**环境变量**：
```bash
export OLLAMA_BASE_URL=http://localhost:11434
```

---

### 2. OpenAI

**优点**：高质量向量，稳定性好

**缺点**：收费，需要 API Key

```bash
# 获取 API Key: https://platform.openai.com/api-keys
export OPENAI_API_KEY=sk-xxxxxx
```

```python
from su_memory.sdk.embedding import EmbeddingManager

manager = EmbeddingManager(
    backend="openai",
    model="text-embedding-3-small",   # 1536维，便宜 ($0.02/1M tokens)
    # 或 model="text-embedding-3-large"  # 3072维，更贵但更好
    # 或 model="text-embedding-ada-002"   # 旧版，1536维
    api_key="sk-xxxxxx"  # 可选，默认从环境变量读取
)

vector = manager.encode("你好，世界")
print(f"向量维度: {manager.dims}")
```

**环境变量**：
```bash
export OPENAI_API_KEY=sk-xxxxxx
export OPENAI_API_BASE_URL=https://api.openai.com/v1  # 可选，代理地址
```

**费用参考**（2024年）：
| 模型 | 维度 | 价格 |
|------|------|------|
| text-embedding-3-small | 1536 | $0.02/1M tokens |
| text-embedding-3-large | 3072 | $0.13/1M tokens |
| text-embedding-ada-002 | 1536 | $0.10/1M tokens |

---

### 3. MiniMax（国内可用）

**优点**：国内可用，价格适中

**缺点**：需要 Group ID

```bash
# 获取 API Key: https://platform.minimax.chat/
export MINIMAX_API_KEY=your_api_key
export MINIMAX_GROUP_ID=your_group_id
```

```python
from su_memory.sdk.embedding import EmbeddingManager

manager = EmbeddingManager(
    backend="minimax",
    model="embo-01",    # 默认
    api_key="your_api_key",
    base_url="https://api.minimax.chat/v1"
)

vector = manager.encode("你好，世界")
print(f"向量维度: {manager.dims}")
```

---

### 4. 本地模型（sentence-transformers）

**优点**：完全本地，无需网络

**缺点**：需要下载模型，首次使用较慢

```bash
pip install sentence-transformers
```

```python
from su_memory.sdk.embedding import EmbeddingManager

manager = EmbeddingManager(
    backend="local",
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # 384维，轻量
    # 或 model_name="sentence-transformers/all-mpnet-base-v2"  # 768维，质量更好
    device="cpu"  # 或 "cuda"
)

vector = manager.encode("你好，世界")
print(f"向量维度: {manager.dims}")
```

**推荐模型**：
| 模型 | 维度 | 速度 | 质量 |
|------|------|------|------|
| all-MiniLM-L6-v2 | 384 | 快 | 良好 |
| all-mpnet-base-v2 | 768 | 中 | 优秀 |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 中 | 优秀（多语言） |

---

### 5. ChromaDB

**优点**：向量数据库，可持久化存储

```bash
pip install chromadb
```

```python
from su_memory.sdk.embedding import EmbeddingManager

manager = EmbeddingManager(
    backend="chroma",
    collection_name="su_memory",
    persist_directory="./chroma_data"  # 可选，持久化目录
)

vector = manager.encode("你好，世界")
```

---

## 完整使用示例

```python
from su_memory import SuMemoryLitePro
from su_memory.sdk.embedding import EmbeddingManager

# 方式1：通过 SDK 自动管理
print("方式1: SDK 自动管理")
pro = SuMemoryLitePro(enable_vector=True)

# 方式2：自定义嵌入服务
print("\n方式2: 自定义 Ollama 服务")
manager = EmbeddingManager(
    backend="ollama",
    model="nomic-embed-text"
)
print(f"当前后端: {manager.backend_name}")
print(f"向量维度: {manager.dims}")

# 测试编码
test_texts = [
    "人工智能改变世界",
    "机器学习是AI的子领域",
    "今天天气真好"
]

print("\n编码测试:")
for text in test_texts:
    vec = manager.encode(text)
    print(f"  '{text}' -> {len(vec)}维向量")
```

---

## 性能对比

| 服务 | 延迟 | 成本 | 隐私 | 推荐场景 |
|------|------|------|------|----------|
| Ollama | ~10-50ms | 免费 | ⭐⭐⭐⭐⭐ | 开发、测试、追求隐私 |
| OpenAI | ~100-300ms | 按量计费 | ⭐⭐ | 生产环境、高质量需求 |
| MiniMax | ~100-300ms | 按量计费 | ⭐⭐ | 国内用户 |
| 本地模型 | ~50-200ms | 免费 | ⭐⭐⭐⭐⭐ | 无网络、生产环境 |
| Hash Fallback | <1ms | 免费 | ⭐⭐⭐⭐⭐ | 临时测试 |

---

## 常见问题

### Q1: 如何设置优先使用的嵌入服务？

```python
# 方式1: 代码中指定
manager = EmbeddingManager(backend="openai")

# 方式2: 环境变量
# export SU_MEMORY_EMBEDDING_PREFERRED=ollama
```

### Q2: 提示 "No embeddings returned" 错误？

```bash
# Ollama 需要先拉取模型
ollama pull nomic-embed-text

# 验证模型已安装
ollama list
```

### Q3: 如何降低 API 成本？

```python
# OpenAI: 使用更小的维度
manager = EmbeddingManager(
    backend="openai",
    model="text-embedding-3-small",
    dims=512  # 截取到512维，成本更低
)
```

### Q4: 嵌入服务不可用时会怎样？

SDK 会自动降级到 Hash Fallback 模式，虽然功能受限但不会崩溃：

```python
from su_memory.sdk.embedding import EmbeddingManager

manager = EmbeddingManager(backend="auto")
# 如果所有服务都不可用，会使用 Hash Fallback
# 向量维度较低，但基本的相似度计算仍然可用
```

---

## 进阶：直接使用 EmbeddingProvider

```python
from su_memory.sdk.embedding import (
    OllamaEmbedder,
    OpenAIEmbedder,
    MiniMaxEmbedder,
    EmbeddingFactory
)

# 直接创建
embedder = OllamaEmbedder()
result = embedder.embed_single("你好")

print(f"模型: {result.model}")
print(f"维度: {result.dimensions}")
print(f"向量: {result.embedding[:5]}...")

# 批量编码
results = embedder.embed([
    "文本1",
    "文本2",
    "文本3"
])
```

---

## 相关文档

- [主 README](../README.md)
- [API 文档](./API.md)
- [四位一体架构](./ARCHITECTURE.md)
