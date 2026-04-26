# su-memory-sdk

> Semantic Memory Engine for TypeScript/JavaScript
> 一行代码让AI应用拥有记忆能力

[![npm][npm-badge]][npm-url]
[![Node][node-badge]][node-url]
[![License][license-badge]][license-url]

## 特性

- 🚀 **简单易用** - 一行代码初始化，立即拥有记忆能力
- 🔗 **LangChain兼容** - 支持 Retriever 和 Tool 接口
- 🌐 **跨语言一致** - TypeScript/JavaScript API与Python SDK保持一致
- 📦 **轻量级** - 体积 < 500KB，无额外依赖
- 🔒 **本地优先** - 数据存储在本地，保护隐私

## 安装

```bash
# npm
npm install su-memory-sdk

# yarn
yarn add su-memory-sdk

# pnpm
pnpm add su-memory-sdk
```

## 快速开始

### TypeScript

```typescript
import { SuMemoryClient, createClient } from 'su-memory-sdk';

const client = createClient({ apiUrl: 'http://localhost:8080' });

async function main() {
  // 添加记忆
  const memoryId = await client.add("今天学习了TypeScript", { source: 'study' });
  
  // 查询记忆
  const results = await client.query("TypeScript学习");
  console.log(results[0].content);
  
  // 获取统计
  const stats = await client.getStats();
  console.log(`共有 ${stats.count} 条记忆`);
}

main();
```

### JavaScript

```javascript
const { SuMemoryClient, createClient } = require('su-memory-sdk');

const client = createClient();

async function main() {
  const memoryId = await client.add("今天学习了JavaScript");
  const results = await client.query("JavaScript学习");
  console.log(results[0].content);
}

main();
```

## API 接口

| 方法 | 描述 | 参数 |
|------|------|------|
| `add(content, metadata?)` | 添加记忆 | content: string, metadata?: object |
| `query(query, topK?)` | 语义查询 | query: string, topK?: number |
| `search(query, filters?)` | 带过滤器搜索 | query: string, filters?: SearchFilters |
| `delete(memoryId)` | 删除记忆 | memoryId: string |
| `getStats()` | 获取统计 | - |
| `clear()` | 清空所有记忆 | - |

## LangChain 集成

### 作为 Retriever 使用

```typescript
import { SuMemoryClient } from 'su-memory-sdk';
import { ConversationalRetrievalChain } from 'langchain/chains';

const client = new SuMemoryClient();
const retriever = client.asRetriever(5);

const chain = ConversationalRetrievalChain.fromLLM(llm, retriever);
const response = await chain.invoke({ chat_history: [], question: query });
```

### 作为 Tool 使用

```typescript
import { SuMemoryClient } from 'su-memory-sdk';

const client = new SuMemoryClient();
const tool = client.asTool('memory_search', '搜索记忆中的相关信息');

const agent = new Agent({ tools: [tool] });
```

## API 服务

需要配合 Python API 服务使用。启动服务:

```bash
cd /path/to/su-memory-sdk
python python_api_server.py --port 8080
```

## 类型定义

TypeScript 提供完整的类型支持:

```typescript
interface QueryResult {
  id: string;
  content: string;
  score: number;
  metadata?: Record<string, unknown>;
}

interface MemoryStats {
  count: number;
  categoryDistribution?: Record<string, number>;
  energyDistribution?: Record<string, number>;
}
```

## 错误处理

```typescript
import { SuMemoryClient, ConnectionError, MemoryNotFoundError } from 'su-memory-sdk';

const client = new SuMemoryClient();

try {
  await client.add("test");
} catch (error) {
  if (error instanceof ConnectionError) {
    console.log('无法连接到API服务');
  } else if (error instanceof MemoryNotFoundError) {
    console.log('记忆不存在');
  }
}
```

## License

Apache-2.0

[npm-badge]: https://img.shields.io/npm/v/su-memory-sdk
[npm-url]: https://www.npmjs.com/package/su-memory-sdk
[node-badge]: https://img.shields.io/node/v/su-memory-sdk
[node-url]: https://nodejs.org/
[license-badge]: https://img.shields.io/badge/License-Apache--2.0-blue
[license-url]: LICENSE
