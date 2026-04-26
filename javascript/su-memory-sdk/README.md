# su-memory-sdk JavaScript 版本

> Semantic Memory Engine for JavaScript
> 一行代码让AI应用拥有记忆能力

## 安装

```bash
npm install su-memory-sdk
```

## 快速开始

```javascript
const { SuMemoryClient, createClient, VERSION } = require('su-memory-sdk');

const client = createClient({ apiUrl: 'http://localhost:8080' });
console.log(`SDK版本: ${VERSION}`);

async function main() {
  // 添加记忆
  const memoryId = await client.add("今天学习了JavaScript");
  console.log(`添加成功: ${memoryId}`);

  // 查询记忆
  const results = await client.query("JavaScript学习");
  console.log(`找到 ${results.length} 条相关记忆`);
  
  for (const result of results) {
    console.log(`[${(result.score * 100).toFixed(1)}%] ${result.content}`);
  }
}

main().catch(console.error);
```

## API 接口

| 方法 | 描述 |
|------|------|
| `add(content, metadata?)` | 添加记忆 |
| `query(query, topK?)` | 语义查询 |
| `search(query, filters?)` | 带过滤器搜索 |
| `delete(memoryId)` | 删除记忆 |
| `getStats()` | 获取统计 |
| `clear()` | 清空所有记忆 |

## License

Apache-2.0
