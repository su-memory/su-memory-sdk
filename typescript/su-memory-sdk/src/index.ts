/**
 * su-memory-sdk - Semantic Memory Engine for TypeScript
 * @version 1.7.0
 * 
 * 一行代码让AI应用拥有记忆能力
 * 
 * @example
 * ```typescript
 * import { SuMemoryClient, createClient } from 'su-memory-sdk';
 * 
 * // 方式1: 使用构造函数
 * const client = new SuMemoryClient({ apiUrl: 'http://localhost:8080' });
 * 
 * // 方式2: 使用工厂函数
 * const client = createClient();
 * 
 * // 添加记忆
 * await client.add("今天学习了TypeScript", { source: 'study' });
 * 
 * // 查询记忆
 * const results = await client.query("TypeScript学习");
 * console.log(results[0].content);
 * ```
 */

// 导出核心客户端
export { SuMemoryClient, createClient, addAndQuery, addBatchAndQuery } from './client.js';

// 导出类型
export type {
  MemoryItem,
  QueryResult,
  SuMemoryClientConfig,
  MemoryStats,
  SearchFilters,
  APIError,
  AddMemoryRequest,
  AddMemoryResponse,
  QueryMemoriesRequest,
  QueryMemoriesResponse,
  SearchMemoriesRequest,
  DeleteMemoryResponse,
  GetStatsResponse,
  RetrieverConfig,
  ToolConfig,
  SDKVersion,
} from './types.js';

// 导出错误类
export {
  SDKError,
  ConnectionError,
  AuthenticationError,
  MemoryNotFoundError,
  ValidationError,
  StorageError,
  RateLimitError,
  TimeoutError,
  ServerError,
  parseAPIError,
} from './errors.js';

// 导出检索器
export {
  SuMemoryRetriever,
  SuMemoryTool,
  createRetriever,
  createTool,
} from './retriever.js';
export type {
  BaseRetrieverInterface,
  RetrievedDocument,
  Tool,
} from './retriever.js';

// SDK版本常量
export const VERSION = '1.7.0';
