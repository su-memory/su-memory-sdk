/**
 * su-memory-sdk JavaScript 入口
 * @version 1.7.0
 * 
 * CommonJS 兼容版本
 * 
 * @example
 * ```javascript
 * const { SuMemoryClient, createClient, VERSION } = require('su-memory-sdk');
 * 
 * const client = createClient({ apiUrl: 'http://localhost:8080' });
 * console.log(`SDK版本: ${VERSION}`);
 * 
 * async function main() {
 *   // 添加记忆
 *   const memoryId = await client.add("今天学习了JavaScript");
 *   console.log(`添加成功: ${memoryId}`);
 * 
 *   // 查询记忆
 *   const results = await client.query("JavaScript学习");
 *   console.log(`找到 ${results.length} 条相关记忆`);
 *   
 *   for (const result of results) {
 *     console.log(`[${(result.score * 100).toFixed(1)}%] ${result.content}`);
 *   }
 * }
 * 
 * main().catch(console.error);
 * ```
 */

'use strict';

const client = require('./client.js');

module.exports = {
  // 核心客户端
  SuMemoryClient: client.SuMemoryClient,
  createClient: client.createClient,
  
  // 便捷方法
  addAndQuery: client.addAndQuery,
  addBatchAndQuery: client.addBatchAndQuery,
  
  // LangChain 兼容
  SuMemoryRetriever: client.SuMemoryRetriever,
  SuMemoryTool: client.SuMemoryTool,
  
  // 错误类
  SDKError: client.SDKError,
  ConnectionError: client.ConnectionError,
  MemoryNotFoundError: client.MemoryNotFoundError,
  ValidationError: client.ValidationError,
  TimeoutError: client.TimeoutError,
  RateLimitError: client.RateLimitError,
  ServerError: client.ServerError,
  
  // 版本
  VERSION: client.VERSION,
};
