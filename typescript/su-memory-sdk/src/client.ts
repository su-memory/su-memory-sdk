/**
 * su-memory-sdk SuMemoryClient
 * @version 1.7.0
 */

import type {
  MemoryItem,
  QueryResult,
  SuMemoryClientConfig,
  MemoryStats,
  SearchFilters,
  AddMemoryResponse,
  QueryMemoriesResponse,
  DeleteMemoryResponse,
  GetStatsResponse,
} from './types.js';
import {
  SDKError,
  ConnectionError,
  MemoryNotFoundError,
  ValidationError,
  TimeoutError,
  RateLimitError,
  ServerError,
} from './errors.js';
import { SuMemoryRetriever, SuMemoryTool, createRetriever, createTool } from './retriever.js';

/**
 * SuMemoryClient - su-memory SDK TypeScript 客户端
 * 
 * 通过HTTP调用Python后端API服务
 * 
 * Example:
 * ```typescript
 * import { SuMemoryClient } from 'su-memory-sdk';
 * 
 * const client = new SuMemoryClient({ apiUrl: 'http://localhost:8080' });
 * 
 * // 添加记忆
 * const memoryId = await client.add("今天学习了TypeScript", { source: 'study' });
 * 
 * // 查询记忆
 * const results = await client.query("TypeScript学习");
 * console.log(results[0].content);
 * 
 * // 获取统计
 * const stats = await client.getStats();
 * console.log(`共有 ${stats.count} 条记忆`);
 * ```
 */
export class SuMemoryClient {
  /** API服务地址 */
  private apiUrl: string;
  /** API密钥 */
  private apiKey?: string;
  /** 默认top_k */
  private topK: number;
  /** 请求超时(ms) */
  private timeout: number;
  /** SDK版本 */
  public readonly version: string = '1.7.0';

  constructor(config: SuMemoryClientConfig = {}) {
    const {
      apiUrl = 'http://localhost:8080',
      apiKey,
      topK = 10,
      timeout = 30000,
    } = config;

    this.apiUrl = apiUrl.replace(/\/$/, ''); // 移除末尾斜杠
    this.apiKey = apiKey;
    this.topK = topK;
    this.timeout = timeout;
  }

  /**
   * 发送HTTP请求
   */
  private async request<T>(
    method: string,
    endpoint: string,
    body?: unknown
  ): Promise<T> {
    const url = `${this.apiUrl}${endpoint}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        
        if (response.status === 404) {
          throw new MemoryNotFoundError(
            (errorBody as { memoryId?: string }).memoryId || 'unknown'
          );
        }
        if (response.status === 429) {
          throw new RateLimitError(
            'Rate limit exceeded',
            (errorBody as { retryAfter?: number }).retryAfter
          );
        }
        if (response.status === 401) {
          throw new SDKError('Authentication failed', 'AUTHENTICATION_ERROR');
        }
        throw new ServerError(
          (errorBody as { message?: string }).message || `HTTP ${response.status}`,
          response.status
        );
      }

      return response.json() as T;
    } catch (error) {
      if (error instanceof SDKError) {
        throw error;
      }
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new TimeoutError(`Request timeout after ${this.timeout}ms`);
        }
        throw new ConnectionError(`Failed to connect to ${this.apiUrl}`, error);
      }
      throw new SDKError('Unknown error occurred');
    }
  }

  /**
   * 添加记忆
   * 
   * @param content - 记忆内容
   * @param metadata - 可选元数据
   * @returns 新增记忆ID
   * 
   * Example:
   * ```typescript
   * const memoryId = await client.add("项目完成", { priority: 'high' });
   * ```
   */
  async add(
    content: string,
    metadata?: Record<string, unknown>
  ): Promise<string> {
    if (!content || typeof content !== 'string') {
      throw new ValidationError('content must be a non-empty string', 'content', content);
    }

    const response = await this.request<AddMemoryResponse>(
      'POST',
      '/api/memories/add',
      { content, metadata }
    );

    return response.memoryId;
  }

  /**
   * 查询记忆
   * 
   * @param query - 查询文本
   * @param topK - 返回数量 (默认配置值)
   * @returns 按相关度排序的记忆列表
   * 
   * Example:
   * ```typescript
   * const results = await client.query("投资回报", 5);
   * for (const result of results) {
   *   console.log(`${result.score}: ${result.content}`);
   * }
   * ```
   */
  async query(query: string, topK?: number): Promise<QueryResult[]> {
    if (!query || typeof query !== 'string') {
      throw new ValidationError('query must be a non-empty string', 'query', query);
    }

    const k = topK ?? this.topK;
    const response = await this.request<QueryMemoriesResponse>(
      'POST',
      '/api/memories/query',
      { query, top_k: k }
    );

    return response.results;
  }

  /**
   * 搜索记忆 (支持过滤器)
   * 
   * @param query - 搜索文本
   * @param filters - 过滤器配置
   * @returns 符合条件的记忆列表
   * 
   * Example:
   * ```typescript
   * const results = await client.search("项目", {
   *   category: 'work',
   *   timeRange: { start: Date.now() - 86400000, end: Date.now() }
   * });
   * ```
   */
  async search(
    query: string,
    filters?: SearchFilters & { topK?: number }
  ): Promise<QueryResult[]> {
    if (!query || typeof query !== 'string') {
      throw new ValidationError('query must be a non-empty string', 'query', query);
    }

    const response = await this.request<QueryMemoriesResponse>(
      'POST',
      '/api/memories/search',
      { query, filters }
    );

    return response.results;
  }

  /**
   * 删除记忆
   * 
   * @param memoryId - 要删除的记忆ID
   * @returns 是否成功
   * 
   * Example:
   * ```typescript
   * const success = await client.delete("mem_123");
   * ```
   */
  async delete(memoryId: string): Promise<boolean> {
    if (!memoryId || typeof memoryId !== 'string') {
      throw new ValidationError('memoryId must be a non-empty string', 'memoryId', memoryId);
    }

    try {
      const response = await this.request<DeleteMemoryResponse>(
        'DELETE',
        `/api/memories/${encodeURIComponent(memoryId)}`
      );
      return response.success;
    } catch (error) {
      if (error instanceof MemoryNotFoundError) {
        return false;
      }
      throw error;
    }
  }

  /**
   * 获取记忆统计
   * 
   * @returns 统计信息
   * 
   * Example:
   * ```typescript
   * const stats = await client.getStats();
   * console.log(`总记忆数: ${stats.count}`);
   * ```
   */
  async getStats(): Promise<MemoryStats> {
    const response = await this.request<GetStatsResponse>(
      'GET',
      '/api/memories/stats'
    );
    return response.stats;
  }

  /**
   * 清空所有记忆
   * 
   * @returns void
   * 
   * Example:
   * ```typescript
   * await client.clear();
   * ```
   */
  async clear(): Promise<void> {
    await this.request<void>('DELETE', '/api/memories/clear');
  }

  /**
   * 获取单条记忆
   * 
   * @param memoryId - 记忆ID
   * @returns 记忆详情
   */
  async get(memoryId: string): Promise<MemoryItem | null> {
    if (!memoryId || typeof memoryId !== 'string') {
      throw new ValidationError('memoryId must be a non-empty string', 'memoryId', memoryId);
    }

    try {
      const response = await this.request<{ memory: MemoryItem }>(
        'GET',
        `/api/memories/${encodeURIComponent(memoryId)}`
      );
      return response.memory;
    } catch (error) {
      if (error instanceof MemoryNotFoundError) {
        return null;
      }
      throw error;
    }
  }

  /**
   * 批量添加记忆
   * 
   * @param items - 记忆列表 [{content, metadata?}]
   * @returns 新增记忆ID列表
   */
  async addBatch(
    items: Array<{ content: string; metadata?: Record<string, unknown> }>
  ): Promise<string[]> {
    if (!Array.isArray(items) || items.length === 0) {
      throw new ValidationError('items must be a non-empty array', 'items', items);
    }

    const response = await this.request<{ memoryIds: string[] }>(
      'POST',
      '/api/memories/add_batch',
      { items }
    );

    return response.memoryIds;
  }

  /**
   * 批量删除记忆
   * 
   * @param memoryIds - 记忆ID列表
   * @returns 删除成功的数量
   */
  async deleteBatch(memoryIds: string[]): Promise<number> {
    if (!Array.isArray(memoryIds) || memoryIds.length === 0) {
      throw new ValidationError('memoryIds must be a non-empty array', 'memoryIds', memoryIds);
    }

    const response = await this.request<{ deletedCount: number }>(
      'POST',
      '/api/memories/delete_batch',
      { memoryIds }
    );

    return response.deletedCount;
  }

  /**
   * 创建 LangChain Retriever
   * 
   * @param topK - 返回数量
   * @returns LangChain 兼容检索器
   * 
   * Example:
   * ```typescript
   * import { ConversationalRetrievalChain } from 'langchain/chains';
   * 
   * const retriever = client.asRetriever(5);
   * const chain = ConversationalRetrievalChain.fromLLM(llm, retriever);
   * ```
   */
  asRetriever(topK?: number): SuMemoryRetriever {
    return createRetriever(this, topK ?? this.topK);
  }

  /**
   * 创建 LangChain Tool
   * 
   * @param name - 工具名称
   * @param description - 工具描述
   * @returns LangChain 兼容工具
   * 
   * Example:
   * ```typescript
   * const tool = client.asTool('memory_search', '搜索相关记忆');
   * const agent = new Agent({ tools: [tool] });
   * ```
   */
  asTool(name: string, description: string): SuMemoryTool {
    return createTool(this, name, description);
  }

  /**
   * 检查API服务健康状态
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.request<{ status: string }>(
        'GET',
        '/api/health'
      );
      return response.status === 'ok';
    } catch {
      return false;
    }
  }

  /**
   * 获取SDK版本信息
   */
  getVersion(): { version: string; major: number; minor: number; patch: number } {
    const parts = this.version.split('.').map(Number);
    const major = parts[0] ?? 1;
    const minor = parts[1] ?? 7;
    const patch = parts[2] ?? 0;
    return { version: this.version, major, minor, patch };
  }
}

/**
 * 默认导出工厂函数
 */
export function createClient(config?: SuMemoryClientConfig): SuMemoryClient {
  return new SuMemoryClient(config);
}

/**
 * 便捷方法: 快速添加并查询
 */
export async function addAndQuery(
  client: SuMemoryClient,
  content: string,
  query: string,
  metadata?: Record<string, unknown>
): Promise<QueryResult[]> {
  await client.add(content, metadata);
  return client.query(query);
}

/**
 * 便捷方法: 批量添加并查询
 */
export async function addBatchAndQuery(
  client: SuMemoryClient,
  items: Array<{ content: string; metadata?: Record<string, unknown> }>,
  query: string
): Promise<QueryResult[]> {
  await client.addBatch(items);
  return client.query(query);
}
