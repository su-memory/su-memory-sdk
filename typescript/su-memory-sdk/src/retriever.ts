/**
 * su-memory-sdk Retriever - LangChain 兼容检索器
 * @version 1.7.0
 */

import type { QueryResult, SearchFilters } from './types.js';

/**
 * 查询客户端接口
 */
export interface QueryClient {
  search(query: string, filters?: SearchFilters & { topK?: number }): Promise<QueryResult[]>;
}

/**
 * LangChain BaseRetriever 兼容接口
 */
export interface BaseRetrieverInterface {
  /**
   * 根据查询字符串获取相关文档
   */
  getRelevantDocuments(query: string): Promise<RetrievedDocument[]>;
}

/**
 * 检索到的文档 (LangChain 格式)
 */
export interface RetrievedDocument {
  /** 页面内容 */
  pageContent: string;
  /** 元数据 */
  metadata: Record<string, unknown>;
}

/**
 * LangChain Tool 接口
 */
export interface Tool {
  /** 工具名称 */
  name: string;
  /** 工具描述 */
  description: string;
  /** 调用函数 */
  func: (input: string) => Promise<string>;
}

/**
 * SuMemoryRetriever - LangChain 兼容检索器
 * 
 * Example:
 * ```typescript
 * import { SuMemoryRetriever } from 'su-memory-sdk';
 * 
 * const client = new SuMemoryClient();
 * const retriever = client.asRetriever(5);
 * const docs = await retriever.getRelevantDocuments("项目投资回报");
 * ```
 */
export class SuMemoryRetriever implements BaseRetrieverInterface {
  private client: QueryClient;
  private topK: number;
  private filters?: SearchFilters;

  constructor(client: QueryClient, topK: number = 5, filters?: SearchFilters) {
    this.client = client;
    this.topK = topK;
    this.filters = filters;
  }

  /**
   * 获取相关文档
   */
  async getRelevantDocuments(query: string): Promise<RetrievedDocument[]> {
    const results = await this.client.search(query, { ...this.filters, topK: this.topK });
    return results.map((result: QueryResult) => ({
      pageContent: result.content,
      metadata: {
        id: result.id,
        score: result.score,
        ...result.metadata,
      },
    }));
  }

  /**
   * 获取原始查询结果
   */
  async getResults(query: string): Promise<QueryResult[]> {
    return this.client.search(query, this.filters);
  }
}

/**
 * SuMemoryTool - LangChain Tool 兼容工具
 * 
 * Example:
 * ```typescript
 * import { SuMemoryTool } from 'su-memory-sdk';
 * 
 * const client = new SuMemoryClient();
 * const tool = client.asTool('memory_search', '搜索记忆中的相关信息');
 * const result = await tool.func("项目进度");
 * ```
 */
export class SuMemoryTool implements Tool {
  public name: string;
  public description: string;
  private client: QueryClient;
  private topK: number;

  constructor(name: string, description: string, client: QueryClient, topK: number = 3) {
    this.name = name;
    this.description = description;
    this.client = client;
    this.topK = topK;
  }

  /**
   * 调用工具
   */
  async func(input: string): Promise<string> {
    try {
      const results = await this.client.search(input, { topK: this.topK });
      if (results.length === 0) {
        return '未找到相关记忆';
      }
      return results
        .map((r: QueryResult, i: number) => `[${i + 1}] ${r.content} (相似度: ${(r.score * 100).toFixed(1)}%)`)
        .join('\n');
    } catch (error) {
      return `搜索失败: ${error instanceof Error ? error.message : 'Unknown error'}`;
    }
  }

  /**
   * 作为 LangChain Tool 使用
   */
  asTool(): Tool {
    return {
      name: this.name,
      description: this.description,
      func: this.func.bind(this),
    };
  }
}

/**
 * 工厂函数: 创建检索器
 */
export function createRetriever(
  client: QueryClient,
  topK?: number,
  filters?: SearchFilters
): SuMemoryRetriever {
  return new SuMemoryRetriever(client, topK, filters);
}

/**
 * 工厂函数: 创建工具
 */
export function createTool(
  client: QueryClient,
  name: string,
  description: string,
  topK?: number
): SuMemoryTool {
  return new SuMemoryTool(name, description, client, topK);
}
