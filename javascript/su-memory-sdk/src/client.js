/**
 * su-memory-sdk JavaScript 客户端
 * @version 1.7.0
 * 
 * CommonJS 兼容版本
 * @example
 * ```javascript
 * const { SuMemoryClient } = require('su-memory-sdk');
 * const client = new SuMemoryClient({ apiUrl: 'http://localhost:8080' });
 * 
 * async function main() {
 *   const memoryId = await client.add("今天学习了JavaScript");
 *   const results = await client.query("JavaScript学习");
 *   console.log(results[0].content);
 * }
 * 
 * main();
 * ```
 */

'use strict';

// 导入类型说明 (JSDoc)
// @typedef {Object} QueryResult
// @property {string} id - 记忆ID
// @property {string} content - 记忆内容
// @property {number} score - 相似度分数
// @property {Object} [metadata] - 元数据

// @typedef {Object} SuMemoryClientConfig
// @property {string} [apiUrl='http://localhost:8080'] - API服务地址
// @property {string} [apiKey] - API密钥
// @property {number} [topK=10] - 默认top_k
// @property {number} [timeout=30000] - 请求超时(ms)

// @typedef {Object} MemoryStats
// @property {number} count - 总记忆数
// @property {Object} [categoryDistribution] - 按类别分布
// @property {Object} [energyDistribution] - 按能量类型分布

/**
 * SuMemoryClient - su-memory SDK JavaScript 客户端
 */
class SuMemoryClient {
  /**
   * @param {SuMemoryClientConfig} config
   */
  constructor(config = {}) {
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
    this.version = '1.7.0';
  }

  /**
   * 发送HTTP请求
   * @private
   */
  async _request(method, endpoint, body) {
    const url = `${this.apiUrl}${endpoint}`;
    const headers = {
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
            errorBody.memoryId || 'unknown'
          );
        }
        if (response.status === 429) {
          throw new RateLimitError(
            'Rate limit exceeded',
            errorBody.retryAfter
          );
        }
        if (response.status === 401) {
          throw new SDKError('Authentication failed', 'AUTHENTICATION_ERROR');
        }
        throw new ServerError(
          errorBody.message || `HTTP ${response.status}`,
          response.status
        );
      }

      return response.json();
    } catch (error) {
      if (error instanceof SDKError) {
        throw error;
      }
      if (error.name === 'AbortError') {
        throw new TimeoutError(`Request timeout after ${this.timeout}ms`);
      }
      if (error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED') {
        throw new ConnectionError(`Failed to connect to ${this.apiUrl}`, error);
      }
      throw new SDKError(error.message || 'Unknown error occurred');
    }
  }

  /**
   * 添加记忆
   * @param {string} content - 记忆内容
   * @param {Object} [metadata] - 可选元数据
   * @returns {Promise<string>} 新增记忆ID
   */
  async add(content, metadata) {
    if (!content || typeof content !== 'string') {
      throw new ValidationError('content must be a non-empty string', 'content');
    }

    const response = await this._request(
      'POST',
      '/api/memories/add',
      { content, metadata }
    );

    return response.memoryId;
  }

  /**
   * 查询记忆
   * @param {string} query - 查询文本
   * @param {number} [topK] - 返回数量
   * @returns {Promise<QueryResult[]>} 按相关度排序的记忆列表
   */
  async query(query, topK) {
    if (!query || typeof query !== 'string') {
      throw new ValidationError('query must be a non-empty string', 'query');
    }

    const k = topK ?? this.topK;
    const response = await this._request(
      'POST',
      '/api/memories/query',
      { query, top_k: k }
    );

    return response.results;
  }

  /**
   * 搜索记忆 (支持过滤器)
   * @param {string} query - 搜索文本
   * @param {Object} [filters] - 过滤器配置
   * @returns {Promise<QueryResult[]>} 符合条件的记忆列表
   */
  async search(query, filters) {
    if (!query || typeof query !== 'string') {
      throw new ValidationError('query must be a non-empty string', 'query');
    }

    const response = await this._request(
      'POST',
      '/api/memories/search',
      { query, filters }
    );

    return response.results;
  }

  /**
   * 删除记忆
   * @param {string} memoryId - 要删除的记忆ID
   * @returns {Promise<boolean>} 是否成功
   */
  async delete(memoryId) {
    if (!memoryId || typeof memoryId !== 'string') {
      throw new ValidationError('memoryId must be a non-empty string', 'memoryId');
    }

    try {
      const response = await this._request(
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
   * @returns {Promise<MemoryStats>} 统计信息
   */
  async getStats() {
    const response = await this._request(
      'GET',
      '/api/memories/stats'
    );
    return response.stats;
  }

  /**
   * 清空所有记忆
   * @returns {Promise<void>}
   */
  async clear() {
    await this._request('DELETE', '/api/memories/clear');
  }

  /**
   * 获取单条记忆
   * @param {string} memoryId - 记忆ID
   * @returns {Promise<Object|null>} 记忆详情
   */
  async get(memoryId) {
    if (!memoryId || typeof memoryId !== 'string') {
      throw new ValidationError('memoryId must be a non-empty string', 'memoryId');
    }

    try {
      const response = await this._request(
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
   * @param {Array<{content: string, metadata?: Object}>} items - 记忆列表
   * @returns {Promise<string[]>} 新增记忆ID列表
   */
  async addBatch(items) {
    if (!Array.isArray(items) || items.length === 0) {
      throw new ValidationError('items must be a non-empty array', 'items');
    }

    const response = await this._request(
      'POST',
      '/api/memories/add_batch',
      { items }
    );

    return response.memoryIds;
  }

  /**
   * 批量删除记忆
   * @param {string[]} memoryIds - 记忆ID列表
   * @returns {Promise<number>} 删除成功的数量
   */
  async deleteBatch(memoryIds) {
    if (!Array.isArray(memoryIds) || memoryIds.length === 0) {
      throw new ValidationError('memoryIds must be a non-empty array', 'memoryIds');
    }

    const response = await this._request(
      'POST',
      '/api/memories/delete_batch',
      { memoryIds }
    );

    return response.deletedCount;
  }

  /**
   * 创建 LangChain Retriever
   * @param {number} [topK] - 返回数量
   * @returns {SuMemoryRetriever} LangChain 兼容检索器
   */
  asRetriever(topK) {
    return new SuMemoryRetriever(this, topK ?? this.topK);
  }

  /**
   * 创建 LangChain Tool
   * @param {string} name - 工具名称
   * @param {string} description - 工具描述
   * @returns {SuMemoryTool} LangChain 兼容工具
   */
  asTool(name, description) {
    return new SuMemoryTool(name, description, this);
  }

  /**
   * 检查API服务健康状态
   * @returns {Promise<boolean>}
   */
  async healthCheck() {
    try {
      const response = await this._request('GET', '/api/health');
      return response.status === 'ok';
    } catch {
      return false;
    }
  }

  /**
   * 获取SDK版本信息
   * @returns {Object} 版本信息
   */
  getVersion() {
    const [major, minor, patch] = this.version.split('.').map(Number);
    return { version: this.version, major, minor, patch };
  }
}

/**
 * LangChain 兼容检索器
 */
class SuMemoryRetriever {
  /**
   * @param {SuMemoryClient} client
   * @param {number} [topK=5]
   */
  constructor(client, topK = 5) {
    this.client = client;
    this.topK = topK;
  }

  /**
   * 获取相关文档
   * @param {string} query - 查询文本
   * @returns {Promise<Array<{pageContent: string, metadata: Object}>>}
   */
  async getRelevantDocuments(query) {
    const results = await this.client.search(query, { topK: this.topK });
    return results.map((result) => ({
      pageContent: result.content,
      metadata: {
        id: result.id,
        score: result.score,
        ...result.metadata,
      },
    }));
  }
}

/**
 * LangChain Tool 兼容工具
 */
class SuMemoryTool {
  /**
   * @param {string} name - 工具名称
   * @param {string} description - 工具描述
   * @param {SuMemoryClient} client
   * @param {number} [topK=3]
   */
  constructor(name, description, client, topK = 3) {
    this.name = name;
    this.description = description;
    this.client = client;
    this.topK = topK;
  }

  /**
   * 调用工具
   * @param {string} input - 输入文本
   * @returns {Promise<string>}
   */
  async func(input) {
    try {
      const results = await this.client.search(input, { topK: this.topK });
      if (results.length === 0) {
        return '未找到相关记忆';
      }
      return results
        .map((r, i) => `[${i + 1}] ${r.content} (相似度: ${(r.score * 100).toFixed(1)}%)`)
        .join('\n');
    } catch (error) {
      return `搜索失败: ${error.message}`;
    }
  }
}

/**
 * SDK基础错误类
 */
class SDKError extends Error {
  constructor(message, code = 'SDK_ERROR') {
    super(message);
    this.name = 'SDKError';
    this.code = code;
  }
}

/**
 * 连接错误
 */
class ConnectionError extends SDKError {
  constructor(message, originalError) {
    super(message, 'CONNECTION_ERROR');
    this.name = 'ConnectionError';
  }
}

/**
 * 记忆不存在错误
 */
class MemoryNotFoundError extends SDKError {
  constructor(memoryId) {
    super(`Memory not found: ${memoryId}`, 'MEMORY_NOT_FOUND');
    this.name = 'MemoryNotFoundError';
    this.memoryId = memoryId;
  }
}

/**
 * 参数验证错误
 */
class ValidationError extends SDKError {
  constructor(message, field) {
    super(message, 'VALIDATION_ERROR');
    this.name = 'ValidationError';
    this.field = field;
  }
}

/**
 * 超时错误
 */
class TimeoutError extends SDKError {
  constructor(message) {
    super(message || 'Request timeout', 'TIMEOUT_ERROR');
    this.name = 'TimeoutError';
  }
}

/**
 * 速率限制错误
 */
class RateLimitError extends SDKError {
  constructor(message, retryAfter) {
    super(message, 'RATE_LIMIT_ERROR');
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

/**
 * 服务端错误
 */
class ServerError extends SDKError {
  constructor(message, statusCode) {
    super(message, 'SERVER_ERROR');
    this.name = 'ServerError';
    this.statusCode = statusCode;
  }
}

/**
 * 工厂函数: 创建客户端
 * @param {SuMemoryClientConfig} config
 * @returns {SuMemoryClient}
 */
function createClient(config) {
  return new SuMemoryClient(config);
}

/**
 * 便捷方法: 快速添加并查询
 * @param {SuMemoryClient} client
 * @param {string} content
 * @param {string} query
 * @param {Object} [metadata]
 * @returns {Promise<QueryResult[]>}
 */
async function addAndQuery(client, content, query, metadata) {
  await client.add(content, metadata);
  return client.query(query);
}

/**
 * 便捷方法: 批量添加并查询
 * @param {SuMemoryClient} client
 * @param {Array<{content: string, metadata?: Object}>} items
 * @param {string} query
 * @returns {Promise<QueryResult[]>}
 */
async function addBatchAndQuery(client, items, query) {
  await client.addBatch(items);
  return client.query(query);
}

module.exports = {
  SuMemoryClient,
  SuMemoryRetriever,
  SuMemoryTool,
  createClient,
  addAndQuery,
  addBatchAndQuery,
  SDKError,
  ConnectionError,
  MemoryNotFoundError,
  ValidationError,
  TimeoutError,
  RateLimitError,
  ServerError,
  VERSION: '1.7.0',
};
