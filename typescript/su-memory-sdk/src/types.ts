/**
 * su-memory-sdk TypeScript 类型定义
 * @version 1.7.0
 */

/**
 * 记忆条目
 */
export interface MemoryItem {
  /** 唯一标识符 */
  id: string;
  /** 记忆内容 */
  content: string;
  /** 可选元数据 */
  metadata?: Record<string, unknown>;
  /** 向量嵌入 (可选) */
  embedding?: number[];
  /** 创建时间戳 */
  timestamp?: number;
}

/**
 * 查询结果
 */
export interface QueryResult {
  /** 记忆ID */
  id: string;
  /** 记忆内容 */
  content: string;
  /** 相似度分数 */
  score: number;
  /** 元数据 */
  metadata?: Record<string, unknown>;
}

/**
 * SuMemoryClient 配置
 */
export interface SuMemoryClientConfig {
  /** API服务地址 */
  apiUrl?: string;
  /** API密钥 (可选) */
  apiKey?: string;
  /** 存储路径 */
  storagePath?: string;
  /** 向量维度 */
  embeddingDim?: number;
  /** 默认top_k */
  topK?: number;
  /** 请求超时(ms) */
  timeout?: number;
}

/**
 * 记忆统计
 */
export interface MemoryStats {
  /** 总记忆数 */
  count: number;
  /** 按类别分布 */
  categoryDistribution?: Record<string, number>;
  /** 按能量类型分布 */
  energyDistribution?: Record<string, number>;
  /** 存储大小(bytes) */
  storageSize?: number;
}

/**
 * 搜索过滤器
 */
export interface SearchFilters {
  /** 类别过滤 */
  category?: string;
  /** 能量类型过滤 */
  energyType?: string;
  /** 时间范围 */
  timeRange?: {
    start: number;
    end: number;
  };
  /** 自定义过滤器 */
  custom?: Record<string, unknown>;
}

/**
 * API错误响应
 */
export interface APIError {
  /** 错误代码 */
  code: string;
  /** 错误消息 */
  message: string;
  /** 详细错误 */
  details?: unknown;
}

/**
 * 添加记忆请求
 */
export interface AddMemoryRequest {
  /** 记忆内容 */
  content: string;
  /** 元数据 */
  metadata?: Record<string, unknown>;
  /** 编码类型 */
  encoding?: 'auto' | 'semantic' | 'bagua';
}

/**
 * 添加记忆响应
 */
export interface AddMemoryResponse {
  /** 新增记忆ID */
  memoryId: string;
}

/**
 * 查询记忆请求
 */
export interface QueryMemoriesRequest {
  /** 查询文本 */
  query: string;
  /** 返回数量 */
  topK?: number;
  /** 查询模式 */
  mode?: 'semantic' | 'causal' | 'hybrid';
}

/**
 * 查询记忆响应
 */
export interface QueryMemoriesResponse {
  /** 检索结果 */
  results: QueryResult[];
}

/**
 * 搜索记忆请求
 */
export interface SearchMemoriesRequest {
  /** 搜索文本 */
  query: string;
  /** 过滤器 */
  filters?: SearchFilters;
}

/**
 * 删除记忆响应
 */
export interface DeleteMemoryResponse {
  /** 是否成功 */
  success: boolean;
  /** 删除的记忆ID */
  memoryId: string;
}

/**
 * 获取统计响应
 */
export interface GetStatsResponse {
  /** 统计信息 */
  stats: MemoryStats;
}

/**
 * LangChain Retriever接口兼容
 */
export interface RetrieverConfig {
  /** 返回数量 */
  topK?: number;
  /** 过滤器 */
  filters?: SearchFilters;
}

/**
 * LangChain Tool接口兼容
 */
export interface ToolConfig {
  /** 工具名称 */
  name: string;
  /** 工具描述 */
  description: string;
}

/**
 * SDK版本信息
 */
export interface SDKVersion {
  /** 主版本号 */
  major: number;
  /** 次版本号 */
  minor: number;
  /** 修订版本号 */
  patch: number;
  /** 版本字符串 */
  version: string;
}
