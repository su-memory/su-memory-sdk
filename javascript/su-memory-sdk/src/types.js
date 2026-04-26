/**
 * su-memory-sdk 类型说明 (JSDoc)
 * @version 1.7.0
 */

/**
 * @typedef {Object} QueryResult
 * @property {string} id - 记忆ID
 * @property {string} content - 记忆内容
 * @property {number} score - 相似度分数 (0-1)
 * @property {Object} [metadata] - 可选元数据
 */

/**
 * @typedef {Object} SuMemoryClientConfig
 * @property {string} [apiUrl='http://localhost:8080'] - API服务地址
 * @property {string} [apiKey] - API密钥 (可选)
 * @property {number} [topK=10] - 默认返回数量
 * @property {number} [timeout=30000] - 请求超时(ms)
 */

/**
 * @typedef {Object} MemoryStats
 * @property {number} count - 总记忆数
 * @property {Object} [categoryDistribution] - 按类别分布 {creative: 5, ...}
 * @property {Object} [energyDistribution] - 按能量类型分布 {earth: 3, ...}
 */

/**
 * @typedef {Object} SearchFilters
 * @property {string} [category] - 类别过滤
 * @property {string} [energyType] - 能量类型过滤
 * @property {{start: number, end: number}} [timeRange] - 时间范围
 * @property {Object} [custom] - 自定义过滤器
 */

/**
 * @typedef {Object} MemoryItem
 * @property {string} id - 记忆ID
 * @property {string} content - 记忆内容
 * @property {Object} [metadata] - 元数据
 * @property {number[]} [embedding] - 向量嵌入
 * @property {number} [timestamp] - 创建时间戳
 */

/**
 * SDK错误类型
 * @see {@link SDKError}
 * @see {@link ConnectionError}
 * @see {@link MemoryNotFoundError}
 * @see {@link ValidationError}
 */

/**
 * LangChain兼容接口
 * @see {@link SuMemoryRetriever}
 * @see {@link SuMemoryTool}
 */

module.exports = {};
