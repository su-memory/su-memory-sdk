/**
 * su-memory-sdk 错误类
 * @version 1.7.0
 */

/**
 * SDK基础错误类
 */
export class SDKError extends Error {
  /** 错误代码 */
  public readonly code: string;
  /** 原始错误 */
  public readonly originalError?: Error;

  constructor(message: string, code: string = 'SDK_ERROR', originalError?: Error) {
    super(message);
    this.name = 'SDKError';
    this.code = code;
    this.originalError = originalError;
  }
}

/**
 * 连接错误 - API服务不可达
 */
export class ConnectionError extends SDKError {
  constructor(message: string, originalError?: Error) {
    super(message, 'CONNECTION_ERROR', originalError);
    this.name = 'ConnectionError';
  }
}

/**
 * 认证错误 - API密钥无效
 */
export class AuthenticationError extends SDKError {
  constructor(message: string, originalError?: Error) {
    super(message, 'AUTHENTICATION_ERROR', originalError);
    this.name = 'AuthenticationError';
  }
}

/**
 * 记忆不存在错误
 */
export class MemoryNotFoundError extends SDKError {
  public readonly memoryId: string;

  constructor(memoryId: string, message?: string) {
    const msg = message || `Memory not found: ${memoryId}`;
    super(msg, 'MEMORY_NOT_FOUND');
    this.name = 'MemoryNotFoundError';
    this.memoryId = memoryId;
  }
}

/**
 * 参数验证错误
 */
export class ValidationError extends SDKError {
  public readonly field?: string;
  public readonly value?: unknown;

  constructor(message: string, field?: string, value?: unknown) {
    super(message, 'VALIDATION_ERROR');
    this.name = 'ValidationError';
    this.field = field;
    this.value = value;
  }
}

/**
 * 存储错误 - 持久化失败
 */
export class StorageError extends SDKError {
  constructor(message: string, originalError?: Error) {
    super(message, 'STORAGE_ERROR', originalError);
    this.name = 'StorageError';
  }
}

/**
 * 速率限制错误
 */
export class RateLimitError extends SDKError {
  public readonly retryAfter?: number;

  constructor(message: string, retryAfter?: number) {
    super(message, 'RATE_LIMIT_ERROR');
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

/**
 * 超时错误
 */
export class TimeoutError extends SDKError {
  constructor(message: string = 'Request timeout') {
    super(message, 'TIMEOUT_ERROR');
    this.name = 'TimeoutError';
  }
}

/**
 * 服务端错误
 */
export class ServerError extends SDKError {
  public readonly statusCode?: number;

  constructor(message: string, statusCode?: number) {
    super(message, 'SERVER_ERROR');
    this.name = 'ServerError';
    this.statusCode = statusCode;
  }
}

/**
 * 工具函数: 解析错误响应
 */
export function parseAPIError(response: {
  code?: string;
  message?: string;
  status?: number;
}): SDKError {
  const { code = 'UNKNOWN_ERROR', message = 'Unknown error', status } = response;

  switch (code) {
    case 'MEMORY_NOT_FOUND':
      return new MemoryNotFoundError(code);
    case 'VALIDATION_ERROR':
      return new ValidationError(message);
    case 'AUTHENTICATION_ERROR':
      return new AuthenticationError(message);
    case 'RATE_LIMIT_ERROR':
      return new RateLimitError(message);
    default:
      return new ServerError(message, status);
  }
}
