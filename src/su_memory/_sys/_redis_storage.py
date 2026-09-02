"""
RedisStorageBackend — Redis 存储后端

使用 redis[hiredis] 异步客户端实现超低延迟向量检索。
支持 RediSearch 向量索引或回退 JSON + 手动余弦相似度。

依赖: pip install redis[hiredis]
  - redis>=5.0.0

v3.0.0: 分布式存储架构的超低延迟后端。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from su_memory._sys._storage_backend import (
    BackendHealth,
    BackendType,
    StorageBackend,
    StorageConfig,
    StorageMemory,
)

logger = logging.getLogger(__name__)


class RedisStorageBackend(StorageBackend):
    """
    Redis 存储后端。

    使用 redis[hiredis] 异步客户端，支持:
    - RediSearch 向量索引 (FT.SEARCH) — 优先
    - JSON + 手动余弦相似度 — 回退
    - TTL 自动过期记忆
    - HSET 元数据存储

    数据结构:
        Key: memory:{memory_id}
        Type: Hash
        Fields:
            - memory_id: str
            - content: str
            - embedding: JSON 序列化的向量
            - metadata: JSON 序列化的元数据
            - energy_type: str
            - created_at: float (epoch)

    特性:
    - 超低延迟 (<5ms P99)
    - 可选的 TTL 自动过期
    - RediSearch 向量索引 (可选)
    - JSON.set 现代 API (redis-py ≥5.0)
    """

    # RediSearch 索引创建命令
    _REDISEARCH_SCHEMA = """
        FT.CREATE {index_name}
        ON HASH PREFIX 1 memory:
        SCHEMA
            content TEXT
            embedding VECTOR HNSW 6
                DIM {dim}
                DISTANCE_METRIC COSINE
                TYPE FLOAT32
            energy_type TAG
            created_at NUMERIC SORTABLE
    """

    def __init__(self, config: StorageConfig | None = None):
        super().__init__(config)
        self._client = None
        self._redsearch_available = False
        self._index_name = "idx_su_memory"

    @property
    def backend_type(self) -> BackendType:
        return BackendType.REDIS

    async def initialize(self) -> bool:
        """初始化 Redis 连接和索引"""
        try:
            import redis.asyncio as aioredis

            cfg = self.config

            self._client = aioredis.Redis(
                host=cfg.redis_host,
                port=cfg.redis_port,
                db=cfg.redis_db,
                password=cfg.redis_password or None,
                decode_responses=False,  # 向量数据需保持 bytes
            )

            # 测试连接
            await self._client.ping()

            # 尝试创建 RediSearch 索引
            try:
                await self._try_create_redsearch_index()
                self._redsearch_available = True
                logger.info("RedisStorageBackend: RediSearch vector index enabled")
            except Exception:
                self._redsearch_available = False
                logger.info("RedisStorageBackend: RediSearch not available, using manual cosine similarity")

            self._initialized = True
            logger.info(
                "RedisStorageBackend initialized: %s:%s/%s",
                cfg.redis_host, cfg.redis_port, cfg.redis_db,
            )
            return True
        except ImportError:
            logger.warning("redis not installed. Install with: pip install redis[hiredis]")
            self._initialized = False
            return False
        except Exception as e:
            logger.error("RedisStorageBackend initialization failed: %s", e)
            self._initialized = False
            return False

    async def _try_create_redsearch_index(self) -> None:
        """尝试创建 RediSearch 向量索引"""
        try:
            # 先检查模块是否加载
            modules = await self._client.execute_command("MODULE", "LIST")
            modules_str = str(modules) if modules else ""
            if "search" not in modules_str.lower():
                self._redsearch_available = False
                return

            # 尝试创建索引（忽略已存在错误）
            cfg = self.config
            create_cmd = self._REDISEARCH_SCHEMA.format(
                index_name=self._index_name,
                dim=cfg.embedding_dim,
            )
            await self._client.execute_command(*create_cmd.split())
        except Exception:
            logger.exception("RedisStorageBackend: RediSearch index creation failed")
            self._redsearch_available = False

    async def add(
        self,
        memory_id: str,
        content: str,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        energy_type: str | None = None,
        created_at: float | None = None,
    ) -> bool:
        """添加单条记忆"""
        if not self._initialized or not self._client:
            return False

        try:
            key = f"memory:{memory_id}"
            ts = created_at or time.time()

            data = {
                "memory_id": memory_id.encode() if isinstance(memory_id, str) else memory_id,
                "content": content.encode() if isinstance(content, str) else content,
                "metadata": json.dumps(metadata or {}),
                "energy_type": (energy_type or ""),
                "created_at": str(ts),
            }

            if embedding:
                # 存储为 32 位浮点字节
                import struct
                emb_bytes = struct.pack(f"{len(embedding)}f", *embedding)
                data["embedding"] = emb_bytes

            await self._client.hset(key, mapping=data)

            # 可选 TTL
            if self.config.redis_ttl:
                await self._client.expire(key, self.config.redis_ttl)

            return True
        except Exception:
            logger.exception("RedisStorageBackend.add failed for memory_id=%s", memory_id)
            return False

    async def add_batch(self, memories: list[StorageMemory]) -> list[str]:
        """批量添加记忆（pipeline 优化）"""
        if not self._initialized or not self._client:
            return []

        try:
            import struct

            async with self._client.pipeline() as pipe:
                for mem in memories:
                    key = f"memory:{mem.memory_id}"
                    ts = mem.created_at or time.time()

                    data = {
                        "memory_id": mem.memory_id,
                        "content": mem.content,
                        "metadata": json.dumps(mem.metadata or {}),
                        "energy_type": mem.energy_type or "",
                        "created_at": str(ts),
                    }

                    if mem.embedding:
                        emb_bytes = struct.pack(f"{len(mem.embedding)}f", *mem.embedding)
                        data["embedding"] = emb_bytes

                    pipe.hset(key, mapping=data)

                    if self.config.redis_ttl:
                        pipe.expire(key, self.config.redis_ttl)

                await pipe.execute()

            return [m.memory_id for m in memories]
        except Exception as e:
            logger.exception("RedisStorageBackend.add_batch failed: %s", e)
            return []

    async def query(
        self,
        vector: list[float] | None,
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[StorageMemory]:
        """向量相似度检索"""
        if not self._initialized or not self._client:
            return []

        try:
            if vector and self._redsearch_available:
                return await self._query_redsearch(vector, top_k, filter_expr)
            elif vector:
                return await self._query_manual(vector, top_k, filter_expr)
            else:
                return await self._query_all(top_k, filter_expr)
        except Exception as e:
            logger.error("RedisStorageBackend.query failed: %s", e)
            return []

    async def _query_redsearch(
        self, vector: list[float], top_k: int, filter_expr: str | None
    ) -> list[StorageMemory]:
        """使用 RediSearch 向量检索"""
        import struct

        emb_bytes = struct.pack(f"{len(vector)}f", *vector)
        # FT.SEARCH 使用 BLOB 格式的向量查询
        query_parts = ["*"]
        if filter_expr:
            query_parts.append(f"@energy_type:{{{self._parse_filter_value(filter_expr)}}}")

        # RediSearch 向量查询
        query_str = " ".join(query_parts)
        try:
            results = await self._client.execute_command(
                "FT.SEARCH", self._index_name, query_str,
                "SORTBY", "created_at", "DESC",
                "LIMIT", "0", str(top_k),
                "PARAMS", "2", "vec", emb_bytes,
                "SORTBY", "__embedding_score",
                "RETURN", "4",
                "memory_id", "content", "metadata", "energy_type",
                "DIALECT", "2",
            )
            return self._parse_redsearch_results(results)
        except Exception:
            logger.exception("RedisStorageBackend: RediSearch query failed, falling back to manual")
            # RediSearch 失败，回退手动检索
            return await self._query_manual(vector, top_k, filter_expr)

    async def _query_manual(
        self, vector: list[float], top_k: int, filter_expr: str | None
    ) -> list[StorageMemory]:
        """手动余弦相似度检索"""
        import struct

        # 扫描所有 memory:* 键
        all_keys = []
        cursor = 0
        while True:
            cursor, keys = await self._client.scan(cursor, match="memory:*", count=100)
            all_keys.extend(keys)
            if cursor == 0:
                break

        if not all_keys:
            return []

        # 批量获取数据
        results = []
        filter_val = self._parse_filter_value(filter_expr) if filter_expr else None

        for key in all_keys:
            data = await self._client.hgetall(key)

            # 解码
            decoded = {}
            for k, v in data.items():
                key_str = k.decode() if isinstance(k, bytes) else k
                decoded[key_str] = v

            # 过滤
            if filter_val:
                energy = decoded.get("energy_type", b"").decode() if isinstance(decoded.get("energy_type"), bytes) else decoded.get("energy_type", "")
                if energy != filter_val:
                    continue

            # 计算余弦相似度
            score = 0.0
            if "embedding" in decoded:
                emb_bytes = decoded["embedding"]
                if isinstance(emb_bytes, bytes):
                    emb = list(struct.unpack(f"{len(emb_bytes)//4}f", emb_bytes))
                    score = self._cosine_similarity(vector, emb)

            memory_id = decoded.get("memory_id", b"").decode() if isinstance(decoded.get("memory_id"), bytes) else decoded.get("memory_id", "")
            content = decoded.get("content", b"").decode() if isinstance(decoded.get("content"), bytes) else decoded.get("content", "")
            metadata_str = decoded.get("metadata", b"{}")
            metadata = json.loads(metadata_str.decode() if isinstance(metadata_str, bytes) else metadata_str)
            energy_type = decoded.get("energy_type", b"").decode() if isinstance(decoded.get("energy_type"), bytes) else decoded.get("energy_type", "")
            created_at_str = decoded.get("created_at", b"0")
            created_at = float(created_at_str.decode() if isinstance(created_at_str, bytes) else created_at_str)

            results.append(StorageMemory(
                memory_id=memory_id,
                content=content,
                metadata=metadata,
                energy_type=energy_type,
                created_at=created_at,
                score=round(score, 4),
            ))

        # 排序取 top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def _query_all(
        self, top_k: int, filter_expr: str | None
    ) -> list[StorageMemory]:
        """无向量时的全量查询"""
        results = []

        cursor = 0
        filter_val = self._parse_filter_value(filter_expr) if filter_expr else None
        collected = 0

        while collected < top_k:
            cursor, keys = await self._client.scan(cursor, match="memory:*", count=100)
            for key in keys:
                if collected >= top_k:
                    break

                data = await self._client.hgetall(key)
                decoded = {}
                for k, v in data.items():
                    key_str = k.decode() if isinstance(k, bytes) else k
                    decoded[key_str] = v

                if filter_val:
                    energy = decoded.get("energy_type", b"").decode() if isinstance(decoded.get("energy_type"), bytes) else decoded.get("energy_type", "")
                    if energy != filter_val:
                        continue

                memory_id = decoded.get("memory_id", b"").decode() if isinstance(decoded.get("memory_id"), bytes) else decoded.get("memory_id", "")
                content = decoded.get("content", b"").decode() if isinstance(decoded.get("content"), bytes) else decoded.get("content", "")
                metadata_str = decoded.get("metadata", b"{}")
                metadata = json.loads(metadata_str.decode() if isinstance(metadata_str, bytes) else metadata_str)
                energy_type = decoded.get("energy_type", b"").decode() if isinstance(decoded.get("energy_type"), bytes) else decoded.get("energy_type", "")
                created_at_str = decoded.get("created_at", b"0")
                created_at = float(created_at_str.decode() if isinstance(created_at_str, bytes) else created_at_str)

                results.append(StorageMemory(
                    memory_id=memory_id,
                    content=content,
                    metadata=metadata,
                    energy_type=energy_type,
                    created_at=created_at,
                    score=0.0,
                ))
                collected += 1

            if cursor == 0:
                break

        return results

    def _parse_redsearch_results(self, raw_results) -> list[StorageMemory]:
        """解析 RediSearch 结果"""
        results = []
        # RediSearch 返回: [total, key1, [field1, val1, ...], key2, ...]
        if not raw_results or len(raw_results) < 2:
            return results

        for i in range(2, len(raw_results), 2):
            if i + 1 >= len(raw_results):
                break
            fields = raw_results[i + 1]
            data = {}
            for j in range(0, len(fields), 2):
                k = fields[j].decode() if isinstance(fields[j], bytes) else fields[j]
                v = fields[j + 1].decode() if isinstance(fields[j + 1], bytes) else fields[j + 1]
                data[k] = v

            results.append(StorageMemory(
                memory_id=data.get("memory_id", ""),
                content=data.get("content", ""),
                metadata=json.loads(data.get("metadata", "{}")),
                energy_type=data.get("energy_type"),
                score=float(data.get("__embedding_score", 0)),
            ))

        return results

    def _parse_filter_value(self, filter_expr: str | None) -> str | None:
        """从过滤表达式提取值"""
        if not filter_expr:
            return None
        # energy_type == 'causal' → causal
        parts = filter_expr.split("==")
        if len(parts) == 2:
            return parts[1].strip().strip("'\"")
        return None

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = (sum(x * x for x in a)) ** 0.5
        norm_b = (sum(x * x for x in b)) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def delete(self, memory_id: str) -> bool:
        """删除指定记忆"""
        if not self._initialized or not self._client:
            return False
        try:
            key = f"memory:{memory_id}"
            result = await self._client.delete(key)
            return result > 0
        except Exception:
            logger.exception("RedisStorageBackend.delete failed for memory_id=%s", memory_id)
            return False

    async def count(self) -> int:
        """获取记忆总数"""
        if not self._initialized or not self._client:
            return 0
        try:
            # 计数 memory:* 键
            count = 0
            cursor = 0
            while True:
                cursor, keys = await self._client.scan(cursor, match="memory:*", count=100)
                count += len(keys)
                if cursor == 0:
                    break
            return count
        except Exception:
            logger.exception("RedisStorageBackend.count failed")
            return 0

    async def health_check(self) -> BackendHealth:
        """Redis 健康检查"""
        t0 = time.time()

        try:
            if not self._client:
                # 快速检测 Redis 是否可达
                try:
                    import redis.asyncio as aioredis
                    cfg = self.config
                    client = aioredis.Redis(
                        host=cfg.redis_host,
                        port=cfg.redis_port,
                        db=cfg.redis_db,
                        password=cfg.redis_password or None,
                        socket_connect_timeout=3,
                    )
                    await client.ping()
                    await client.aclose()
                    return BackendHealth(
                        available=True,
                        backend_type=BackendType.REDIS,
                        detail="Redis reachable (client not initialized)",
                    )
                except Exception as e:
                    logger.debug("Redis quick-connect failed: %s", e)
                    return BackendHealth(
                        available=False,
                        backend_type=BackendType.REDIS,
                        detail="Redis not reachable",
                    )

            await self._client.ping()
            cnt = await self.count()
            latency = (time.time() - t0) * 1000

            return BackendHealth(
                available=True,
                backend_type=BackendType.REDIS,
                latency_ms=round(latency, 2),
                memory_count=cnt,
                detail=(
                    f"Redis {self.config.redis_host}:{self.config.redis_port}"
                    f"{' (RediSearch)' if self._redsearch_available else ''}"
                ),
            )
        except Exception as e:
            logger.exception("RedisStorageBackend.health_check failed")
            return BackendHealth(
                available=False,
                backend_type=BackendType.REDIS,
                error=str(e),
            )

    async def close(self) -> None:
        """关闭 Redis 连接"""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._initialized = False
