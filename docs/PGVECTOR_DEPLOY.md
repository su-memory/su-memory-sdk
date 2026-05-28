# pgvector 部署与调优指南 (v2.7.0)

su-memory v2.7.0 支持 PostgreSQL + pgvector 作为高性能向量存储后端，配合三层分级存储架构。

## 环境准备

### 1. 安装 PostgreSQL + pgvector

```bash
# macOS
brew install pgvector

# Ubuntu/Debian
sudo apt install postgresql-16-pgvector

# Docker (推荐)
docker run -d --name pgvector \
  -e POSTGRES_PASSWORD=sumemory \
  -e POSTGRES_DB=sumemory \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 2. 安装 Python 依赖

```bash
pip install su-memory[pgvector]
# 等价于: pip install su-memory pgvector asyncpg "sqlalchemy[asyncio]>=2.0"
```

### 3. 设置连接

```bash
export PG_DSN="postgresql+asyncpg://postgres:sumemory@localhost:5432/sumemory"
```

## PgVectorBackend 使用

```python
import asyncio
from su_memory.storage.pgvector_backend import PgVectorBackend
from su_memory.storage.base import AsyncMemoryItem

async def main():
    # 初始化后端
    backend = PgVectorBackend(
        dsn="postgresql+asyncpg://postgres:pass@localhost:5432/sumemory",
        dims=768,                # 向量维度
        pool_size=10,            # 连接池大小
        index_type="ivfflat",    # "ivfflat" 或 "hnsw"
    )
    await backend.ainit()

    # 写入记忆
    item = AsyncMemoryItem(
        id="mem_001",
        content="项目ROI增长了25%",
        embedding=[0.1, 0.2, ..., 0.768],  # 768维向量
        tier="hot",
    )
    await backend.aadd_memory(item)

    # 向量检索
    results = await backend.aquery([0.1, 0.2, ..., 0.768], top_k=10)
    for r in results:
        print(f"[{r.id}] {r.content}")

    # 层级管理
    await backend.aset_tier("mem_001", "warm")
    tier_counts = await backend.aget_tier_counts()
    print(tier_counts)  # {"hot": 1000, "warm": 5000}

    await backend.aclose()

asyncio.run(main())
```

## 分层存储 (TieredStorage)

```python
from su_memory.storage.tiered import TieredStorage, TierConfig

config = TierConfig(
    hot_capacity=10_000,        # hot tier 最大 1万条
    warm_capacity=50_000,       # warm tier 最大 5万条
    hot_backend=pgvector_backend,
    warm_backend=sqlite_backend,
    access_threshold=10,        # 访问 10 次晋升到 hot
    idle_days_demote=30,        # 30 天未访问降到 cold
    auto_rebalance=True,        # 自动再平衡
)

storage = TieredStorage(config)
await storage.ainit()

# 查询自动跨层路由
results = await storage.aquery(embedding, top_k=10)

# 手动触发再平衡
stats = await storage.arebalance()
# {"promoted": 150, "demoted": 200}
```

## 索引策略选择

| 场景 | 推荐索引 | 说明 |
|------|----------|------|
| 写入密集型 | IVFFlat | 写入快，索引构建轻量 |
| 查询密集型 | HNSW | 查询延迟低（P50 <10ms） |
| 混合负载 | IVFFlat + 定期重建 | 平衡读写 |

```python
# HNSW 模式
backend = PgVectorBackend(
    dsn=PG_DSN,
    index_type="hnsw",
    # HNSW 参数通过 PostgreSQL 设置
    # SET hnsw.m = 16;
    # SET hnsw.ef_construction = 200;
)
```

## 数据迁移

```bash
# SQLite → pgvector
su-memory migrate --from sqlite --to pgvector \
  --db su_memory.db \
  --pg-dsn "postgresql+asyncpg://postgres:pass@localhost:5432/sumemory" \
  --batch-size 500

# 查看分层统计
su-memory tier-stats --pg-dsn "postgresql+asyncpg://..."
```

## 性能调优

### 连接池配置

```python
# 写入密集：大池
PgVectorBackend(dsn=PG_DSN, pool_size=20)

# 查询密集：小池 + 大量 overflow
PgVectorBackend(dsn=PG_DSN, pool_size=5, max_overflow=20)
```

### PostgreSQL 参数

```sql
-- postgresql.conf
shared_buffers = 2GB
work_mem = 64MB
maintenance_work_mem = 512MB
effective_cache_size = 6GB

-- 向量操作优化
max_parallel_workers_per_gather = 4
```

### 批量写入

```python
# 推荐批量大小 250-500
await backend.aadd_batch(items)  # items: List[AsyncMemoryItem]
```

### 向量维度性能

| 维度 | 索引大小/千条 | 查询 P50 | 写入吞吐 |
|------|:---:|:---:|:---:|
| 384d | ~4 MB | 3ms | 1200 ops/s |
| 768d | ~8 MB | 5ms | 800 ops/s |
| 1536d | ~16 MB | 10ms | 400 ops/s |

## 生产检查清单

- [ ] PostgreSQL 16+ / pgvector 0.7+
- [ ] `shared_buffers` ≥ 总数据量的 25%
- [ ] 连接池 `pool_size` 匹配 worker 数量
- [ ] 索引类型匹配负载特征
- [ ] 定期 `VACUUM ANALYZE memories`
- [ ] 监控 `pg_stat_user_tables` 查询统计
- [ ] 冷数据归档策略已配置

## 故障排查

```sql
-- 检查 pgvector 扩展
SELECT * FROM pg_extension WHERE extname = 'vector';

-- 检查索引状态
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'memories';

-- 查看表大小
SELECT pg_size_pretty(pg_total_relation_size('memories'));

-- 检查慢查询
SELECT query, mean_exec_time
FROM pg_stat_statements
WHERE query LIKE '%memories%'
ORDER BY mean_exec_time DESC
LIMIT 10;
```
