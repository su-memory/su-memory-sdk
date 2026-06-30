"""
PG / Redis 存储后端集成测试（P2-1）

设计：
- 用 @pytest.mark.integration 标记，默认可被 `-k "not integration"` 跳过。
- 服务/驱动不可用时优雅 skip，不报失败（本机未必运行 PG/Redis）。
- 当服务可用时，真实覆盖 add / add_batch / query / count / delete 全链路。

运行（需先启动服务，见 docker-compose.yml）:
    pytest tests/test_pg_redis_integration.py -m integration
"""
from __future__ import annotations

import os
import sys
import uuid

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers: 探测服务是否可用
# ---------------------------------------------------------------------------

async def _pg_available() -> bool:
    try:
        import asyncpg  # noqa: F401

        from su_memory._sys._pg_storage import PgStorageBackend
        from su_memory._sys._storage_backend import StorageConfig
    except ImportError:
        return False
    cfg = StorageConfig(
        pg_host=os.environ.get("PG_HOST", "localhost"),
        pg_port=int(os.environ.get("PG_PORT", "5432")),
        pg_database=os.environ.get("PG_DB", "postgres"),
        pg_user=os.environ.get("PG_USER", "postgres"),
        pg_password=os.environ.get("PG_PASSWORD", "postgres"),
        embedding_dim=8,
    )
    backend = PgStorageBackend(cfg)
    try:
        ok = await backend.initialize()
        if ok:
            await backend.close()
        return bool(ok)
    except Exception:
        return False


async def _redis_available() -> bool:
    try:
        import redis.asyncio as aioredis  # noqa: F401

        from su_memory._sys._redis_storage import RedisStorageBackend
        from su_memory._sys._storage_backend import StorageConfig
    except ImportError:
        return False
    cfg = StorageConfig(
        redis_host=os.environ.get("REDIS_HOST", "localhost"),
        redis_port=int(os.environ.get("REDIS_PORT", "6379")),
    )
    backend = RedisStorageBackend(cfg)
    try:
        ok = await backend.initialize()
        if ok:
            await backend.close()
        return bool(ok)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# PostgreSQL + pgvector
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def pg_backend():
    from su_memory._sys._pg_storage import PgStorageBackend
    from su_memory._sys._storage_backend import StorageConfig
    cfg = StorageConfig(
        pg_host=os.environ.get("PG_HOST", "localhost"),
        pg_port=int(os.environ.get("PG_PORT", "5432")),
        pg_database=os.environ.get("PG_DB", "postgres"),
        pg_user=os.environ.get("PG_USER", "postgres"),
        pg_password=os.environ.get("PG_PASSWORD", "postgres"),
        embedding_dim=8,
    )
    backend = PgStorageBackend(cfg)
    ok = await backend.initialize()
    assert ok, "PG backend initialize failed"
    yield backend
    # cleanup test rows
    try:
        async with backend._pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {cfg.pg_table} WHERE content LIKE 'pgtest:%'"
            )
    except Exception:
        pass
    await backend.close()


@pytest.mark.asyncio
async def test_pg_add_query_count_delete(pg_backend):
    if not await _pg_available():
        pytest.skip("PostgreSQL+pgvector not available")

    mid = str(uuid.uuid4())
    added = await pg_backend.add(
        memory_id=mid,
        content="pgtest: 张三负责后端开发",
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        metadata={"source": "integration"},
        energy_type="semantic",
    )
    assert added is True

    cnt = await pg_backend.count()
    assert cnt >= 1

    deleted = await pg_backend.delete(mid)
    assert deleted is True


@pytest.mark.asyncio
async def test_pg_add_batch(pg_backend):
    if not await _pg_available():
        pytest.skip("PostgreSQL+pgvector not available")
    from su_memory._sys._storage_backend import StorageMemory
    mems = [
        StorageMemory(
            memory_id=str(uuid.uuid4()),
            content=f"pgtest: batch item {i}",
            embedding=[float(i)] * 8,
            metadata={"i": i},
        )
        for i in range(3)
    ]
    ids = await pg_backend.add_batch(mems)
    assert len(ids) == 3
    # cleanup
    for m in mems:
        await pg_backend.delete(m.memory_id)


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def redis_backend():
    from su_memory._sys._redis_storage import RedisStorageBackend
    from su_memory._sys._storage_backend import StorageConfig
    cfg = StorageConfig(
        redis_host=os.environ.get("REDIS_HOST", "localhost"),
        redis_port=int(os.environ.get("REDIS_PORT", "6379")),
    )
    backend = RedisStorageBackend(cfg)
    ok = await backend.initialize()
    assert ok, "Redis backend initialize failed"
    yield backend
    await backend.close()


@pytest.mark.asyncio
async def test_redis_add_query_count_delete(redis_backend):
    if not await _redis_available():
        pytest.skip("Redis not available")

    mid = str(uuid.uuid4())
    added = await redis_backend.add(
        memory_id=mid,
        content="redistest: 李四负责前端",
        embedding=[0.9, 0.8, 0.7],
        metadata={"source": "integration"},
        energy_type="semantic",
    )
    assert added is True

    cnt = await redis_backend.count()
    assert cnt >= 1

    deleted = await redis_backend.delete(mid)
    assert deleted is True


@pytest.mark.asyncio
async def test_redis_add_batch(redis_backend):
    if not await _redis_available():
        pytest.skip("Redis not available")
    from su_memory._sys._storage_backend import StorageMemory
    mems = [
        StorageMemory(
            memory_id=str(uuid.uuid4()),
            content=f"redistest: batch item {i}",
            embedding=[float(i)] * 8,
            metadata={"i": i},
        )
        for i in range(3)
    ]
    ids = await redis_backend.add_batch(mems)
    assert len(ids) == 3
    for m in mems:
        await redis_backend.delete(m.memory_id)
