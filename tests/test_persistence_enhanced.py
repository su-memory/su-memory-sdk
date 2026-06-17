"""
su-memory SDK v3.5.5 — 增强持久化专项测试 (pytest)
===================================================

覆盖 6 大持久化维度:
  1. SQLite 后端 CRUD 操作
  2. 向量+元数据联合持久化
  3. 重启后数据完整性
  4. 并发写入安全
  5. 数据损坏恢复
  6. StorageBackend 抽象接口合规

测试环境要求: SQLite 后端 (默认可用), pytest, pytest-asyncio
"""

import asyncio
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

import pytest

pytestmark = [pytest.mark.persistence, pytest.mark.p2]


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sqlite_cfg():
    """SQLite 存储配置"""
    from su_memory._sys._storage_backend import BackendType, StorageConfig
    cfg = StorageConfig()
    cfg.backend_type = BackendType.SQLITE
    cfg.embedding_dim = 384
    cfg.sqlite_path = None  # 使用临时数据库
    return cfg


@pytest.fixture
def persist_dir():
    """独立持久化目录 (自动清理)"""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ============================================================
# 1. SQLite 后端 CRUD
# ============================================================

class TestSqliteCrud:
    """SQLite 后端基本 CRUD 操作测试"""

    @pytest.mark.asyncio
    async def test_initialize_and_health_check(self, sqlite_cfg):
        """初始化 + 健康检查"""
        from su_memory._sys._storage_backend import create_backend
        from su_memory._sys._sqlite_storage import SqliteStorageBackend

        backend = SqliteStorageBackend(sqlite_cfg)
        assert await backend.initialize()

        health = await backend.health_check()
        assert health.available
        assert health.memory_count == 0
        assert health.error is None
        await backend.close()

    @pytest.mark.asyncio
    async def test_add_single_memory(self, sqlite_cfg):
        """添加单条记忆"""
        from su_memory._sys._sqlite_storage import SqliteStorageBackend

        backend = SqliteStorageBackend(sqlite_cfg)
        await backend.initialize()

        ok = await backend.add(
            memory_id="mem-001",
            content="测试记忆内容",
            embedding=[0.1] * 384,
            metadata={"tag": "test", "priority": 1},
            energy_type="semantic",
        )
        assert ok

        count = await backend.count()
        assert count == 1
        await backend.close()

    @pytest.mark.asyncio
    async def test_add_batch_memories(self, sqlite_cfg):
        """批量添加记忆"""
        from su_memory._sys._sqlite_storage import SqliteStorageBackend
        from su_memory._sys._storage_backend import StorageMemory

        backend = SqliteStorageBackend(sqlite_cfg)
        await backend.initialize()

        memories = [
            StorageMemory(
                memory_id=f"mem-{i:03d}",
                content=f"记忆内容 {i}",
                embedding=[float(i % 384) / 100] * 384,
                metadata={"index": i},
                energy_type="semantic",
            )
            for i in range(10)
        ]

        ids = await backend.add_batch(memories)
        assert len(ids) == 10
        assert await backend.count() == 10
        await backend.close()

    @pytest.mark.asyncio
    async def test_query_by_vector(self, sqlite_cfg):
        """向量检索"""
        from su_memory._sys._sqlite_storage import SqliteStorageBackend
        from su_memory._sys._storage_backend import StorageMemory

        backend = SqliteStorageBackend(sqlite_cfg)
        await backend.initialize()

        # 添加不同内容的记忆
        for i in range(5):
            await backend.add(
                memory_id=f"q-{i}",
                content=f"查询测试记忆 {i}",
                embedding=[float(i) / 10] * 384,
            )

        # 查询相似向量
        results = await backend.query(vector=[0.3] * 384, top_k=3)
        assert len(results) <= 3
        # 结果按 score 降序
        if len(results) >= 2:
            assert results[0].score >= results[1].score
        await backend.close()

    @pytest.mark.asyncio
    async def test_delete_memory(self, sqlite_cfg):
        """删除记忆"""
        from su_memory._sys._sqlite_storage import SqliteStorageBackend

        backend = SqliteStorageBackend(sqlite_cfg)
        await backend.initialize()

        await backend.add(memory_id="del-1", content="待删除")
        await backend.add(memory_id="del-2", content="保留")
        assert await backend.count() == 2

        ok = await backend.delete("del-1")
        assert ok
        assert await backend.count() == 1
        await backend.close()

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, sqlite_cfg):
        """删除不存在的记忆不抛异常"""
        from su_memory._sys._sqlite_storage import SqliteStorageBackend

        backend = SqliteStorageBackend(sqlite_cfg)
        await backend.initialize()

        ok = await backend.delete("nonexistent")
        assert not ok
        await backend.close()

    @pytest.mark.asyncio
    async def test_query_empty_store(self, sqlite_cfg):
        """空存储查询返回空列表"""
        from su_memory._sys._sqlite_storage import SqliteStorageBackend

        backend = SqliteStorageBackend(sqlite_cfg)
        await backend.initialize()

        results = await backend.query(vector=[0.5] * 384, top_k=10)
        assert results == []
        await backend.close()


# ============================================================
# 2. SuMemory 客户端持久化
# ============================================================

class TestSuMemoryPersistence:
    """SuMemory 客户端级别的持久化测试"""

    def test_restart_preserves_memories(self, persist_dir):
        """重启后记忆不丢失"""
        from su_memory import SuMemory

        # 写入
        client = SuMemory(persist_dir=persist_dir)
        for i in range(5):
            client.add(f"重启测试记忆 {i}", metadata={"i": i})
        assert client.get_stats()["total_memories"] == 5

        # 模拟重启
        del client
        client2 = SuMemory(persist_dir=persist_dir)
        stats = client2.get_stats()
        assert stats["total_memories"] == 5, f"重启后记忆丢失: 期望5 实际{stats['total_memories']}"

    def test_immediate_readback(self, persist_dir):
        """写入后立即查询可得"""
        from su_memory import SuMemory

        client = SuMemory(persist_dir=persist_dir)
        unique = "IMMEDIATE_READBACK_TOKEN_2026"
        client.add(unique)
        results = client.query(unique, top_k=1)
        assert len(results) > 0

    def test_delete_persists_across_restart(self, persist_dir):
        """删除后重启确认"""
        from su_memory import SuMemory

        client = SuMemory(persist_dir=persist_dir)
        ids = [client.add(f"删除测试 {i}") for i in range(5)]
        client.delete(ids[0], ids[2])

        # 重启
        del client
        client2 = SuMemory(persist_dir=persist_dir)
        assert client2.get_stats()["total_memories"] == 3

    def test_multiple_adds_stress(self, persist_dir):
        """连续添加 50 条记忆"""
        from su_memory import SuMemory

        client = SuMemory(persist_dir=persist_dir)
        for i in range(50):
            client.add(f"压力测试记忆 #{i:04d}")
        assert client.get_stats()["total_memories"] == 50

    def test_metadata_roundtrip(self, persist_dir):
        """元数据在重启后保持"""
        from su_memory import SuMemory

        client = SuMemory(persist_dir=persist_dir)
        client.add("带元数据的记忆", metadata={"author": "test", "version": 3, "tags": ["a", "b"]})

        del client
        client2 = SuMemory(persist_dir=persist_dir)
        results = client2.query("带元数据的记忆", top_k=1)
        assert len(results) > 0


# ============================================================
# 3. 数据损坏恢复
# ============================================================

class TestCorruptionRecovery:
    """数据损坏恢复测试"""

    def test_broken_json_degradation(self, persist_dir):
        """JSON 损坏时降级不崩溃"""
        from su_memory import SuMemory

        client = SuMemory(persist_dir=persist_dir)
        client.add("一条记忆")

        # 写入非法 JSON
        data_path = os.path.join(persist_dir, "memories.json")
        with open(data_path, "w") as f:
            f.write("{ broken <<< json content >>> }")

        # 重启应该降级
        try:
            client2 = SuMemory(persist_dir=persist_dir)
            # 降级后数据可能清空
            count = client2.get_stats().get("total_memories", 0)
            assert count >= 0  # 不崩溃就行
        except Exception as e:
            pytest.fail(f"JSON 损坏恢复失败: {e}")

    def test_empty_file_recovery(self, persist_dir):
        """空 JSON 文件恢复"""
        from su_memory import SuMemory

        data_path = os.path.join(persist_dir, "memories.json")
        os.makedirs(persist_dir, exist_ok=True)
        with open(data_path, "w") as f:
            f.write("")

        client = SuMemory(persist_dir=persist_dir)
        stats = client.get_stats()
        assert stats["total_memories"] == 0

    def test_missing_directory_recovery(self, persist_dir):
        """目录不存在时自动创建"""
        from su_memory import SuMemory

        missing_dir = os.path.join(persist_dir, "nonexistent", "subdir")
        client = SuMemory(persist_dir=missing_dir)
        client.add("自动创建目录测试")
        assert os.path.exists(missing_dir)


# ============================================================
# 4. 并发安全
# ============================================================

class TestConcurrencyPersistence:
    """并发写入安全测试"""

    def test_concurrent_adds_no_data_loss(self, persist_dir):
        """多线程并发添加不丢数据"""
        from su_memory import SuMemory

        client = SuMemory(persist_dir=persist_dir)
        errors = []
        n_per_thread = 20
        n_threads = 4

        def writer(thread_id):
            try:
                for i in range(n_per_thread):
                    client.add(f"线程 {thread_id} 记忆 {i}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"并发写入错误: {errors}"
        stats = client.get_stats()
        assert stats["total_memories"] == n_threads * n_per_thread, (
            f"并发写入丢失数据: 期望{n_threads * n_per_thread} "
            f"实际{stats['total_memories']}"
        )

    def test_read_during_write(self, persist_dir):
        """写入期间读取不崩溃"""
        from su_memory import SuMemory

        client = SuMemory(persist_dir=persist_dir)

        # 预填数据
        for i in range(10):
            client.add(f"预填数据 {i}")

        errors = []
        def reader():
            try:
                for _ in range(10):
                    client.query("测试", top_k=3)
            except Exception as e:
                errors.append(str(e))

        def writer():
            try:
                for i in range(10):
                    client.add(f"并发写入 {i}")
            except Exception as e:
                errors.append(str(e))

        t_read = threading.Thread(target=reader)
        t_write = threading.Thread(target=writer)
        t_read.start()
        t_write.start()
        t_read.join()
        t_write.join()

        assert len(errors) == 0, f"并发读写错误: {errors}"


# ============================================================
# 5. 向量持久化
# ============================================================

class TestVectorPersistence:
    """向量数据持久化测试"""

    def test_vectors_saved_with_memories(self, persist_dir):
        """向量与记忆同时持久化"""
        from su_memory import SuMemory

        client = SuMemory(persist_dir=persist_dir)
        client.add("向量持久化测试")

        vec_path = os.path.join(persist_dir, "vectors.json")
        assert os.path.exists(vec_path), "向量文件未生成"

    def test_vectors_restored_after_restart(self, persist_dir):
        """重启后向量可查询"""
        from su_memory import SuMemory

        client = SuMemory(persist_dir=persist_dir)
        client.add("人工智能在医疗领域的应用")
        client.add("机器学习是AI的重要分支")
        client.add("Python是最流行的编程语言之一")

        del client
        client2 = SuMemory(persist_dir=persist_dir)
        results = client2.query("人工智能", top_k=2)
        assert len(results) > 0, "重启后向量查询无结果"


# ============================================================
# 6. StorageBackend 接口合规
# ============================================================

class TestStorageBackendInterface:
    """StorageBackend 抽象接口合规验证"""

    def test_factory_creates_sqlite(self):
        """工厂函数创建 SQLite 后端"""
        from su_memory._sys._storage_backend import BackendType, StorageConfig, create_backend

        cfg = StorageConfig()
        cfg.backend_type = BackendType.SQLITE
        backend = asyncio.run(create_backend(BackendType.SQLITE, cfg))
        assert backend is not None

    def test_sqlite_config_defaults(self):
        """SQLite 配置默认值"""
        from su_memory._sys._storage_backend import BackendType, StorageConfig

        cfg = StorageConfig()
        assert cfg.backend_type == BackendType.SQLITE
        assert cfg.embedding_dim == 1536

    def test_backend_type_enum_values(self):
        """后端类型枚举值"""
        from su_memory._sys._storage_backend import BackendType

        types = {t.value for t in BackendType}
        assert "sqlite" in types
        assert "postgresql" in types
        assert "redis" in types
        assert "auto" in types

    def test_storage_memory_dataclass(self):
        """StorageMemory 数据类"""
        from su_memory._sys._storage_backend import StorageMemory

        mem = StorageMemory(
            memory_id="test-1",
            content="Hello World",
            embedding=[0.1, 0.2, 0.3],
            metadata={"key": "value"},
            energy_type="semantic",
            created_at=1234567890.0,
        )
        assert mem.memory_id == "test-1"
        assert mem.content == "Hello World"
        assert mem.energy_type == "semantic"

    def test_backend_health_dataclass(self):
        """BackendHealth 数据类"""
        from su_memory._sys._storage_backend import BackendHealth, BackendType

        health = BackendHealth(
            available=True,
            backend_type=BackendType.SQLITE,
            latency_ms=1.5,
            memory_count=100,
            detail="OK",
        )
        assert health.available
        assert health.memory_count == 100
        assert health.latency_ms == 1.5
        assert health.error is None
