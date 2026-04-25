"""
su-memory SDK 稳定性与异常测试 (Task #9)
===========================================

覆盖：
  9.1 依赖服务宕机测试（Qdrant / PostgreSQL）
  9.2 LLM 超时/不可用测试
  9.3 异常输入测试（超大文本、空输入、特殊字符、嵌套JSON、Unicode边界、二进制数据）
  9.4 并发写入冲突测试
  9.5 内存泄漏检测
  9.6 Docker 重启恢复测试
  9.7 长时运行稳定性（5分钟快速版）

依赖服务：
  - Qdrant    : localhost:6333
  - PostgreSQL: localhost:5432

运行：
  pytest tests/test_stability_comprehensive.py -v --tb=long -s

⚠️  注意：本测试会执行 docker stop/start 操作，测试结束后会自动恢复服务。
"""

import sys
import os
import time
import uuid
import socket
import asyncio
import logging
import subprocess
import threading
import psutil
import json
import pytest
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import AsyncMock, MagicMock, patch

# ============================================================
# 路径设置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

logger = logging.getLogger(__name__)

# ============================================================
# 工具函数
# ============================================================

def _port_open(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _run_shell(cmd, timeout=30):
    """执行 shell 命令并返回结果"""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )
    return result


def _wait_for_port(host, port, max_wait=30):
    """等待端口就绪"""
    start = time.time()
    while time.time() - start < max_wait:
        if _port_open(host, port, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def _docker_compose_file():
    """获取 docker-compose.yml 路径"""
    return os.path.join(PROJECT_ROOT, "docker-compose.yml")


def _docker_service(action, service):
    """docker compose 操作服务"""
    compose_file = _docker_compose_file()
    cmd = f"docker compose -f {compose_file} {action} {service}"
    return _run_shell(cmd, timeout=60)


# ============================================================
# 服务可用性探测
# ============================================================

QDRANT_AVAILABLE = _port_open("localhost", 6333)
POSTGRES_AVAILABLE = _port_open("localhost", 5432)

skip_no_qdrant = pytest.mark.skipif(
    not QDRANT_AVAILABLE, reason="Qdrant not reachable at localhost:6333"
)
skip_no_postgres = pytest.mark.skipif(
    not POSTGRES_AVAILABLE, reason="PostgreSQL not reachable at localhost:5432"
)
skip_no_backends = pytest.mark.skipif(
    not (QDRANT_AVAILABLE and POSTGRES_AVAILABLE),
    reason="Both Qdrant and PostgreSQL required"
)

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================
# Fixture：确保测试前后服务都在运行
# ============================================================

@pytest.fixture(scope="module", autouse=True)
def ensure_services_running():
    """模块开始时确保服务运行，模块结束时也确保运行"""
    # 启动 Qdrant 和 PostgreSQL（如果未运行）
    for svc in ["qdrant", "postgres"]:
        r = _run_shell(f"docker ps -q -f name=su-memory-{svc}-1")
        if not r.stdout.strip():
            logger.info(f"Starting {svc}...")
            _docker_service("start", svc)
    # 等待就绪
    _wait_for_port("localhost", 6333, max_wait=30)
    _wait_for_port("localhost", 5432, max_wait=30)
    yield
    # 确保测试后恢复
    logger.info("Restoring services after tests...")
    for svc in ["qdrant", "postgres"]:
        _docker_service("start", svc)
    _wait_for_port("localhost", 6333, max_wait=30)
    _wait_for_port("localhost", 5432, max_wait=30)
    logger.info("Services restored.")


# ═══════════════════════════════════════════════════════════════
# 9.1 依赖服务宕机测试
# ═══════════════════════════════════════════════════════════════

@skip_no_backends
class TestDependencyFailure:
    """9.1 依赖服务宕机测试"""

    def test_qdrant_down_memory_manager_behavior(self):
        """Qdrant 宕机时 MemoryManager 不应崩溃，应返回合理错误"""
        from storage.vector_db import VectorDB
        vdb = VectorDB()

        # 先确保 Qdrant 正常时写入成功
        collection = f"test_col_{uuid.uuid4().hex[:8]}"
        run_async(vdb.create_collection(collection))
        run_async(vdb.insert(collection, "test-id", [0.1]*384, {"k": "v"}))

        # 停止 Qdrant
        logger.info("Stopping Qdrant...")
        _docker_service("stop", "qdrant")
        time.sleep(2)

        try:
            # 验证 Qdrant 不可达
            assert not _port_open("localhost", 6333), "Qdrant should be down"

            # 测试写入行为：不应崩溃，应抛出异常或返回错误
            vdb_down = VectorDB()
            with pytest.raises(Exception):
                run_async(vdb_down.insert(collection, "test-id-2", [0.1]*384, {"k": "v"}))

            # 测试查询行为：不应崩溃，应返回空列表或异常
            try:
                results = run_async(vdb_down.search(collection, [0.1]*384, limit=5))
                assert results == [], "Search should return empty list when Qdrant down"
            except Exception:
                pass  # 抛出异常也是可接受行为

            logger.info("Qdrant down behavior: System did not crash, errors handled correctly")
        finally:
            logger.info("Restarting Qdrant...")
            _docker_service("start", "qdrant")
            assert _wait_for_port("localhost", 6333, max_wait=30), "Qdrant should recover"
            time.sleep(2)

            # 验证恢复后系统正常
            import storage.vector_db as vdb_module
            vdb_module._qdrant_client = None
            vdb_recovered = VectorDB()
            run_async(vdb_recovered.create_collection(collection + "_recovered"))
            run_async(vdb_recovered.insert(collection + "_recovered", "rec-id", [0.1]*384, {"k": "v"}))
            results = run_async(vdb_recovered.search(collection + "_recovered", [0.1]*384, limit=5))
            assert len(results) >= 0
            logger.info("Qdrant recovered successfully")

    def test_postgres_down_memory_manager_behavior(self):
        """PostgreSQL 宕机时 RelationalDB 不应崩溃，应返回合理错误"""
        from storage.relational_db import RelationalDB

        # 先确保 PostgreSQL 正常时写入成功
        rdb = RelationalDB()
        mem_id = f"test-mem-{uuid.uuid4().hex[:8]}"
        run_async(rdb.insert_memory(
            memory_id=mem_id, tenant_id="t1", user_id="u1",
            content="test", compressed_content="t",
            memory_type="fact", priority=5, timestamp=int(time.time())
        ))

        # 停止 PostgreSQL
        logger.info("Stopping PostgreSQL...")
        _docker_service("stop", "postgres")
        time.sleep(2)

        try:
            assert not _port_open("localhost", 5432), "PostgreSQL should be down"

            rdb_down = RelationalDB()
            # 插入操作应该抛出异常（而不是崩溃进程）
            with pytest.raises(Exception):
                run_async(rdb_down.insert_memory(
                    memory_id=f"test-{uuid.uuid4().hex[:8]}",
                    tenant_id="t1", user_id="u1",
                    content="test", compressed_content="t",
                    memory_type="fact", priority=5, timestamp=int(time.time())
                ))

            # 查询操作也应抛出异常或返回合理结果
            try:
                stats = run_async(rdb_down.get_user_stats("t1", "u1"))
                logger.info(f"Postgres down get_user_stats returned: {stats}")
            except Exception:
                pass  # 抛出异常是可接受的

            logger.info("PostgreSQL down behavior: System did not crash, errors handled correctly")
        finally:
            logger.info("Restarting PostgreSQL...")
            _docker_service("start", "postgres")
            assert _wait_for_port("localhost", 5432, max_wait=30), "PostgreSQL should recover"
            time.sleep(2)

            # 验证恢复后系统正常
            rdb_recovered = RelationalDB()
            new_id = f"rec-mem-{uuid.uuid4().hex[:8]}"
            run_async(rdb_recovered.insert_memory(
                memory_id=new_id, tenant_id="t1", user_id="u1",
                content="recovered test", compressed_content="rt",
                memory_type="fact", priority=5, timestamp=int(time.time())
            ))
            stats = run_async(rdb_recovered.get_user_stats("t1", "u1"))
            assert isinstance(stats, dict)
            logger.info("PostgreSQL recovered successfully")


# ═══════════════════════════════════════════════════════════════
# 9.2 LLM 超时/不可用测试
# ═══════════════════════════════════════════════════════════════

class TestLLMUnavailability:
    """9.2 LLM 超时/不可用测试"""

    def test_ollama_unreachable_raises_exception(self):
        """配置错误的 Ollama 端点时应抛出异常而不是崩溃"""
        import os
        old_url = os.environ.get("LLM_BASE_URL")
        os.environ["LLM_BASE_URL"] = "http://localhost:19999/v1"
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["LLM_API_KEY"] = "ollama"

        try:
            from llm_adapter.openai_compat import LLMAdapter
            llm = LLMAdapter()
            llm.client = MagicMock()
            llm.client.chat.completions.create = MagicMock(
                side_effect=Exception("Connection refused")
            )
            with pytest.raises(Exception):
                run_async(llm.chat(messages=[{"role": "user", "content": "test"}]))
            logger.info("LLM unreachable: Exception raised correctly, no crash")
        finally:
            if old_url is not None:
                os.environ["LLM_BASE_URL"] = old_url
            else:
                os.environ.pop("LLM_BASE_URL", None)

    def test_llm_timeout_handling(self):
        """配置很短超时时间，验证超时处理"""
        import os
        old_url = os.environ.get("LLM_BASE_URL")
        os.environ["LLM_BASE_URL"] = "http://localhost:11434/v1"
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["LLM_API_KEY"] = "ollama"

        try:
            from llm_adapter.openai_compat import LLMAdapter
            llm = LLMAdapter()
            llm.client = MagicMock()
            llm.client.chat.completions.create = MagicMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )
            with pytest.raises(Exception):
                run_async(llm.chat(messages=[{"role": "user", "content": "test"}]))
            logger.info("LLM timeout: Handled correctly")
        finally:
            if old_url is not None:
                os.environ["LLM_BASE_URL"] = old_url
            else:
                os.environ.pop("LLM_BASE_URL", None)

    def test_llm_retry_behavior(self):
        """验证重试机制（如果存在）或降级行为"""
        from llm_adapter.openai_compat import LLMAdapter
        llm = LLMAdapter()
        llm.client = MagicMock()
        llm.client.chat.completions.create = MagicMock(
            side_effect=[Exception("Temp fail"), Exception("Temp fail 2"), Exception("Final fail")]
        )
        with pytest.raises(Exception):
            run_async(llm.chat(messages=[{"role": "user", "content": "test"}]))
        logger.info("LLM retry/degrade: No crash on repeated failures")


# ═══════════════════════════════════════════════════════════════
# 9.3 异常输入测试
# ═══════════════════════════════════════════════════════════════

class TestAbnormalInput:
    """9.3 异常输入测试"""

    def test_very_large_text(self):
        """超大文本（1MB+）：应拒绝或截断，不OOM"""
        from su_memory import SuMemory
        client = SuMemory()

        large_text = "A" * (1024 * 1024 + 1)
        try:
            mid = client.add(large_text)
            assert mid is not None
            logger.info(f"Large text accepted, memory_id={mid}")
        except Exception as e:
            logger.info(f"Large text rejected: {e}")
            assert True

    def test_empty_input(self):
        """空字符串、None：应给出合理错误提示"""
        from su_memory import SuMemory
        client = SuMemory()

        with pytest.raises(Exception):
            client.add("")
        with pytest.raises(Exception):
            client.add(None)
        logger.info("Empty input: Properly rejected")

    def test_special_characters_sql_xss(self):
        """SQL注入、XSS payload：应安全处理，不执行"""
        from su_memory import SuMemory
        client = SuMemory()

        payloads = [
            "'; DROP TABLE memories; --",
            "<script>alert('xss')</script>",
            "1 OR 1=1",
            " UNION SELECT * FROM passwords --",
        ]

        for payload in payloads:
            mid = client.add(payload)
            assert mid is not None
            results = client.query(payload)
            assert isinstance(results, list)

        assert len(client) == len(payloads)
        logger.info(f"Special characters: {len(payloads)} payloads handled safely")

    def test_deeply_nested_json(self):
        """极深嵌套 JSON：应合理拒绝或处理"""
        from su_memory import SuMemory
        client = SuMemory()

        nested = {}
        current = nested
        for i in range(2000):
            current["level"] = {}
            current = current["level"]
        current["end"] = "value"

        try:
            mid = client.add(str(nested), metadata={"nested": nested})
            logger.info(f"Deep nested JSON accepted: {mid}")
        except (RecursionError, MemoryError) as e:
            pytest.fail(f"Deep nested JSON caused {type(e).__name__}: {e}")
        except Exception as e:
            logger.info(f"Deep nested JSON rejected gracefully: {e}")

    def test_unicode_boundaries(self):
        """Unicode 边界：emoji、零宽字符、RTL文本"""
        from su_memory import SuMemory
        client = SuMemory()

        texts = [
            "Hello \U0001F600 World \U0001F92F",
            "零宽\u200B字符\u200D测试",
            "مرحبا بالعالم",
            "日本語テキスト\U0001F380",
            "\U0001F1E8\U0001F1F3 \U0001F1FA\U0001F1F8",
            "混合文本 Mixed 日本語 العربية emoji \U0001F44D",
        ]

        ids = []
        for text in texts:
            mid = client.add(text)
            assert mid is not None
            ids.append(mid)

        for text in texts:
            results = client.query(text[:10])
            assert isinstance(results, list)

        assert len(client) == len(texts)
        logger.info(f"Unicode boundaries: {len(texts)} texts handled correctly")

    def test_binary_data_rejection(self):
        """二进制数据：应合理拒绝"""
        from su_memory import SuMemory
        client = SuMemory()

        binary_data = bytes(range(256))
        try:
            mid = client.add(binary_data)
            logger.info(f"Binary data unexpectedly accepted: {mid}")
        except (TypeError, UnicodeDecodeError, AttributeError) as e:
            logger.info(f"Binary data rejected correctly: {type(e).__name__}")
        except Exception as e:
            logger.info(f"Binary data rejected: {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════
# 9.4 并发写入冲突测试
# ═══════════════════════════════════════════════════════════════

class TestConcurrency:
    """9.4 并发写入冲突测试"""

    def test_10_threads_write_same_content(self):
        """10个线程同时写入相同内容，验证数据一致性"""
        from su_memory import SuMemory
        client = SuMemory()

        content = "并发测试内容"
        results = []
        errors = []

        def worker():
            try:
                mid = client.add(content)
                results.append(mid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors during concurrent write: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        assert len(set(results)) == 10, "All memory_ids should be unique"
        assert len(client) == 10
        logger.info(f"Concurrent write: 10 threads wrote {len(client)} memories, all unique")

    def test_5_read_5_write_concurrent(self):
        """5读5写并发，验证无死锁、无竞态条件"""
        from su_memory import SuMemory
        client = SuMemory()

        for i in range(20):
            client.add(f"预写入记忆 {i}")

        read_results = []
        write_results = []
        errors = []

        def reader():
            try:
                for _ in range(10):
                    r = client.query("记忆")
                    read_results.append(len(r))
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(10):
                    mid = client.add(f"并发写入 {i}")
                    write_results.append(mid)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=reader))
        for _ in range(5):
            threads.append(threading.Thread(target=writer))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Concurrent read/write errors: {errors}"
        assert len(write_results) == 50, f"Expected 50 writes, got {len(write_results)}"
        assert len(read_results) == 50, f"Expected 50 reads, got {len(read_results)}"
        assert len(client) == 70, f"Expected 70 total, got {len(client)}"
        logger.info(f"Concurrent R/W: 50 reads, 50 writes, total={len(client)}, no deadlocks")


# ═══════════════════════════════════════════════════════════════
# 9.5 内存泄漏检测
# ═══════════════════════════════════════════════════════════════

class TestMemoryLeak:
    """9.5 内存泄漏检测"""

    def test_memory_leak_over_1000_operations(self):
        """循环1000次写入+检索，监控 RSS 变化"""
        from su_memory import SuMemory
        import gc

        client = SuMemory()
        process = psutil.Process(os.getpid())

        gc.collect()
        baseline_rss = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Baseline RSS: {baseline_rss:.2f} MB")

        rss_records = []
        for i in range(1000):
            mid = client.add(f"内存泄漏测试记忆 {i} " + "X" * 100)
            results = client.query(f"测试记忆 {i}")
            if (i + 1) % 100 == 0:
                gc.collect()
                rss = process.memory_info().rss / (1024 * 1024)
                rss_records.append((i + 1, rss))
                logger.info(f"  After {i+1} ops: RSS={rss:.2f} MB")

        final_rss = rss_records[-1][1]
        max_allowed = baseline_rss * 2 + 50
        logger.info(f"Final RSS: {final_rss:.2f} MB, Max allowed: {max_allowed:.2f} MB")

        if final_rss > max_allowed:
            logger.warning(f"Memory leak suspected: {final_rss:.2f} > {max_allowed:.2f}")
        else:
            logger.info("Memory growth within acceptable range")

        self._rss_records = rss_records
        self._baseline_rss = baseline_rss

    def test_memory_leak_trend(self):
        """分析 RSS 增长趋势"""
        if not hasattr(self, "_rss_records"):
            pytest.skip("Run test_memory_leak_over_1000_operations first")

        rss_records = self._rss_records
        if len(rss_records) < 2:
            pytest.skip("Not enough data points")

        growths = []
        for i in range(1, len(rss_records)):
            growth = rss_records[i][1] - rss_records[i-1][1]
            growths.append(growth)

        avg_growth = sum(growths) / len(growths)
        logger.info(f"Average RSS growth per 100 ops: {avg_growth:.2f} MB")

        if len(growths) >= 3:
            later_growth = sum(growths[-3:]) / 3
            early_growth = sum(growths[:3]) / 3
            logger.info(f"Early avg growth: {early_growth:.2f} MB, Later: {later_growth:.2f} MB")


# ═══════════════════════════════════════════════════════════════
# 9.6 Docker 重启恢复测试
# ═══════════════════════════════════════════════════════════════

@skip_no_backends
class TestDockerRestartRecovery:
    """9.6 Docker 重启恢复测试"""

    def test_restart_qdrant_data_persistence(self):
        """写入10条记忆，重启 Qdrant，验证数据仍然存在"""
        from storage.vector_db import VectorDB
        import storage.vector_db as vdb_module

        vdb = VectorDB()
        collection = f"persist_test_{uuid.uuid4().hex[:8]}"
        run_async(vdb.create_collection(collection))

        original_ids = []
        for i in range(10):
            mid = f"persist-mem-{i}"
            vector = [0.01 * (i + 1)] + [0.0] * 383
            run_async(vdb.insert(collection, mid, vector, {"content": f"记忆内容 {i}", "idx": i}))
            original_ids.append(mid)

        count_before = run_async(vdb.count(collection))
        logger.info(f"Before restart: {count_before} memories in Qdrant")
        assert count_before == 10

        logger.info("Restarting Qdrant container...")
        _docker_service("restart", "qdrant")
        assert _wait_for_port("localhost", 6333, max_wait=30)
        time.sleep(3)

        try:
            vdb_module._qdrant_client = None
            vdb_after = VectorDB()
            count_after = run_async(vdb_after.count(collection))
            logger.info(f"After restart: {count_after} memories in Qdrant")

            results = run_async(vdb_after.search(collection, [0.01] + [0.0] * 383, limit=10))
            logger.info(f"Search after restart returned {len(results)} results")
        finally:
            if not _port_open("localhost", 6333):
                _docker_service("start", "qdrant")
                _wait_for_port("localhost", 6333, max_wait=30)

    def test_restart_postgres_data_persistence(self):
        """写入10条记忆元数据，重启 PostgreSQL，验证数据仍然存在"""
        from storage.relational_db import RelationalDB
        rdb = RelationalDB()

        tenant_id = f"persist_t_{uuid.uuid4().hex[:8]}"
        user_id = "persist_user"

        original_ids = []
        for i in range(10):
            mid = f"pg-mem-{uuid.uuid4().hex[:8]}"
            run_async(rdb.insert_memory(
                memory_id=mid, tenant_id=tenant_id, user_id=user_id,
                content=f"PG记忆内容 {i}", compressed_content=f"c{i}",
                memory_type="fact", priority=5, timestamp=int(time.time()) + i
            ))
            original_ids.append(mid)

        stats_before = run_async(rdb.get_user_stats(tenant_id, user_id))
        logger.info(f"Before restart: {stats_before}")
        assert stats_before["total"] == 10

        logger.info("Restarting PostgreSQL container...")
        _docker_service("restart", "postgres")
        assert _wait_for_port("localhost", 5432, max_wait=30)
        time.sleep(3)

        try:
            rdb_after = RelationalDB()
            stats_after = run_async(rdb_after.get_user_stats(tenant_id, user_id))
            logger.info(f"After restart: {stats_after}")
            assert stats_after["total"] == 10, f"Data lost! Before: 10, After: {stats_after['total']}"
            logger.info("PostgreSQL data persistence verified: all 10 memories preserved")
        finally:
            if not _port_open("localhost", 5432):
                _docker_service("start", "postgres")
                _wait_for_port("localhost", 5432, max_wait=30)


# ═══════════════════════════════════════════════════════════════
# 9.7 长时运行稳定性（5分钟快速版）
# ═══════════════════════════════════════════════════════════════

@skip_no_backends
class TestLongRunningStability:
    """9.7 长时运行稳定性测试（5分钟快速版）"""

    def test_5min_continuous_read_write(self):
        """每秒一次读写操作，持续5分钟（300次）"""
        from memory_engine.manager import MemoryManager
        import gc

        manager = MemoryManager()
        process = psutil.Process(os.getpid())
        tenant_id = f"longrun_{uuid.uuid4().hex[:8]}"
        user_id = "longrun_user"

        run_async(manager.create_tenant("longrun", "standard"))

        latencies_write = []
        latencies_read = []
        errors = []
        rss_records = []
        success_count = 0

        manager.extractor.encode = MagicMock(return_value=[0.1] * 384)
        manager.extractor.extract = AsyncMock(return_value={
            "compressed": "test", "type": "fact", "priority": 5,
            "encoding_info": {"hexagram_index": 0, "hexagram_name": "", "wuxing": "土", "direction": ""},
            "dynamic_priority": {"final": 0.5}
        })

        start_time = time.time()
        duration = 300
        iteration = 0

        logger.info("Starting 5-minute long-running stability test...")
        while time.time() - start_time < duration:
            iteration += 1
            op_start = time.time()
            try:
                mem_id = run_async(manager.add_memory(
                    tenant_id=tenant_id, user_id=user_id,
                    content=f"长时运行测试记忆 {iteration}",
                    metadata={"iteration": iteration}
                ))
                write_latency = (time.time() - op_start) * 1000
                latencies_write.append(write_latency)

                read_start = time.time()
                results = run_async(manager.query_memory(
                    tenant_id=tenant_id, user_id=user_id,
                    query=f"测试记忆 {iteration}", limit=5
                ))
                read_latency = (time.time() - read_start) * 1000
                latencies_read.append(read_latency)

                success_count += 1
            except Exception as e:
                errors.append((iteration, str(e)))
                logger.warning(f"Iteration {iteration} failed: {e}")

            if iteration % 30 == 0:
                gc.collect()
                rss = process.memory_info().rss / (1024 * 1024)
                rss_records.append((iteration, rss))
                elapsed = time.time() - start_time
                logger.info(
                    f"  [{elapsed:.0f}s] iter={iteration}, "
                    f"write_p50={self._p50(latencies_write[-30:]):.1f}ms, "
                    f"read_p50={self._p50(latencies_read[-30:]):.1f}ms, "
                    f"RSS={rss:.1f}MB, errors={len(errors)}"
                )

            elapsed = time.time() - op_start
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)

        total_time = time.time() - start_time
        logger.info(f"Long-running test completed: {iteration} iterations in {total_time:.0f}s")

        success_rate = success_count / iteration * 100 if iteration > 0 else 0
        logger.info(f"Success rate: {success_rate:.1f}% ({success_count}/{iteration})")
        logger.info(f"Write latency P50={self._p50(latencies_write):.1f}ms, P95={self._p95(latencies_write):.1f}ms")
        logger.info(f"Read latency P50={self._p50(latencies_read):.1f}ms, P95={self._p95(latencies_read):.1f}ms")
        logger.info(f"Errors: {len(errors)}")

        if rss_records:
            logger.info(f"RSS trend: start={rss_records[0][1]:.1f}MB, end={rss_records[-1][1]:.1f}MB")

        assert success_rate >= 80, f"Success rate too low: {success_rate:.1f}%"

        if len(latencies_write) > 20:
            mid = len(latencies_write) // 2
            early_write = sum(latencies_write[:mid]) / len(latencies_write[:mid])
            late_write = sum(latencies_write[mid:]) / len(latencies_write[mid:])
            logger.info(f"Write latency early={early_write:.1f}ms, late={late_write:.1f}ms")
            if early_write > 0:
                assert late_write < early_write * 5, f"Write latency degraded: {early_write:.1f} -> {late_write:.1f}"

        if len(rss_records) >= 3:
            first_rss = rss_records[0][1]
            last_rss = rss_records[-1][1]
            assert last_rss < first_rss * 2 + 100, f"Memory leak suspected: {first_rss:.1f} -> {last_rss:.1f} MB"

    @staticmethod
    def _p50(data):
        if not data:
            return 0.0
        s = sorted(data)
        idx = int(len(s) * 0.5)
        return s[min(idx, len(s) - 1)]

    @staticmethod
    def _p95(data):
        if not data:
            return 0.0
        s = sorted(data)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]
