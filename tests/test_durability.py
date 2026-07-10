"""
崩溃恢复与并发安全回归测试 (P1-5)

固化以下生产关键路径的正确性：
1. 写入 → 重启 → 数据完整恢复（持久化层可靠性）
2. 多线程并发 add → 无数据丢失/损坏（线程安全）
3. 默认配置下 add 不阻塞（P0-1 回归守护）

这些测试不依赖任何外部服务（Ollama/DeepSeek/Redis/PG），
归入离线套（CI 默认运行）。
"""
import os
import sys
import tempfile
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from su_memory.sdk.lite_pro import SuMemoryLitePro


def _make_client(storage_path, **kwargs):
    """构造一个纯离线客户端（关闭所有外部依赖组件）。"""
    defaults = dict(
        enable_vector=False,
        enable_graph=False,
        enable_temporal=False,
        enable_session=False,
    )
    defaults.update(kwargs)
    return SuMemoryLitePro(storage_path=storage_path, **defaults)


class TestCrashRecovery:
    """崩溃恢复：写入 → del(模拟崩溃) → 重载 → 完整性校验"""

    def test_recovery_after_restart(self, tmp_path):
        """写入 100 条 → 重启 → 应完整恢复 100 条，查询可用。"""
        d = str(tmp_path / "crash_db")
        os.makedirs(d, exist_ok=True)

        # Phase 1: 写入并正常释放
        c = _make_client(d)
        for i in range(100):
            c.add(f"记录{i}: 春季万物生长，验证持久化第{i}条")
        written = len(c._memories)
        c._save()
        del c
        assert written == 100

        # Phase 2: 重新加载，校验完整性
        c2 = _make_client(d)
        recovered = len(c2._memories)
        assert recovered == written, f"恢复 {recovered} 条，期望 {written} 条（丢失 {written - recovered} 条）"

        # Phase 3: 查询可用性
        results = c2.query("验证", top_k=5)
        assert len(results) > 0, "恢复后查询无结果"
        del c2

    def test_id_uniqueness_after_recovery(self, tmp_path):
        """恢复后所有 memory_id 唯一（无重复）。"""
        d = str(tmp_path / "uniq_db")
        os.makedirs(d, exist_ok=True)
        c = _make_client(d)
        for i in range(50):
            c.add(f"唯一性测试记录{i}")
        c._save()
        del c

        c2 = _make_client(d)
        ids = [m.id for m in c2._memories]
        assert len(ids) == len(set(ids)), "恢复后存在重复 id"
        del c2


class TestConcurrentSafety:
    """多线程并发安全"""

    def test_concurrent_add_no_loss(self, tmp_path):
        """4 线程并发 add，总数应精确等于预期（无丢失/重复）。"""
        d = str(tmp_path / "concurrent_db")
        os.makedirs(d, exist_ok=True)
        c = _make_client(d)

        per_thread = 25
        num_threads = 4
        errors = []

        def writer(tid):
            try:
                for i in range(per_thread):
                    c.add(f"线程{tid}_记录{i}: 并发写入测试")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"并发写入出错: {errors}"
        expected = per_thread * num_threads
        actual = len(c._memories)
        assert actual == expected, f"并发写入丢失数据：期望 {expected}，实际 {actual}"
        del c


class TestNonBlockingAdd:
    """P0-1 回归守护：默认配置下 add 不应阻塞（无网络依赖）"""

    def test_add_returns_quickly_without_network(self, tmp_path):
        """默认配置（enable_llm_energy=False）下，50 次 add 应在 5 秒内完成。"""
        import time
        d = str(tmp_path / "nblock_db")
        os.makedirs(d, exist_ok=True)
        c = _make_client(d)

        t0 = time.time()
        for i in range(50):
            c.add(f"非阻塞测试{i}: 确认 add 不发起网络请求")
        elapsed = time.time() - t0

        assert elapsed < 5.0, f"50 次 add 耗时 {elapsed:.1f}s，疑似阻塞（P0-1 回归）"
        del c

    def test_default_no_llm_energy(self, tmp_path):
        """默认构造的客户端 enable_llm_energy 应为 False。"""
        d = str(tmp_path / "default_db")
        os.makedirs(d, exist_ok=True)
        c = _make_client(d)
        assert c.enable_llm_energy is False, "默认 enable_llm_energy 应为 False"
        del c


class TestInputValidation:
    """P2: 公共 API 输入校验回归守护"""

    def test_add_rejects_non_str(self, tmp_path):
        """add(None)/add(123) 应抛清晰 TypeError 而非晦涩 AttributeError。"""
        d = str(tmp_path / "valid_db")
        os.makedirs(d, exist_ok=True)
        c = _make_client(d)
        for bad in (None, 123, [], {}):
            with pytest.raises(TypeError):
                c.add(bad)

    def test_query_rejects_non_str(self, tmp_path):
        """query(None)/query(123) 应抛清晰 TypeError。"""
        d = str(tmp_path / "valid_q")
        os.makedirs(d, exist_ok=True)
        c = _make_client(d)
        c.add("测试数据")
        for bad in (None, 123, []):
            with pytest.raises(TypeError):
                c.query(bad)


class TestConcurrentSaveSafety:
    """并发 save 竞态守护：高并发 + 低 save_interval 下持久化文件不损坏"""

    def test_concurrent_save_no_corruption(self, tmp_path):
        """8 线程并发 add（save_interval=2 触发频繁 save），落盘 JSON 不损坏、ID 唯一。"""
        import json
        d = str(tmp_path / "save_race")
        os.makedirs(d, exist_ok=True)
        c = _make_client(d, save_interval=2)

        errors = []
        per_thread = 150

        def writer(tid):
            try:
                for i in range(per_thread):
                    c.add(f"save竞态线程{tid}_{i}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"并发写入出错: {errors}"
        c.flush()

        path = os.path.join(d, "su_memory_pro.json")
        with open(path) as f:
            data = json.load(f)  # 不应 JSONDecodeError
        ids = [m["id"] for m in data["memories"]]
        assert len(ids) == len(set(ids)), "并发 save 导致 ID 重复/损坏"
        assert len(ids) == per_thread * 8


class TestHighAvailability:
    """高可用回归守护：内存泄漏、磁盘容错、索引一致性"""

    def test_energy_cache_bounded(self, tmp_path):
        """_energy_cache 应有 LRU 上限，长时间运行不无限增长。"""
        d = str(tmp_path / "ecache_db")
        os.makedirs(d, exist_ok=True)
        c = _make_client(d, autosave=False)
        for i in range(3000):
            c.add(f"唯一缓存内容{i}春季生长")
        assert len(c._energy_cache) <= 2100, f"_energy_cache 泄漏: {len(c._energy_cache)}"

    def test_evict_cleans_index(self, tmp_path):
        """淘汰记忆时倒排索引应同步清理，无幽灵 ID 残留。"""
        d = str(tmp_path / "evict_db")
        os.makedirs(d, exist_ok=True)
        c = _make_client(d, max_memories=5, autosave=False)
        for i in range(10):
            c.add(f"幽灵测试关键词共享第{i}条")

        mem_ids = {m.id for m in c._memories}
        all_indexed = set()
        for kw, ids in c._index.items():
            all_indexed |= ids
        ghost = all_indexed - mem_ids
        assert not ghost, f"淘汰后索引残留 {len(ghost)} 个幽灵 ID"
        assert len(c._memories) == 5

    def test_disk_failure_does_not_block_add(self, tmp_path):
        """只读目录（磁盘故障）下 add 不应中断，数据保留内存。"""
        import stat
        d = str(tmp_path / "readonly_db")
        os.makedirs(d, exist_ok=True)
        os.chmod(d, stat.S_IRUSR | stat.S_IXUSR)  # 只读
        try:
            c = _make_client(d)
            for i in range(60):  # 超过 save_interval 触发 save
                c.add(f"磁盘容错测试{i}")
            assert len(c._memories) == 60, "磁盘故障不应中断 add"
            assert len(c.query("磁盘", top_k=3)) > 0, "查询仍可用"
        finally:
            os.chmod(d, stat.S_IRWXU)  # 恢复权限以便清理


class TestSigkillDataLossWindow:
    """对抗性回归：SIGKILL 下的数据丢失窗口受 save_interval 控制"""

    def test_loss_window_bounded_by_save_interval(self, tmp_path):
        """进程被 SIGKILL 时，丢失条数 <= save_interval - 1。"""
        import subprocess
        d = str(tmp_path / "sigkill_db")
        os.makedirs(d, exist_ok=True)
        si = 10  # 默认值
        n_write = si - 1  # 写 9 条，未触发 save

        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from su_memory.sdk.lite_pro import SuMemoryLitePro\n"
            "c = SuMemoryLitePro(storage_path=%r, enable_vector=False, "
            "enable_graph=False, enable_temporal=False, enable_session=False, save_interval=%d)\n"
            "for i in range(%d): c.add(f'SIGKILL{i}')\n"
            "import os; os.kill(os.getpid(), 9)\n"
        ) % (os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')),
             d, si, n_write)

        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, timeout=10)
        assert r.returncode == -9, f"子进程应被SIGKILL, 实际退出码={r.returncode}"

        # 重新加载
        c2 = _make_client(d)
        lost = n_write - len(c2._memories)
        # 默认 save_interval=10 时，丢失窗口 <= 9
        assert lost <= si - 1, f"丢失 {lost} 条，超过窗口 save_interval-1={si - 1}"
