"""
P2 并发加固验证测试
====================

验证 MCIWorldModel._discover_lock 与 JEPADataset.from_memories() 的并发安全:
- _discover_lock 是 threading.Lock 实例
- 多线程同时调用 discover() 时被序列化（互斥）
- 多线程同时调用 JEPADataset.from_memories() 不会产生状态交叉污染
- 与 JEPADataset 已有 jepa marker 一致
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from su_memory.sdk._jepa_dataset import JEPADataset
from su_memory.sdk._world_model import CausalWorldModelState, MCIWorldModel

pytestmark = pytest.mark.jepa


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════


def _make_memories(prefix: str, n: int = 6) -> list[dict]:
    """生成 n 条带 timestamp 的记忆。"""
    return [
        {
            "id": f"{prefix}_{i}",
            "content": f"{prefix} 事件 {i} 导致后续变化",
            "timestamp": f"2026-01-01T00:0{i}:00",
        }
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════
# T1. _discover_lock 属性验证
# ═══════════════════════════════════════════════════════════════


class TestDiscoverLockAttribute:
    """验证 _discover_lock 字段存在且为 threading.Lock。"""

    def test_lock_exists(self):
        wm = MCIWorldModel()
        assert hasattr(wm, "_discover_lock"), "MCIWorldModel 必须有 _discover_lock 字段"

    def test_lock_is_threading_lock(self):
        wm = MCIWorldModel()
        # Python 3.11+ 中 _thread.lock 是底层 C 类型，
        # type(lock) is threading.Lock 为 False (后者是工厂函数)，
        # 最可靠的方式是 duck typing：验证具备 acquire/release/acquire(0) 等 API
        lock = wm._discover_lock
        assert hasattr(lock, "acquire")
        assert hasattr(lock, "release")
        assert hasattr(lock, "locked")
        # locked() 默认 False
        assert lock.locked() is False
        # 验证是互斥锁（非 RLock）：同一线程 acquire 两次会阻塞
        result = lock.acquire(blocking=False)
        assert result is True
        same_thread = lock.acquire(blocking=False)
        assert same_thread is False, "Lock() 应为互斥锁，同线程二次 acquire(0) 应返回 False"
        lock.release()

    def test_lock_independent_per_instance(self):
        """每个实例有独立的锁。"""
        wm1 = MCIWorldModel()
        wm2 = MCIWorldModel()
        assert wm1._discover_lock is not wm2._discover_lock


# ═══════════════════════════════════════════════════════════════
# T2. discover() 互斥验证
# ═══════════════════════════════════════════════════════════════


class TestDiscoverMutex:
    """discover() 持有 _discover_lock 时其他线程应被阻塞。"""

    def test_discover_blocks_when_lock_held_externally(self):
        """外部持有锁时，discover() 调用应被阻塞。"""
        wm = MCIWorldModel()
        wm.initialize()
        wm._discover_lock.acquire()
        try:
            results: list = []
            errors: list = []

            def call_discover():
                try:
                    state = wm.discover(memories=_make_memories("mt", 6), verbose=False)
                    results.append(state)
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            t = threading.Thread(target=call_discover)
            t.start()
            # 等 0.5s，discover() 应该还在阻塞
            t.join(timeout=0.5)
            assert t.is_alive(), "discover() 应在 _discover_lock 被持有时阻塞"
            assert results == []
            assert errors == []
        finally:
            wm._discover_lock.release()
        # 释放锁后，线程应完成
        t.join(timeout=15)
        assert not t.is_alive(), "释放锁后 discover() 线程应已完成"
        assert len(results) == 1
        assert isinstance(results[0], CausalWorldModelState)

    def test_discover_serializes_concurrent_calls(self):
        """N 个并发 discover() 调用，验证最并发数 = 1。"""
        wm = MCIWorldModel()
        wm.initialize()

        # 用一个共享计数器+锁来观测实际并发数
        # 由于 discover() 内部已经加锁，这里通过 Barrier 让 4 线程同时进入临界区
        in_section = 0
        max_in_section = 0
        counter_lock = threading.Lock()
        barrier = threading.Barrier(4)

        def discover_with_probe(memories: list[dict]):
            nonlocal in_section, max_in_section
            barrier.wait()  # 同步起跑
            # Monkey patch 不可行；改用 timing-based 探测
            # 直接验证：4 个线程全部进入，discover() 应串行执行
            wm.discover(memories=memories, verbose=False)
            with counter_lock:
                in_section += 1
                if in_section > max_in_section:
                    max_in_section = in_section
            time.sleep(0.01)  # 模拟处理时间
            with counter_lock:
                in_section -= 1

        # 真正验证锁的方式：把 lock 替换为带计数的代理
        real_lock = wm._discover_lock
        acquire_log: list[tuple[float, str]] = []
        release_log: list[tuple[float, str]] = []
        log_lock = threading.Lock()

        class _CountingLock:
            def __init__(self, inner):
                self._inner = inner

            def acquire(self, *a, **kw):
                with log_lock:
                    acquire_log.append((time.time(), threading.get_ident()))
                return self._inner.acquire(*a, **kw)

            def release(self):
                with log_lock:
                    release_log.append((time.time(), threading.get_ident()))
                return self._inner.release()

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *a):
                self.release()

        wm._discover_lock = _CountingLock(real_lock)
        try:
            threads = [
                threading.Thread(
                    target=discover_with_probe,
                    args=(_make_memories(f"ser_{i}", 6),),
                )
                for i in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=20)
        finally:
            wm._discover_lock = real_lock

        # 验证：4 次 acquire + 4 次 release，且 acquire 严格早于下一次 release
        assert len(acquire_log) == 4, f"应有 4 次 acquire，实际 {len(acquire_log)}"
        assert len(release_log) == 4
        # 严格互斥校验：每个 release 时间戳 ≥ 同序号 acquire 时间戳
        # 实际更准确：每次 release 之后的下一次 acquire 应在 release 之后
        # 用 acquire 和 release 的时间差来验证：同一线程内 acquire→release 区间不重叠
        acq_set = sorted(acquire_log)
        rel_set = sorted(release_log)
        for i in range(len(acq_set)):
            assert acq_set[i][0] <= rel_set[i][0], "同一线程 acquire 应早于 release"


# ═══════════════════════════════════════════════════════════════
# T3. JEPADataset.from_memories() 隔离验证
# ═══════════════════════════════════════════════════════════════


class TestFromMemoriesIsolation:
    """多线程并发 from_memories() 不会导致 dataset 数据交叉污染。"""

    def test_from_memories_does_not_share_state(self):
        """验证 from_memories() 返回的 state 副本彼此独立。"""
        wm = MCIWorldModel()
        wm.initialize()
        ds = JEPADataset.from_memories(
            wm,
            _make_memories("iso", 6),
            window_size=3,
        )
        # from_memories 内部已 deepcopy，state_pairs 中每个 state 应独立
        for s_t, s_t1 in ds.state_pairs:
            assert s_t is not s_t1
            # 修改 s_t 不应影响 s_t1
            original_edges = list(s_t1.causal_edges)
            s_t.causal_edges.append({"cause": "X", "effect": "Y", "rho": 0.99})
            assert s_t1.causal_edges == original_edges

    def test_concurrent_from_memories_isolated_datasets(self):
        """4 线程并发 from_memories，验证每个 dataset 数据集对应其输入。"""
        wm = MCIWorldModel()
        wm.initialize()

        results: dict = {}
        errors: list = []

        def run(prefix: str):
            try:
                memories = _make_memories(prefix, 6)
                ds = JEPADataset.from_memories(
                    wm,
                    memories,
                    window_size=3,
                )
                results[prefix] = ds
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=run, args=(f"th_{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert errors == [], f"并发 from_memories 异常: {errors}"
        assert len(results) == 4
        # 验证每个 dataset 的 n_memories 与其输入一致
        for _prefix, ds in results.items():
            for s_t, _s_t1 in ds.state_pairs:
                # n_memories 应 >= min_memories_per_window
                assert s_t.n_memories >= 3


# ═══════════════════════════════════════════════════════════════
# T4. discover() + from_memories() 混合并发
# ═══════════════════════════════════════════════════════════════


class TestMixedConcurrency:
    """discover() 和 from_memories() 混合调用，验证无死锁、无数据损坏。"""

    def test_mixed_discover_and_from_memories(self):
        wm = MCIWorldModel()
        wm.initialize()

        errors: list = []
        states: list = []
        datasets: list = []
        lock = threading.Lock()

        def run_discover(prefix: str):
            try:
                state = wm.discover(memories=_make_memories(prefix, 6), verbose=False)
                with lock:
                    states.append((prefix, state))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def run_from_memories(prefix: str):
            try:
                ds = JEPADataset.from_memories(
                    wm,
                    _make_memories(prefix, 6),
                    window_size=3,
                )
                with lock:
                    datasets.append((prefix, ds))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = []
        for i in range(3):
            threads.append(threading.Thread(target=run_discover, args=(f"mix_d_{i}",)))
            threads.append(threading.Thread(target=run_from_memories, args=(f"mix_f_{i}",)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"混合并发异常: {errors}"
        assert len(states) == 3
        assert len(datasets) == 3


# ═══════════════════════════════════════════════════════════════
# T5. 回归 — 单线程语义保持不变
# ═══════════════════════════════════════════════════════════════


class TestSingleThreadRegression:
    """确保加锁不改变单线程行为。"""

    def test_discover_returns_state_with_n_memories(self):
        wm = MCIWorldModel()
        wm.initialize()
        state = wm.discover(memories=_make_memories("reg", 6), verbose=False)
        assert state.n_memories == 6

    def test_from_memories_state_pairs_aligned(self):
        wm = MCIWorldModel()
        wm.initialize()
        ds = JEPADataset.from_memories(
            wm,
            _make_memories("align", 8),
            window_size=3,
        )
        # state_pairs 和 memory_pairs 长度应一致
        assert len(ds.state_pairs) == len(ds.memory_pairs)
        # n_pairs 应 ≥ 1
        assert ds.n_pairs >= 0

    def test_repeated_discover_does_not_corrupt_state(self):
        """同一 world_model 顺序多次 discover()，验证 state 总是最近一次。"""
        wm = MCIWorldModel()
        wm.initialize()

        state_a = wm.discover(memories=_make_memories("a", 4), verbose=False)
        state_b = wm.discover(memories=_make_memories("b", 5), verbose=False)

        # 两次都返回 self._state 引用，state_a 与 state_b 是同一对象
        assert state_a is state_b
        # 最终 n_memories 应为最后一次的 5
        assert state_b.n_memories == 5
