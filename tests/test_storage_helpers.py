"""
存储后端初始化辅助模块测试

覆盖: sdk/_storage_helpers 与 sdk/_storage_init 的类型映射、
sqlite 后端实际创建、未知类型回退、以及 create_backend 失败降级。
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from su_memory._sys._storage_backend import BackendType
from su_memory.sdk import _storage_helpers as helpers
from su_memory.sdk import _storage_init as sdk_init


class StubInstance:
    """模拟 SuMemoryLite/SuMemoryLitePro 的最小属性面"""

    def __init__(self):
        self._storage_backend = None
        self._storage_backend_type = None


class _FakeLoop:
    """模拟"无运行中事件循环"状态, 让初始化走 asyncio.run 分支"""

    def is_running(self) -> bool:
        return False


class TestTypeMap:
    def test_get_type_map_fills_and_caches(self):
        m = helpers._get_type_map()
        assert m["sqlite"] is BackendType.SQLITE
        assert m["postgresql"] is BackendType.POSTGRESQL
        assert m["redis"] is BackendType.REDIS
        assert m["auto"] is BackendType.AUTO
        assert helpers._get_type_map() is m

    def test_sdk_init_type_map(self):
        m = sdk_init._get_type_map()
        assert m["sqlite"] is BackendType.SQLITE


class TestHelpersInit:
    def test_sqlite_backend_created(self, tmp_path):
        inst = StubInstance()
        helpers.init_storage_backend(inst, "sqlite", str(tmp_path), label="TestLite")
        assert inst._storage_backend is not None
        assert inst._storage_backend.backend_type is BackendType.SQLITE

    def test_unknown_type_falls_back_to_sqlite(self, tmp_path):
        inst = StubInstance()
        helpers.init_storage_backend(inst, "unknown_xyz", str(tmp_path))
        assert inst._storage_backend is not None

    def test_create_backend_failure_sets_none(self, tmp_path, monkeypatch):
        def boom(*_a, **_kw):
            # 非 RuntimeError: 走优雅降级分支(RuntimeError 会触发重试语义)
            raise ConnectionError("no backend available")
        monkeypatch.setattr(asyncio, "get_event_loop", lambda: _FakeLoop())
        monkeypatch.setattr(
            "su_memory._sys._storage_backend.create_backend", boom
        )
        inst = StubInstance()
        helpers.init_storage_backend(inst, "sqlite", str(tmp_path))
        assert inst._storage_backend is None


class TestSdkInit:
    def test_sqlite_backend_created(self, tmp_path):
        backend = sdk_init.init_storage_backend("sqlite", str(tmp_path), "T")
        assert backend is not None
        assert backend.backend_type is BackendType.SQLITE

    def test_unknown_type_returns_none(self, tmp_path):
        assert sdk_init.init_storage_backend("unknown_xyz", str(tmp_path)) is None

    def test_create_backend_failure_returns_none(self, tmp_path, monkeypatch):
        def boom(*_a, **_kw):
            raise ConnectionError("no backend available")
        monkeypatch.setattr(asyncio, "get_event_loop", lambda: _FakeLoop())
        monkeypatch.setattr(
            "su_memory._sys._storage_backend.create_backend", boom
        )
        assert sdk_init.init_storage_backend("sqlite", str(tmp_path)) is None
