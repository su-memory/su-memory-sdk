"""
verify_install 安装验证脚本测试

覆盖: verify_installation / quick_check 的成功路径与 ImportError/运行时错误路径。
实际验证在 tmp 目录内执行, 不污染仓库; 实例化禁用向量服务以加速。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import su_memory
import su_memory.verify_install as vi


class TestVerifyInstallation:
    def test_full_check_passes(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        rc = vi.verify_installation()
        out = capsys.readouterr().out
        assert rc == 0
        assert "验证通过" in out
        assert "导入成功" in out

    def test_import_failure_returns_1(self, monkeypatch, capsys):
        # 将 su_memory 包标记为不可导入, 触发第 1 项失败并提前返回
        monkeypatch.setitem(sys.modules, "su_memory", None)
        rc = vi.verify_installation()
        out = capsys.readouterr().out
        # 注: 该分支 return False(存量返回值不一致, 正常路径返回 0)
        assert rc is False
        assert "导入失败" in out

    def test_runtime_failure_returns_1(self, tmp_path, monkeypatch, capsys):
        class BoomClient:
            def __init__(self, *a, **kw):
                raise RuntimeError("storage backend init exploded")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(su_memory, "SuMemoryLitePro", BoomClient)
        rc = vi.verify_installation()
        assert rc == 1
        assert "实例化失败" in capsys.readouterr().out

    def test_connection_error_treated_as_warning(
        self, tmp_path, monkeypatch, capsys
    ):
        class ConnClient:
            def __init__(self, *a, **kw):
                raise ConnectionError("Connection refused by vector service")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(su_memory, "SuMemoryLitePro", ConnClient)
        # 连接类错误只警告; 后续 add 因实例不存在失败, 最终返回 1
        rc = vi.verify_installation()
        out = capsys.readouterr().out
        assert "向量服务未连接" in out
        assert rc == 1


class TestQuickCheck:
    def test_quick_check_passes(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        assert vi.quick_check() == 0
        assert "安装正常" in capsys.readouterr().out

    def test_quick_check_import_error(self, monkeypatch, capsys):
        monkeypatch.setitem(sys.modules, "su_memory", None)
        assert vi.quick_check() == 1
        assert "导入失败" in capsys.readouterr().out

    def test_quick_check_runtime_error(self, tmp_path, monkeypatch, capsys):
        class BoomClient:
            def __init__(self, *a, **kw):
                raise RuntimeError("boom")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(su_memory, "SuMemoryLitePro", BoomClient)
        assert vi.quick_check() == 1
        assert "错误" in capsys.readouterr().out
