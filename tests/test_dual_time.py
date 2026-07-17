"""
C4 双时间模型测试

验证：区分事件发生时间 vs 记录入库时间，时序检索/衰减基于事件时间。
医疗级要求：历史检验结果晚录入时，时间排序正确。
"""
from __future__ import annotations

import os
import time
import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


class TestDualTimeModel:
    """双时间模型测试"""

    def test_default_event_time_equals_record_time(self, tmp_path):
        """不传 event_time 时，event_time = 入库时间"""
        from su_memory.clinical import ClinicalMemoryClient
        client = ClinicalMemoryClient(
            storage_path=str(tmp_path / "dt"),
            embedding_backend="none",
            compliance_level=None,
            safety_screen=False,
        )
        before = int(time.time())
        client.add_patient_event("P001", "即时事件", "plan")
        after = int(time.time())
        hits = client.recall("P001", "即时事件")
        assert len(hits) >= 1
        et = hits[0]["event_time"]
        assert before <= et <= after  # event_time ≈ 入库时间

    def test_explicit_event_time_independent(self, tmp_path):
        """显式 event_time 应独立于入库时间"""
        from su_memory.clinical import ClinicalMemoryClient
        client = ClinicalMemoryClient(
            storage_path=str(tmp_path / "dt2"),
            embedding_backend="none",
            compliance_level=None,
            safety_screen=False,
        )
        # 事件发生在 3 天前
        three_days_ago = int(time.time()) - 3 * 86400
        client.add_patient_event(
            "P001", "三天前的检验结果", "lab_result",
            event_time=three_days_ago,
        )
        hits = client.recall("P001", "检验结果")
        assert len(hits) >= 1
        assert hits[0]["event_time"] == three_days_ago
        # event_time != timestamp（入库时间是现在）
        assert hits[0]["event_time"] < hits[0]["timestamp"]

    def test_lab_trend_uses_event_time(self, tmp_path):
        """检验趋势排序应基于事件时间，非入库时间"""
        from su_memory.clinical import ClinicalMemoryClient
        client = ClinicalMemoryClient(
            storage_path=str(tmp_path / "dt3"),
            embedding_backend="none",
            compliance_level=None,
            safety_screen=False,
        )
        # 先录入较新的检验（但事件时间较早）
        client.add_lab_value("P001", "白蛋白", 35.0, "g/L", "35-55",
                             event_time=int(time.time()) - 86400)  # 1天前
        # 后录入较旧的检验（但事件时间更早）
        client.add_lab_value("P001", "白蛋白", 30.0, "g/L", "35-55",
                             event_time=int(time.time()) - 3 * 86400)  # 3天前

        trend = client.get_lab_trend("P001", "白蛋白")
        # 按事件时间升序：30.0（3天前）→ 35.0（1天前）
        assert len(trend.values) == 2
        assert trend.values[0] == 30.0  # 更早的事件排前面
        assert trend.values[1] == 35.0

    def test_backward_compat_no_event_time(self, tmp_path):
        """旧调用（无 event_time）仍正常工作"""
        from su_memory.clinical import ClinicalMemoryClient
        client = ClinicalMemoryClient(
            storage_path=str(tmp_path / "dt4"),
            embedding_backend="none",
            compliance_level=None,
            safety_screen=False,
        )
        mid = client.add_patient_event("P001", "兼容测试", "plan")
        assert mid is not None
        hits = client.recall("P001", "兼容")
        assert len(hits) >= 1
        # event_time 字段存在（=入库时间）
        assert "event_time" in hits[0]


class TestEngineDualTime:
    """引擎层双时间测试"""

    def test_engine_add_with_event_time(self, tmp_path):
        """引擎 add() 直接支持 event_time"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro
        engine = SuMemoryLitePro(
            storage_path=str(tmp_path / "eng"),
            enable_vector=False,
            enable_llm_energy=False,
        )
        event_ts = int(time.time()) - 7 * 86400  # 一周前
        engine.add("历史事件", metadata={"patient_id": "P1"}, event_time=event_ts)
        results = engine.query("历史事件")
        assert len(results) >= 1
        assert results[0]["event_time"] == event_ts
