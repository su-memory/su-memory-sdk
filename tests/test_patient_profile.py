"""
患者纵向记忆增强测试 — P2-S1 验证

检验值趋势、异常筛查、诊疗轨迹、就诊对比。
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


@pytest.fixture
def space(tmp_path):
    from su_memory.sdk.lite_pro import SuMemoryLitePro
    from su_memory.clinical import PatientMemorySpace
    client = SuMemoryLitePro(
        storage_path=str(tmp_path / "profile_test"),
        embedding_backend="none",
        enable_llm_energy=False,
    )
    return PatientMemorySpace(client)


class TestLabTrend:
    """检验值趋势测试"""

    def test_upward_trend(self, space):
        """白蛋白上升趋势"""
        for val in [30.0, 32.0, 35.0, 38.0]:
            space.add_lab_value("P001", "白蛋白", val, "g/L", "35-55")
        trend = space.get_lab_trend("P001", "白蛋白")
        assert trend.direction.value == "up"
        assert trend.count == 4
        assert trend.change_pct > 0

    def test_downward_trend(self, space):
        """白蛋白下降趋势"""
        for val in [40.0, 38.0, 35.0, 30.0]:
            space.add_lab_value("P001", "白蛋白", val, "g/L", "35-55")
        trend = space.get_lab_trend("P001", "白蛋白")
        assert trend.direction.value == "down"

    def test_stable_trend(self, space):
        """稳定趋势"""
        for val in [35.0, 35.5, 36.0]:
            space.add_lab_value("P001", "白蛋白", val, "g/L", "35-55")
        trend = space.get_lab_trend("P001", "白蛋白")
        assert trend.direction.value == "stable"

    def test_insufficient_data(self, space):
        """数据不足"""
        space.add_lab_value("P001", "白蛋白", 35.0, "g/L", "35-55")
        trend = space.get_lab_trend("P001", "白蛋白")
        assert trend.direction.value == "insufficient"

    def test_patient_isolation(self, space):
        """不同患者的检验值不串"""
        space.add_lab_value("P001", "白蛋白", 30.0, "g/L", "35-55")
        space.add_lab_value("P002", "白蛋白", 50.0, "g/L", "35-55")
        t1 = space.get_lab_trend("P001", "白蛋白")
        t2 = space.get_lab_trend("P002", "白蛋白")
        assert t1.first_value == 30.0
        assert t2.first_value == 50.0


class TestAbnormalLabs:
    """异常检验筛查测试"""

    def test_find_abnormal(self, space):
        """找出异常检验值"""
        space.add_lab_value("P001", "白蛋白", 28.0, "g/L", "35-55")  # 低
        space.add_lab_value("P001", "白蛋白", 40.0, "g/L", "35-55")  # 正常
        space.add_lab_value("P001", "钾", 6.5, "mmol/L", "3.5-5.5")  # 高
        abnormal = space.find_abnormal_labs("P001")
        names = [a["lab_name"] for a in abnormal]
        assert "白蛋白" in names
        assert "钾" in names

    def test_no_abnormal(self, space):
        """全部正常时返回空"""
        space.add_lab_value("P001", "白蛋白", 40.0, "g/L", "35-55")
        abnormal = space.find_abnormal_labs("P001")
        assert len(abnormal) == 0


class TestCareTrajectory:
    """诊疗轨迹测试"""

    def test_trajectory_ordered(self, space):
        """轨迹按时间排序"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro
        client = space._client
        client.add("入院评估", metadata={"patient_id": "P001", "event_type": "admission"})
        client.add("营养筛查", metadata={"patient_id": "P001", "event_type": "screening"})
        client.add("制定方案", metadata={"patient_id": "P001", "event_type": "plan"})

        traj = space.get_care_trajectory("P001")
        assert len(traj) == 3
        assert traj[0]["event_type"] == "admission"
        assert traj[-1]["event_type"] == "plan"

    def test_trajectory_patient_isolated(self, space):
        """轨迹只返回指定患者"""
        client = space._client
        client.add("P001事件", metadata={"patient_id": "P001", "event_type": "plan"})
        client.add("P002事件", metadata={"patient_id": "P002", "event_type": "plan"})
        traj = space.get_care_trajectory("P001")
        assert len(traj) == 1
        assert traj[0]["content"] == "P001事件"


class TestComparePeriods:
    """就诊对比测试"""

    def test_compare_two_periods(self, space):
        """对比两个时间段"""
        import time
        client = space._client
        now = time.time()

        # period_a：1个方案事件
        client.add("方案A", metadata={
            "patient_id": "P001", "event_type": "plan",
            "timestamp": now - 100,
        })
        # period_b：2个随访事件
        client.add("随访1", metadata={
            "patient_id": "P001", "event_type": "followup",
            "timestamp": now - 10,
        })
        client.add("随访2", metadata={
            "patient_id": "P001", "event_type": "followup",
            "timestamp": now - 5,
        })

        result = space.compare_periods(
            "P001",
            (now - 200, now - 50),  # period_a
            (now - 50, now + 10),   # period_b
        )
        assert "changes" in result
        assert result["period_b"]["event_count"] >= result["period_a"]["event_count"]
