"""
共享临床适配层测试 — P2-S4 验证

端到端验证 ClinicalMemoryClient 整合所有 Phase 1+2 能力。
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


@pytest.fixture
def client(tmp_path):
    from su_memory.clinical import ClinicalMemoryClient
    return ClinicalMemoryClient(
        storage_path=str(tmp_path / "client_test"),
        embedding_backend="none",
        compliance_level="mask",
    )


class TestBasicOperations:
    """基础操作测试"""

    def test_add_patient_event(self, client):
        """写入患者事件"""
        mid = client.add_patient_event("P001", "高蛋白方案1800kcal", "plan")
        assert mid is not None

    def test_recall_with_isolation(self, client):
        """召回带患者隔离"""
        client.add_patient_event("P001", "P001专属方案", "plan")
        client.add_patient_event("P002", "P002专属方案", "plan")
        hits = client.recall("P001", "方案", top_k=5)
        for h in hits:
            assert h["metadata"]["patient_id"] == "P001"

    def test_health_check(self, client):
        """健康检查返回模块状态"""
        health = client.health_check()
        assert health["modules"]["association"] is True
        assert health["modules"]["compliance"] is True


class TestKnowledgeIntegration:
    """知识库集成测试"""

    def test_check_drug_interaction(self, client):
        """药物交互查询"""
        results = client.check_drug_interaction(["华法林"])
        assert len(results) >= 1
        assert any(r.nutrient == "维生素K" for r in results)

    def test_get_lab_reference(self, client):
        """检验参考值查询"""
        ref = client.get_lab_reference("白蛋白")
        assert ref is not None
        assert ref.unit == "g/L"


class TestLabTrendIntegration:
    """检验趋势集成测试"""

    def test_add_and_trend(self, client):
        """写入检验值 + 趋势查询"""
        for val in [30.0, 32.0, 35.0]:
            client.add_lab_value("P001", "白蛋白", val, "g/L", "35-55")
        trend = client.get_lab_trend("P001", "白蛋白")
        assert trend.direction.value == "up"
        assert trend.count == 3

    def test_find_abnormal(self, client):
        """异常检验筛查"""
        client.add_lab_value("P001", "白蛋白", 28.0, "g/L", "35-55")
        abnormal = client.find_abnormal_labs("P001")
        assert any(a["lab_name"] == "白蛋白" for a in abnormal)


class TestComplianceIntegration:
    """合规集成测试"""

    def test_phi_auto_masked(self, client):
        """PHI 自动脱敏"""
        mid = client.add_patient_event("P001", "入院", "admission", metadata={
            "patient_name": "张三",
            "id_card": "330102199001011234",
        })
        node = client._engine._graph._nodes[mid]
        assert node.metadata["patient_name"] == "张*"

    def test_purge(self, client):
        """删除权"""
        client.add_patient_event("P001", "记忆1", "plan")
        client.add_patient_event("P001", "记忆2", "plan")
        report = client.purge_patient("P001")
        assert report is not None
        assert report.memories_deleted == 2
        hits = client.recall("P001", "记忆")
        assert len(hits) == 0

    def test_audit_trail(self, client):
        """审计追溯"""
        client.add_patient_event("P001", "test", "plan")
        assert client.audit is not None
        entries = client.audit.query(patient_id="P001")
        assert len(entries) >= 1


class TestFeedbackFlywheel:
    """反馈飞轮集成测试"""

    def test_train_from_feedback(self, client):
        """反馈训练"""
        mid = client.add_patient_event("P001", "高蛋白方案", "plan")
        conf = client.train_from_feedback(mid, rating=5, action="accept")
        assert conf is not None
        assert conf > 0.5


class TestPatternMining:
    """模式提炼集成测试"""

    def test_mine_patterns(self, client):
        """模式提炼"""
        for i in range(5):
            client.add_patient_event("P001", f"糖尿病低蛋白方案变体{i}", "plan")
        patterns = client.mine_patterns(min_support=3)
        assert len(patterns) >= 1


class TestFullE2E:
    """端到端全链路测试"""

    def test_full_workflow(self, client):
        """完整工作流：写入→检索→反馈→趋势→删除"""
        # 写入
        mid = client.add_patient_event("P001", "华法林抗凝治疗", "plan",
                                        metadata={"patient_name": "李四"})
        # 检索
        hits = client.recall("P001", "华法林", top_k=5)
        assert len(hits) >= 1
        # 反馈
        client.train_from_feedback(mid, rating=5, action="accept")
        # 检验趋势
        client.add_lab_value("P001", "白蛋白", 30.0, "g/L", "35-55")
        client.add_lab_value("P001", "白蛋白", 38.0, "g/L", "35-55")
        trend = client.get_lab_trend("P001", "白蛋白")
        assert trend.direction.value == "up"
        # 药物交互
        interactions = client.check_drug_interaction(["华法林"])
        assert len(interactions) >= 1
        # 删除
        report = client.purge_patient("P001")
        assert report.memories_deleted >= 1
        # 验证清除
        assert len(client.recall("P001", "华法林")) == 0


class TestRecallPaging:
    """recall 分页倍增拉取测试（P0-2 回归）"""

    def test_multi_patient_sparse_recall(self, client):
        """多患者场景下分页倍增能找到稀疏患者的记忆。

        场景：P001 只有 1 条相关记忆，P002 有大量相似记忆占据检索窗口
        的前排。旧的 top_k*3 固定窗口会把 P001 顶到窗口外；分页倍增会
        扩大窗口直到把 P001 的记忆纳入。
        """
        # P002 写入大量记忆占据检索窗口前排（与查询语义高度相似）
        for i in range(20):
            client.add_patient_event("P002", f"营养方案记录 第{i}次门诊", "plan")

        # P001 只写 1 条，语义上也能命中查询但排在 P002 之后
        client.add_patient_event("P001", "营养方案记录 首次", "plan")

        # recall P001：分页倍增应扩大窗口直到找到 P001 的记忆
        hits = client.recall("P001", "营养方案", top_k=1, max_fetch=500)
        assert len(hits) == 1
        assert hits[0]["metadata"]["patient_id"] == "P001"

    def test_recall_max_fetch_cap(self, client):
        """max_fetch 上限保护，避免无限拉取"""
        # 大量患者各写 1 条，确保超过 max_fetch
        for i in range(10):
            client.add_patient_event(f"P{i:03d}", f"华法林方案 {i}", "plan")

        # top_k=5 但患者不存在时不应无限拉
        hits = client.recall("P999", "华法林", top_k=5, max_fetch=50)
        assert hits == []

    def test_recall_respects_top_k(self, client):
        """返回数量不超过 top_k"""
        for i in range(8):
            client.add_patient_event("P001", f"营养方案记录 {i}", "plan")
        hits = client.recall("P001", "营养方案", top_k=3)
        assert len(hits) <= 3
