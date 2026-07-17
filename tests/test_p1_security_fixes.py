"""
P1 安全修复验证测试

验证 V2/V3/V7/V9 四个中危漏洞已修复。
"""
from __future__ import annotations

import os
import threading
import time
import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


class TestV2NormalizedMatching:
    """V2: 分词归一化匹配（防空格绕过）"""

    def test_spaced_drug_name_detected(self):
        """药名加空格应被检出（归一化后匹配）"""
        from su_memory.clinical import MedicalKnowledgeBase, SafetyGate
        gate = SafetyGate(MedicalKnowledgeBase())
        # "华 法 林" 加空格试图绕过
        screened = gate.screen([{"memory_id": "m1", "content": "华 法 林 抗凝"}])
        assert screened[0]["risk_level"] == "contraindicated"
        assert any("华法林" in f for f in screened[0]["risk_flags"])

    def test_normal_drug_still_detected(self):
        """正常药名仍正确检出"""
        from su_memory.clinical import MedicalKnowledgeBase, SafetyGate
        gate = SafetyGate(MedicalKnowledgeBase())
        screened = gate.screen([{"memory_id": "m1", "content": "华法林抗凝"}])
        assert screened[0]["risk_level"] == "contraindicated"

    def test_safe_text_no_false_positive(self):
        """非药物文本无误报"""
        from su_memory.clinical import MedicalKnowledgeBase, SafetyGate
        gate = SafetyGate(MedicalKnowledgeBase())
        screened = gate.screen([{"memory_id": "m1", "content": "普通营养建议"}])
        assert screened[0]["risk_level"] == "safe"


class TestV3AllergyFromText:
    """V3: 过敏原从自由文本提取（不依赖结构化记忆）"""

    def test_allergy_from_free_text(self, tmp_path):
        """过敏信息在自由文本里也能被检测"""
        from su_memory.clinical import ClinicalMemoryClient
        c = ClinicalMemoryClient(
            storage_path=str(tmp_path / "v3"),
            embedding_backend="none", compliance_level=None,
        )
        # 写一条自由文本过敏记忆（非 event_type=allergy）
        c.add_patient_event("P1", "患者对花生过敏，需避免", "history")
        # 再写含花生物质的方案
        c.add_patient_event("P1", "建议补充花生蛋白粉", "plan")
        hits = c.recall("P1", "花生蛋白")
        # 应检出过敏冲突（V3: 从文本提取了花生过敏原）
        assert any("过敏" in f for h in hits for f in h.get("risk_flags", []))


class TestV7ConcurrentVersionChain:
    """V7: 并发 update_fact 不分叉"""

    def test_concurrent_updates_no_fork(self, tmp_path):
        """并发更新同一 fact_key 不产生双 active"""
        from su_memory.clinical import ClinicalMemoryClient
        c = ClinicalMemoryClient(
            storage_path=str(tmp_path / "v7"),
            embedding_backend="none", compliance_level=None,
        )
        # 先建 v1
        c.update_clinical_fact("P1", "plan", "v1_base")

        # 并发 5 个线程同时 update
        errors: list[Exception] = []
        def worker(n):
            try:
                c.update_clinical_fact("P1", "plan", f"v_concurrent_{n}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # 版本链应连续递增，无分叉
        history = c.get_fact_history("P1", "plan")
        versions = [h["version"] for h in history]
        # 应为 1,2,3,4,5,6 连续（v1 + 5 并发）
        assert versions == sorted(versions), f"版本非单调: {versions}"
        assert len(versions) == 6, f"应6个版本,实{len(versions)}: {versions}"
        # 只有一个 active
        actives = [h for h in history if h["active"]]
        assert len(actives) == 1, f"应1个active,实{len(actives)}"


class TestV9EventTimeSentinel:
    """V9: event_time 合法性校验"""

    def test_negative_event_time_rejected(self, tmp_path):
        """负 event_time 回退到入库时间"""
        from su_memory.clinical import ClinicalMemoryClient
        c = ClinicalMemoryClient(
            storage_path=str(tmp_path / "v9neg"),
            embedding_backend="none", compliance_level=None,
        )
        c.add_patient_event("P1", "负时间事件", "plan", event_time=-1)
        hits = c.recall("P1", "负时间")
        assert len(hits) >= 1
        # 负数被拒，event_time 应回退到入库时间（>0）
        assert hits[0]["event_time"] > 0

    def test_future_event_time_clamped(self, tmp_path):
        """未来 event_time clamp 到 now"""
        from su_memory.clinical import ClinicalMemoryClient
        c = ClinicalMemoryClient(
            storage_path=str(tmp_path / "v9fut"),
            embedding_backend="none", compliance_level=None,
        )
        future = int(time.time()) + 86400 * 365
        c.add_patient_event("P1", "未来事件", "plan", event_time=future)
        hits = c.recall("P1", "未来事件")
        assert len(hits) >= 1
        # 未来时间被 clamp，event_time ≈ 入库时间
        assert hits[0]["event_time"] <= int(time.time()) + 60

    def test_zero_event_time_falls_back(self, tmp_path):
        """event_time=0 回退到入库时间（V9: 0 视为未设置）"""
        from su_memory.clinical import ClinicalMemoryClient
        c = ClinicalMemoryClient(
            storage_path=str(tmp_path / "v9zero"),
            embedding_backend="none", compliance_level=None,
        )
        c.add_patient_event("P1", "零时间事件", "plan", event_time=0)
        hits = c.recall("P1", "零时间")
        assert len(hits) >= 1
        # 0 视为未设置，effective_time 回退 timestamp
        assert hits[0]["event_time"] == hits[0]["timestamp"]
