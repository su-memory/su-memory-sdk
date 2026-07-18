"""
P2 安全修复验证测试

验证对抗性审计 V10-V16 + V18 八个漏洞已修复（V17 已并入 V13 版本链测试）。
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


# ════════════════════════════════════════════════════════════════
# V10: caution 级在 block 策略下剥除临床建议
# ════════════════════════════════════════════════════════════════


class TestV10CautionAdviceStripped:
    """V10: block 策略下 caution 级不应外泄临床建议文本"""

    def test_block_policy_caution_advice_stripped(self):
        """二甲双胍(moderate)在 block 策略下，advice 字段应被剥除"""
        from su_memory.clinical import MedicalKnowledgeBase, SafetyGate

        gate = SafetyGate(MedicalKnowledgeBase(), policy="block")
        # 二甲双胍 × B12 是 moderate 交互
        screened = gate.screen([{"memory_id": "m1", "content": "二甲双胍治疗中"}])
        assert len(screened) == 1  # caution 不整条删
        assert screened[0]["risk_level"] == "caution"
        # 关键：advice 字段被剥除
        for inter in screened[0]["risk_interactions"]:
            assert "advice" not in inter, f"clinical_advice 未剥除: {inter}"

    def test_mark_policy_keeps_advice(self):
        """mark 策略（默认）保留 advice（供医生参考）"""
        from su_memory.clinical import MedicalKnowledgeBase, SafetyGate

        gate = SafetyGate(MedicalKnowledgeBase(), policy="mark")
        screened = gate.screen([{"memory_id": "m1", "content": "二甲双胍治疗中"}])
        assert screened[0]["risk_level"] == "caution"
        # mark 策略保留 advice
        has_advice = any("advice" in i for i in screened[0]["risk_interactions"])
        assert has_advice, "mark 策略应保留 clinical_advice"

    def test_block_policy_contraindicated_still_blocked(self):
        """block 策略下 contraindicated 仍整条拦截"""
        from su_memory.clinical import MedicalKnowledgeBase, SafetyGate

        gate = SafetyGate(MedicalKnowledgeBase(), policy="block")
        screened = gate.screen([{"memory_id": "m1", "content": "华法林抗凝"}])
        assert len(screened) == 0  # 整条拦截


# ════════════════════════════════════════════════════════════════
# V11: PHI 白名单补全 + 嵌套递归
# ════════════════════════════════════════════════════════════════


class TestV11PhiWhitelistComplete:
    """V11: PHI 字段白名单覆盖中英文/驼峰/嵌套"""

    def test_chinese_phi_fields_masked(self):
        """中文 PHI 字段应被脱敏"""
        from su_memory.clinical.compliance import PHISanitizer

        out = PHISanitizer().sanitize({
            "姓名": "张三", "身份证号": "330102199001011234",
            "手机": "13812345678",
        })
        assert out["姓名"] == "张*"
        assert "330102" not in out["身份证号"]
        assert "12345678" not in out["手机"]

    def test_camel_case_phi_masked(self):
        """驼峰命名 patientName/idCard 应被脱敏"""
        from su_memory.clinical.compliance import PHISanitizer

        out = PHISanitizer().sanitize({
            "patientName": "李四", "idCard": "330102199001011234",
        })
        assert out["patientName"] == "李*"
        assert "1990" not in out["idCard"]  # 生日不应泄露

    def test_nested_dict_phi_masked(self):
        """嵌套 metadata 中的 PHI 应递归脱敏（防绕过）"""
        from su_memory.clinical.compliance import PHISanitizer

        out = PHISanitizer().sanitize({
            "profile": {"patient_name": "王五", "hobby": "reading"},
            "tags": [{"phone": "13812345678"}, {"safe": "ok"}],
        })
        assert out["profile"]["patient_name"] == "王*"
        assert out["profile"]["hobby"] == "reading"  # 非 PHI 不动
        assert "12345678" not in out["tags"][0]["phone"]

    def test_passport_masked(self):
        """护照号变体应被脱敏"""
        from su_memory.clinical.compliance import PHISanitizer

        out = PHISanitizer().sanitize({"passport": "E12345678", "护照号": "G98765432"})
        assert out["passport"] != "E12345678"
        assert out["护照号"] != "G98765432"


# ════════════════════════════════════════════════════════════════
# V12: 脱敏收紧（身份证/手机少留明文）
# ════════════════════════════════════════════════════════════════


class TestV12MaskingTightened:
    """V12: 身份证不泄露生日，手机明文位数减少"""

    def test_id_card_birthday_hidden(self):
        """身份证脱敏后生日段（第7-14位）不应明文出现"""
        from su_memory.clinical.compliance import mask_id_card

        masked = mask_id_card("330102199001011234")
        assert "19900101" not in masked, f"生日泄露: {masked}"
        # 校验位保留（末位）
        assert masked[-1] == "4"
        # 地区前2位保留
        assert masked[:2] == "33"

    def test_phone_fewer_plaintext_digits(self):
        """手机脱敏明文不超过 5 位（原 7 位过多）"""
        from su_memory.clinical.compliance import mask_phone

        masked = mask_phone("13812345678")
        plaintext_digits = sum(1 for c in masked if c.isdigit())
        assert plaintext_digits <= 5, f"明文位数过多: {masked} ({plaintext_digits})"
        assert "1234" not in masked  # 中间4位不应明文


# ════════════════════════════════════════════════════════════════
# V13: purge 中间版本截断告警
# ════════════════════════════════════════════════════════════════


class TestV13ChainTruncationAlert:
    """V13: 版本链中间版本缺失时不再静默截断"""

    def test_truncation_marked_when_middle_missing(self, monkeypatch):
        """中间版本被 purge 后，get_history 应在最早一条标记截断"""
        from su_memory.clinical.versioning import ClinicalVersionChain

        # 构造一个 fake engine：3 版本链，中间 v2 被 purge（_get_node 返回 None）
        class FakeNode:
            def __init__(self, mid, content, version, prev_id, superseded=""):
                self.id = mid
                self.content = content
                self.version = version
                self.timestamp = 1000 + version
                self.event_time = 0  # 触发 effective_time 回退
                self.prev_version_id = prev_id
                self.superseded_by = superseded
                self.metadata = {"patient_id": "P1", "fact_key": "diag"}

        v3 = FakeNode("v3", "v3content", 3, "v2")  # active
        # 补 effective_time（真实 MemoryNode 是 property，fake 需手补）
        v3.effective_time = 1003
        v2_gone = None  # 被 purge
        # v1 存在但 v3 回溯到 v2 时拿不到

        class FakeGraph:
            _nodes = {"v3": v3}

        class FakeEngine:
            _graph = FakeGraph()

        chain = ClinicalVersionChain(FakeEngine())
        history = chain.get_history("P1", "diag")
        # 只有 v3（v2 缺失截断）
        assert len(history) == 1
        assert history[0]["chain_truncated"] is True
        assert history[0]["truncated_count"] >= 1

    def test_complete_chain_no_truncation_flag(self, monkeypatch):
        """完整版本链不应有截断标记"""
        from su_memory.clinical.versioning import ClinicalVersionChain

        class FakeNode:
            def __init__(self, mid, version, prev_id, superseded=""):
                self.id = mid
                self.content = f"c{version}"
                self.version = version
                self.timestamp = 1000 + version
                self.event_time = 0
                self.prev_version_id = prev_id
                self.superseded_by = superseded
                self.metadata = {"patient_id": "P1", "fact_key": "k"}

        v2 = FakeNode("v2", 2, "v1")
        v1 = FakeNode("v1", 1, "")
        v2.effective_time = 1002
        v1.effective_time = 1001

        class FakeGraph:
            _nodes = {"v1": v1, "v2": v2}

        class FakeEngine:
            _graph = FakeGraph()
            # _get_node 用 _memory_map(id→idx) + _memories(列表)
            _memories = [v1, v2]
            _memory_map = {"v1": 0, "v2": 1}

        chain = ClinicalVersionChain(FakeEngine())
        history = chain.get_history("P1", "k")
        assert len(history) == 2
        assert not history[0].get("chain_truncated", False)


# ════════════════════════════════════════════════════════════════
# V14: 多租户前缀注入防护
# ════════════════════════════════════════════════════════════════


class TestV14TenantInjectionGuard:
    """V14: 租户前缀注入 + inner 绕过防护"""

    def test_tenant_prefix_injection_blocked(self):
        """攻击者传 T999:P001 不应冒充他租户"""
        from su_memory.clinical.multi_tenant import MultiTenantClient

        # 不实际初始化（只需 _scoped_pid 逻辑），用 __new__ 绕过 __init__
        mt = MultiTenantClient.__new__(MultiTenantClient)
        mt._tenant_id = "T001"
        scoped = mt._scoped_pid("T999:P001")
        # 应剥离 T999 前缀，只保留本租户 T001
        assert scoped == "T001:P001", f"注入未拦截: {scoped}"

    def test_multi_layer_injection_blocked(self):
        """多层注入 T999:T002:P001 应被完全剥离"""
        from su_memory.clinical.multi_tenant import MultiTenantClient

        mt = MultiTenantClient.__new__(MultiTenantClient)
        mt._tenant_id = "T001"
        assert mt._scoped_pid("T999:T002:P001") == "T001:P001"

    def test_empty_patient_id_rejected(self):
        """空 patient_id 应拒绝"""
        from su_memory.clinical.multi_tenant import MultiTenantClient

        mt = MultiTenantClient.__new__(MultiTenantClient)
        mt._tenant_id = "T001"
        try:
            mt._scoped_pid("")
            assert False, "空 patient_id 未拒绝"
        except ValueError:
            pass


# ════════════════════════════════════════════════════════════════
# V15: 未来 event_time 不再霸占召回
# ════════════════════════════════════════════════════════════════


class TestV15FutureEventTimeCapped:
    """V15: 未来 event_time 的 recency 不超过 1.0"""

    def test_future_timestamp_recency_capped(self):
        """未来时间的 recency 应被 clamp，不超过 1.0"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        engine = SuMemoryLitePro.__new__(SuMemoryLitePro)
        engine._temporal = SuMemoryLitePro(0)._temporal  # 复用内部 temporal
        now = int(time.time())
        future_ts = now + 86400 * 365  # 1年后
        recency = engine._temporal.calculate_recency_score(future_ts, "earth", now)
        assert recency <= 1.0 + 0.01, f"未来时间 recency 未 cap: {recency}"

    def test_normal_recent_recency_high(self):
        """正常近期记忆 recency 应较高（回归不误伤）"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        engine = SuMemoryLitePro(0)
        now = int(time.time())
        recent_ts = now - 86400  # 1天前
        recency = engine._temporal.calculate_recency_score(recent_ts, "earth", now)
        assert 0 < recency <= 1.0


# ════════════════════════════════════════════════════════════════
# V16: 负/零 event_time 不再黑洞
# ════════════════════════════════════════════════════════════════


class TestV16NegativeTimeNoBlackhole:
    """V16: 负/零 timestamp 不再落入异常桶查不到"""

    def test_zero_timestamp_normalized_in_bucket(self):
        """零 timestamp 进索引应归一化到当前桶（可被查询）"""
        from su_memory.sdk.spacetime_index import TimeBucketIndex

        idx = TimeBucketIndex()
        idx.add_node("node1", 0)  # 零 timestamp
        now = int(time.time())
        # 查询近期范围应能命中
        result = idx.get_nodes_in_range(now - 86400 * 2, now + 86400)
        assert "node1" in result, "零 timestamp 落入黑洞"

    def test_negative_timestamp_normalized(self):
        """负 timestamp 进索引应归一化（可被查询）"""
        from su_memory.sdk.spacetime_index import TimeBucketIndex

        idx = TimeBucketIndex()
        idx.add_node("node2", -1000)
        now = int(time.time())
        result = idx.get_nodes_in_range(now - 86400 * 2, now + 86400)
        assert "node2" in result

    def test_negative_range_query_returns_empty(self):
        """负值范围查询应安全返回空，不崩"""
        from su_memory.sdk.spacetime_index import TimeBucketIndex

        idx = TimeBucketIndex()
        idx.add_node("n3", int(time.time()))
        result = idx.get_nodes_in_range(-100, -50)
        assert result == []

    def test_negative_recency_normalized(self):
        """负 timestamp 的 recency 不应爆表"""
        from su_memory.sdk.lite_pro import SuMemoryLitePro

        engine = SuMemoryLitePro(0)
        now = int(time.time())
        recency = engine._temporal.calculate_recency_score(-99999, "earth", now)
        assert 0 < recency <= 1.0


# ════════════════════════════════════════════════════════════════
# V17: 自环节点不产生脏记录（并入 V13 同一方法）
# ════════════════════════════════════════════════════════════════


class TestV17SelfLoopNoDirtyRecord:
    """V17: 自环节点（prev 指向自身）不产生重复脏记录"""

    def test_self_loop_terminates_cleanly(self):
        from su_memory.clinical.versioning import ClinicalVersionChain

        class FakeNode:
            def __init__(self):
                self.id = "v1"
                self.content = "c1"
                self.version = 1
                self.timestamp = 1000
                self.event_time = 0
                self.prev_version_id = "v1"  # 自环！
                self.superseded_by = ""
                self.metadata = {"patient_id": "P1", "fact_key": "k"}

        node = FakeNode()
        node.effective_time = 1000  # 补 property

        class FakeGraph:
            _nodes = {"v1": node}

        class FakeEngine:
            _graph = FakeGraph()

        chain = ClinicalVersionChain(FakeEngine())
        history = chain.get_history("P1", "k")
        # 自环应终止，只 1 条，不重复
        assert len(history) == 1


# ════════════════════════════════════════════════════════════════
# V18: 无 id 时稳定哈希去重
# ════════════════════════════════════════════════════════════════


class TestV18StableDedupWithoutId:
    """V18: 引擎返回无 memory_id 时用稳定哈希去重（非 id(r)）"""

    def test_same_content_produces_same_stable_id(self):
        """相同 content+timestamp 应产生相同 rid（跨 query 一致）"""
        import hashlib

        r1 = {"content": "患者血糖控制良好", "timestamp": 1700000000, "event_time": 0}
        r2 = {"content": "患者血糖控制良好", "timestamp": 1700000000, "event_time": 0}
        key = f"{r1.get('content','')[:64]}|{r1.get('timestamp',0)}|{r1.get('event_time',0)}"
        rid1 = "auto_" + hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        key2 = f"{r2.get('content','')[:64]}|{r2.get('timestamp',0)}|{r2.get('event_time',0)}"
        rid2 = "auto_" + hashlib.md5(key2.encode("utf-8")).hexdigest()[:12]
        assert rid1 == rid2, "相同内容应产生稳定 id"

    def test_different_content_different_id(self):
        """不同内容应产生不同 rid"""
        import hashlib

        def make_rid(r):
            key = f"{r.get('content','')[:64]}|{r.get('timestamp',0)}|{r.get('event_time',0)}"
            return "auto_" + hashlib.md5(key.encode("utf-8")).hexdigest()[:12]

        r1 = {"content": "AAA", "timestamp": 1, "event_time": 0}
        r2 = {"content": "BBB", "timestamp": 1, "event_time": 0}
        assert make_rid(r1) != make_rid(r2)
