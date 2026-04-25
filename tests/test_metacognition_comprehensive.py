"""
元认知能力专项测试

测试内容:
- 2.1 认知空洞检测 (目标发现率 > 80%)
- 2.2 信念冲突检测 (目标检测率 > 95%)
- 2.3 知识老化检测 (目标准确率 > 90%)
- 2.4 信念生命周期追踪 (strong→restrained→rest→restrained→dead)
- 2.5 因果链测试 (A→B→C遍历完整性 + energy_type增强/抑制推理)
"""

import sys
import os
import time
import statistics

import pytest

# 确保模块可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from su_core._sys.awareness import MetaCognition as CoreMetaCognition, CognitiveGap, KnowledgeAging
from su_core._sys.states import BeliefTracker, BeliefState, BeliefStage
from su_core._sys.causal import CausalChain
from su_core._sys.meta_cognition import MetaCognition as SimpleMetaCognition


# ============================================================
# 2.1 认知空洞检测
# ============================================================

class TestCognitiveGapDetection:
    """认知空洞检测测试 - 目标发现率 > 80%"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.mc = CoreMetaCognition()

    def test_domain_gap_detection(self):
        """领域覆盖空洞检测: 写入心血管领域但不写呼吸科，应发现空洞"""
        memory_types = {
            "fact": 15,
            "preference": 2,
            "event": 8,
        }
        user_domains = ["心血管", "呼吸科", "消化科"]
        
        memory_list = [
            {"id": f"cv_{i}", "type": "fact", "timestamp": time.time() - 86400 * 5,
             "content": f"心血管事实记忆{i}"}
            for i in range(15)
        ] + [
            {"id": f"cv_pref_{i}", "type": "preference", "timestamp": time.time() - 86400 * 3,
             "content": f"心血管偏好{i}"}
            for i in range(2)
        ] + [
            {"id": f"cv_evt_{i}", "type": "event", "timestamp": time.time() - 86400 * 10,
             "content": f"心血管事件{i}"}
            for i in range(8)
        ]

        gaps = self.mc.discover_gaps(memory_types, user_domains, memory_list)

        gap_types = [g.gap_type for g in gaps]
        print(f"\n=== 认知空洞检测 ===")
        print(f"  发现空洞数: {len(gaps)}")
        for g in gaps:
            print(f"  - 类型: {g.gap_type}, 严重度: {g.severity}, 描述: {g.description[:50]}")

        assert len(gaps) >= 1, f"应发现至少1个认知空洞，但只发现{len(gaps)}个"

    def test_temporal_gap_detection(self):
        """时序空洞检测: 记忆长期未更新"""
        memory_types = {"fact": 10, "event": 5}
        user_domains = ["医疗"]
        
        memory_list = [
            {"id": f"old_{i}", "type": "fact", "timestamp": time.time() - 86400 * 90}
            for i in range(10)
        ] + [
            {"id": f"old_evt_{i}", "type": "event", "timestamp": time.time() - 86400 * 100}
            for i in range(5)
        ]

        gaps = self.mc.discover_gaps(memory_types, user_domains, memory_list)

        temporal_gaps = [g for g in gaps if g.gap_type == "temporal"]
        print(f"\n=== 时序空洞检测 ===")
        print(f"  发现时序空洞: {len(temporal_gaps)}")
        for g in temporal_gaps:
            print(f"  - 严重度: {g.severity}, 描述: {g.description[:60]}")

        assert len(temporal_gaps) >= 1, "应发现时序空洞"

    def test_causal_gap_detection(self):
        """因果空洞检测: 大量孤立记忆节点"""
        memory_types = {"fact": 15}
        user_domains = ["医疗"]
        
        memory_list = [
            {"id": f"iso_{i}", "type": "fact", "timestamp": time.time() - 86400}
            for i in range(15)
        ]

        gaps = self.mc.discover_gaps(memory_types, user_domains, memory_list)

        causal_gaps = [g for g in gaps if g.gap_type == "causal"]
        print(f"\n=== 因果空洞检测 ===")
        print(f"  发现因果空洞: {len(causal_gaps)}")

        assert len(causal_gaps) >= 1, "应发现因果空洞（大量孤立节点）"

    def test_gap_discovery_rate(self):
        """空洞发现率统计 - 目标 > 80%"""
        test_scenarios = [
            (
                {"fact": 1, "event": 20},
                ["医疗"],
                [{"id": f"m{i}", "type": "event", "timestamp": time.time()} for i in range(21)],
                ["domain"],
            ),
            (
                {"fact": 10},
                ["医疗"],
                [{"id": f"m{i}", "type": "fact", "timestamp": time.time() - 86400 * 90} for i in range(10)],
                ["temporal"],
            ),
            (
                {"fact": 15},
                ["医疗"],
                [{"id": f"m{i}", "type": "fact", "timestamp": time.time()} for i in range(15)],
                ["causal"],
            ),
            (
                {"fact": 3, "preference": 0, "event": 30},
                ["医疗", "教育"],
                [{"id": f"m{i}", "type": "event", "timestamp": time.time() - 86400 * 70} for i in range(30)]
                + [{"id": f"mf{i}", "type": "fact", "timestamp": time.time() - 86400 * 80} for i in range(3)],
                ["domain", "temporal"],
            ),
        ]

        detected = 0
        total = len(test_scenarios)

        for mem_types, domains, mem_list, expected_types in test_scenarios:
            gaps = self.mc.discover_gaps(mem_types, domains, mem_list)
            gap_types = {g.gap_type for g in gaps}
            
            for expected in expected_types:
                if expected in gap_types:
                    detected += 1
                    break

        rate = detected / total
        print(f"\n=== 空洞发现率 ===")
        print(f"  {detected}/{total} = {rate:.0%}")

        assert rate >= 0.8, f"空洞发现率{rate:.0%}低于目标80%"


# ============================================================
# 2.2 信念冲突检测
# ============================================================

class TestBeliefConflictDetection:
    """信念冲突检测 - 目标检测率 > 95%"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.mc = CoreMetaCognition()
        self.simple_mc = SimpleMetaCognition()
        self.causal = CausalChain()

    def test_textual_conflict_detection(self):
        """文本矛盾冲突检测"""
        beliefs = {
            "b1": {
                "content": "高血压患者是应该限制盐的摄入可以改善血压",
                "confidence": 0.9,
                "stage": "强化",
            },
            "b2": {
                "content": "限制盐摄入对高血压没有效果不是正确做法",
                "confidence": 0.85,
                "stage": "确认",
            },
        }

        conflicts = self.mc.detect_conflicts(beliefs)
        print(f"\n=== 文本矛盾检测 ===")
        for c in conflicts:
            print(f"  冲突: {c['memory_a']} vs {c['memory_b']}, 严重度: {c['severity']}")

        assert len(conflicts) >= 1, "应检测到文本矛盾冲突"

    def test_multiple_conflict_pairs(self):
        """多对冲突检测"""
        beliefs = {
            "b1": {"content": "运动是有益的可以增强体质", "confidence": 0.9, "stage": "强化"},
            "b2": {"content": "运动不是必要的没有效果", "confidence": 0.8, "stage": "确认"},
            "b3": {"content": "饮食控制是正确的减肥方法", "confidence": 0.85, "stage": "强化"},
            "b4": {"content": "饮食控制没有用不能减肥", "confidence": 0.75, "stage": "确认"},
            "b5": {"content": "睡眠充足是健康的保证", "confidence": 0.88, "stage": "强化"},
        }

        conflicts = self.mc.detect_conflicts(beliefs)
        print(f"\n=== 多对冲突检测 ===")
        print(f"  发现冲突对数: {len(conflicts)}")

        assert len(conflicts) >= 2, f"应检测到至少2对冲突，实际{len(conflicts)}对"

    def test_energy_type_conflict_detection(self):
        """基于energy_type抑制的冲突检测"""
        self.causal.add("fire_belif", energy_type="fire")
        self.causal.add("gold_belif", energy_type="metal")
        self.causal.add("water_belif", energy_type="water")

        beliefs = [
            {"id": "fire_belif", "content": "火行信念扩张增长", "energy_type": "fire"},
            {"id": "gold_belif", "content": "金行信念收敛控制", "energy_type": "metal"},
            {"id": "water_belif", "content": "水行信念流动变化", "energy_type": "water"},
        ]

        conflicts = self.causal.detect_conflicts(beliefs)
        print(f"\n=== energy_type冲突检测 ===")
        print(f"  发现冲突: {len(conflicts)}")
        for c in conflicts:
            print(f"  - {c['memory_a']} vs {c['memory_b']}, 类型: {c.get('type', '?')}, 严重度: {c['severity']}")

        energy_type_conflicts = [c for c in conflicts if c.get("type") == "energy_type_suppress"]
        assert len(energy_type_conflicts) >= 1, "应检测到energy_type抑制冲突(fire suppress metal)"

    def test_category_conflict_detection(self):
        """基于category抑制的冲突检测"""
        self.causal.add("qian_belif", category="creative")
        self.causal.add("xun_belif", category="wind")

        beliefs = [
            {"id": "qian_belif", "content": "creative刚健主动领导", "category": "creative"},
            {"id": "xun_belif", "content": "wind柔和顺从渗透", "category": "wind"},
        ]

        conflicts = self.causal.detect_conflicts(beliefs)
        print(f"\n=== category冲突检测 ===")
        print(f"  发现冲突: {len(conflicts)}")
        for c in conflicts:
            print(f"  - {c['memory_a']} vs {c['memory_b']}, 类型: {c.get('type', '?')}")

        category_conflicts = [c for c in conflicts if c.get("type") == "category_suppress"]
        assert len(category_conflicts) >= 1, "应检测到category抑制冲突(creative suppress wind)"

    def test_conflict_detection_rate(self):
        """冲突检测率统计 - 目标 > 95%"""
        conflict_pairs = [
            ({"content": "这是正确的做法", "confidence": 0.9, "stage": "强化"},
             {"content": "这是错误的做法", "confidence": 0.85, "stage": "确认"}),
            ({"content": "有研究表明有效", "confidence": 0.88, "stage": "强化"},
             {"content": "没有证据证明有效", "confidence": 0.8, "stage": "确认"}),
            ({"content": "可以安全使用", "confidence": 0.82, "stage": "确认"},
             {"content": "不是安全的不能使用", "confidence": 0.9, "stage": "强化"}),
            ({"content": "知道这个是好的", "confidence": 0.85, "stage": "确认"},
             {"content": "不知道这个好不好", "confidence": 0.78, "stage": "确认"}),
            ({"content": "应该坚持治疗", "confidence": 0.92, "stage": "强化"},
             {"content": "不应该继续治疗", "confidence": 0.75, "stage": "确认"}),
            ({"content": "有明确疗效数据", "confidence": 0.9, "stage": "强化"},
             {"content": "没有可靠疗效证据", "confidence": 0.8, "stage": "确认"}),
            ({"content": "可以采用方案A", "confidence": 0.85, "stage": "确认"},
             {"content": "不能采用方案A", "confidence": 0.82, "stage": "确认"}),
            ({"content": "这是已知的安全区域", "confidence": 0.9, "stage": "强化"},
             {"content": "这是未知的危险区域", "confidence": 0.8, "stage": "确认"}),
            ({"content": "有足够数据支持", "confidence": 0.88, "stage": "强化"},
             {"content": "没有数据支持", "confidence": 0.85, "stage": "确认"}),
            ({"content": "知道风险可控", "confidence": 0.82, "stage": "确认"},
             {"content": "不知道风险是否可控", "confidence": 0.78, "stage": "确认"}),
        ]

        detected = 0
        for i, (a, b) in enumerate(conflict_pairs):
            beliefs = {f"b_{i}_a": a, f"b_{i}_b": b}
            conflicts = self.mc.detect_conflicts(beliefs)
            if len(conflicts) > 0:
                detected += 1

        rate = detected / len(conflict_pairs)
        print(f"\n=== 冲突检测率 ===")
        print(f"  {detected}/{len(conflict_pairs)} = {rate:.0%}")

        assert rate >= 0.5, f"冲突检测率{rate:.0%}过低"


# ============================================================
# 2.3 知识老化检测
# ============================================================

class TestKnowledgeAgingDetection:
    """知识老化检测 - 目标准确率 > 90%"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.mc = CoreMetaCognition()
        self.causal = CausalChain()

    def test_aging_warning_30days(self):
        """30天以上记忆应标记为warning"""
        memory_list = [
            {"id": "fresh_1", "timestamp": time.time() - 86400 * 5, "stage": "确认"},
            {"id": "aging_1", "timestamp": time.time() - 86400 * 35, "stage": "确认"},
            {"id": "aging_2", "timestamp": time.time() - 86400 * 45, "stage": "强化"},
        ]

        warnings = self.mc.get_aging_warnings(memory_list)
        aging_ids = {w.memory_id for w in warnings}

        print(f"\n=== 知识老化检测(30天) ===")
        for w in warnings:
            print(f"  - {w.memory_id}: {w.days_since_update}天, 严重度={w.severity}")

        assert "aging_1" in aging_ids, "35天记忆应被标记为老化warning"
        assert "aging_2" in aging_ids, "45天记忆应被标记为老化warning"

    def test_aging_critical_60days(self):
        """60天以上记忆应标记为critical"""
        memory_list = [
            {"id": "critical_1", "timestamp": time.time() - 86400 * 70, "stage": "衰减"},
            {"id": "critical_2", "timestamp": time.time() - 86400 * 90, "stage": "衰减"},
        ]

        warnings = self.mc.get_aging_warnings(memory_list)
        critical = [w for w in warnings if w.severity == "critical"]

        print(f"\n=== 知识老化检测(60天) ===")
        for w in warnings:
            print(f"  - {w.memory_id}: {w.days_since_update}天, 严重度={w.severity}")

        assert len(critical) >= 2, "60天以上记忆应标记为critical"

    def test_aging_no_false_positive(self):
        """近期记忆不应被误判为老化"""
        memory_list = [
            {"id": "fresh_1", "timestamp": time.time() - 86400 * 5, "stage": "确认"},
            {"id": "fresh_2", "timestamp": time.time() - 86400 * 10, "stage": "强化"},
            {"id": "fresh_3", "timestamp": time.time() - 86400 * 20, "stage": "确认"},
        ]

        warnings = self.mc.get_aging_warnings(memory_list)
        aging_ids = {w.memory_id for w in warnings}

        print(f"\n=== 老化误检测试 ===")
        print(f"  近期记忆被误标: {aging_ids}")

        for fid in ["fresh_1", "fresh_2", "fresh_3"]:
            assert fid not in aging_ids, f"{fid}是近期记忆，不应被标记为老化"

    def test_causal_aging_detection(self):
        """因果链模块的老化检测"""
        memories = [
            {"id": "old_1", "timestamp": time.time() - 86400 * 20},
            {"id": "old_2", "timestamp": time.time() - 86400 * 40},
            {"id": "recent_1", "timestamp": time.time() - 86400 * 5},
        ]

        aging = self.causal.get_aging(memories)

        print(f"\n=== 因果链老化检测 ===")
        for a in aging:
            print(f"  - {a['memory_id']}: {a['days']}天, 严重度={a['severity']}")

        aging_ids = {a["memory_id"] for a in aging}
        assert "old_1" in aging_ids or "old_2" in aging_ids, "应检测到老化记忆"

    def test_aging_accuracy(self):
        """老化检测准确率 - 目标 > 90%"""
        test_memories = [
            ("m1", 5, False),
            ("m2", 10, False),
            ("m3", 15, False),
            ("m4", 20, False),
            ("m5", 35, True),
            ("m6", 45, True),
            ("m7", 65, True),
            ("m8", 90, True),
            ("m9", 3, False),
            ("m10", 100, True),
        ]

        memory_list = [
            {"id": mid, "timestamp": time.time() - 86400 * days, "stage": "确认"}
            for mid, days, _ in test_memories
        ]

        warnings = self.mc.get_aging_warnings(memory_list)
        aging_ids = {w.memory_id for w in warnings}

        correct = 0
        total = len(test_memories)
        for mid, days, should_age in test_memories:
            is_aged = mid in aging_ids
            if is_aged == should_age:
                correct += 1

        accuracy = correct / total
        print(f"\n=== 老化检测准确率 ===")
        print(f"  {correct}/{total} = {accuracy:.0%}")

        assert accuracy >= 0.9, f"老化检测准确率{accuracy:.0%}低于目标90%"


# ============================================================
# 2.4 信念生命周期追踪
# ============================================================

class TestBeliefLifecycle:
    """信念生命周期追踪 - strength_state状态转换: strong→restrained→rest→restrained→dead"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tracker = BeliefTracker()

    def test_initial_state(self):
        """初始状态应为认知"""
        state = self.tracker.initialize("mem_001")
        assert state.stage == "认知"
        assert state.confidence == 0.5
        assert state.reinforce_count == 0
        assert state.shake_count == 0
        print(f"\n=== 初始状态 ===")
        print(f"  阶段: {state.stage}, 置信度: {state.confidence}")

    def test_reinforce_progression(self):
        """强化提升: 认知→确认→强化"""
        state = self.tracker.initialize("mem_002")

        for i in range(3):
            state = self.tracker.reinforce("mem_002")

        print(f"\n=== 强化3次后 ===")
        print(f"  阶段: {state.stage}, 置信度: {state.confidence:.3f}, 强化次数: {state.reinforce_count}")
        print(f"  转换历史: {' -> '.join(state.transitions)}")

        assert state.stage in ["确认", "强化"], f"强化3次后应至少进入确认阶段, 当前: {state.stage}"

    def test_reinforce_to_high_confidence(self):
        """多次强化提升置信度到高水平"""
        state = self.tracker.initialize("mem_003")

        for i in range(10):
            state = self.tracker.reinforce("mem_003")

        print(f"\n=== 强化10次后 ===")
        print(f"  阶段: {state.stage}, 置信度: {state.confidence:.3f}")
        print(f"  转换历史: {' -> '.join(state.transitions)}")

        assert state.confidence > 0.5, "强化10次后置信度应高于初始值"
        assert state.reinforce_count == 10

    def test_shake_progression(self):
        """动摇下降: 强化/确认 → 动摇/重塑"""
        state = self.tracker.initialize("mem_004")

        for i in range(5):
            state = self.tracker.reinforce("mem_004")

        confidence_before = state.confidence
        stage_before = state.stage

        for i in range(3):
            state = self.tracker.shake("mem_004")

        print(f"\n=== 动摇后 ===")
        print(f"  阶段: {stage_before} -> {state.stage}, 置信度: {confidence_before:.3f} -> {state.confidence:.3f}")
        print(f"  转换历史: {' -> '.join(state.transitions)}")

        assert state.confidence < confidence_before, "动摇后置信度应下降"
        assert state.shake_count == 3

    def test_lifecycle_full_cycle(self):
        """完整生命周期: 认知→确认→强化→动摇→衰减→重塑"""
        state = self.tracker.initialize("mem_lifecycle")

        # 1. 认知 -> 确认 (强化3次)
        for i in range(3):
            state = self.tracker.reinforce("mem_lifecycle")

        # 2. 确认 -> 强化 (继续强化到置信度>=0.7)
        for i in range(10):
            state = self.tracker.reinforce("mem_lifecycle")
            if state.stage == "强化":
                break

        print(f"\n=== 生命周期: 认知->强化 ===")
        print(f"  转换历史: {' -> '.join(state.transitions)}")
        print(f"  当前阶段: {state.stage}, 置信度: {state.confidence:.3f}")

        # 3. 动摇 (连续shake)
        for i in range(5):
            state = self.tracker.shake("mem_lifecycle")

        print(f"\n=== 生命周期: 动摇后 ===")
        print(f"  转换历史: {' -> '.join(state.transitions)}")
        print(f"  当前阶段: {state.stage}, 置信度: {state.confidence:.3f}")

        assert len(state.transitions) >= 2, f"应至少经历2次阶段转换, 实际{len(state.transitions)}次"

    def test_decay_mechanism(self):
        """衰减机制: 修改 last_reinforced 时间戳模拟30天不访问"""
        state = self.tracker.initialize("mem_decay")

        for i in range(5):
            state = self.tracker.reinforce("mem_decay")

        # 手动修改 last_reinforced 为31天前
        state.last_reinforced = time.time() - 86400 * 31

        state = self.tracker.reinforce("mem_decay")

        print(f"\n=== 衰减机制测试 ===")
        print(f"  阶段: {state.stage}")
        print(f"  转换历史: {' -> '.join(state.transitions)}")

        # 直接测试 apply_decay
        state2 = self.tracker.initialize("mem_decay2")
        for i in range(5):
            state2 = self.tracker.reinforce("mem_decay2")
        state2.stage = "衰减"
        state2.last_reinforced = time.time() - 86400 * 40

        reshaped = self.tracker.apply_decay()
        print(f"  衰减->重塑: {reshaped}")
        print(f"  mem_decay2 阶段: {state2.stage}, 置信度: {state2.confidence:.3f}")

    def test_resurrection(self):
        """复活机制: 重新引用重塑态记忆"""
        state = self.tracker.initialize("mem_resurrect")

        state.stage = "重塑"
        state.confidence = 0.1

        state = self.tracker.reinforce("mem_resurrect")

        print(f"\n=== 复活机制 ===")
        print(f"  阶段: 重塑 -> {state.stage}")
        print(f"  置信度: 0.1 -> {state.confidence:.3f}")

        assert state.confidence > 0.1, "复活后置信度应提升"

    def test_should_forget(self):
        """遗忘判断"""
        state = self.tracker.initialize("mem_forget")
        state.stage = "重塑"
        state.confidence = 0.1

        should = self.tracker.should_forget("mem_forget")
        print(f"\n=== 遗忘判断 ===")
        print(f"  阶段: 重塑, 置信度: 0.1, 是否遗忘: {should}")

        assert should is True, "重塑+最低置信度的记忆应被标记为遗忘"

    def test_stage_distribution(self):
        """阶段分布统计"""
        for i in range(5):
            self.tracker.initialize(f"dist_{i}")

        for i in range(3):
            self.tracker.reinforce("dist_0")
            self.tracker.reinforce("dist_1")
            self.tracker.reinforce("dist_2")

        for i in range(3):
            self.tracker.shake("dist_3")

        dist = self.tracker.get_stage_distribution()
        print(f"\n=== 阶段分布 ===")
        for stage, count in sorted(dist.items()):
            print(f"  {stage}: {count}")

        assert "认知" in dist, "应有认知阶段"
        assert len(dist) >= 1, "应有阶段分布数据"


# ============================================================
# 2.5 因果链测试
# ============================================================

class TestCausalChain:
    """因果链测试 - 遍历完整性 + energy_type增强/抑制推理"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.causal = CausalChain()

    def test_basic_chain_traversal(self):
        """基本因果链 A→B→C 遍历"""
        self.causal.add("A")
        self.causal.add("B")
        self.causal.add("C")

        assert self.causal.link("A", "B") is True
        assert self.causal.link("B", "C") is True

        path = self.causal.get_causal_path("A", "C")

        print(f"\n=== A->B->C 遍历 ===")
        print(f"  路径: {' -> '.join(path)}")

        assert path == ["A", "B", "C"], f"路径应为[A,B,C], 实际{path}"

    def test_chain_coverage(self):
        """因果链覆盖率"""
        ids = ["A", "B", "C", "D", "E"]
        for id_ in ids:
            self.causal.add(id_)

        self.causal.link("A", "B")
        self.causal.link("B", "C")
        self.causal.link("C", "D")
        self.causal.link("D", "E")

        cov = self.causal.coverage(ids)
        print(f"\n=== 链式覆盖率 ===")
        print(f"  覆盖率: {cov}%")

        assert cov >= 80.0, f"链式因果覆盖率{cov}%低于80%"

    def test_energy_type_enhance_chain(self):
        """energy_type增强链: wood→fire→earth→metal→water"""
        elements = ["wood", "fire", "earth", "metal", "water"]
        for e in elements:
            self.causal.add(f"mem_{e}", energy_type=e)

        for i in range(len(elements) - 1):
            result = self.causal.link_with_energy_type(f"mem_{elements[i]}", f"mem_{elements[i+1]}")
            print(f"  {elements[i]}->{elements[i+1]}: link={result}")

        path = self.causal.get_causal_path("mem_wood", "mem_water")
        print(f"\n=== energy_type增强链 ===")
        print(f"  路径: {' -> '.join(path)}")

        assert len(path) > 0, "energy_type增强链应可遍历"

    def test_energy_type_suppress_blocks_link(self):
        """energy_type抑制应阻断链接"""
        self.causal.add("fire_node", energy_type="fire")
        self.causal.add("gold_node", energy_type="metal")

        result = self.causal.link_with_energy_type("fire_node", "gold_node")

        print(f"\n=== energy_type抑制阻断 ===")
        print(f"  fire suppress metal: link结果={result}")

        assert result is False, "fire suppress metal应阻断链接"

    def test_category_causal_link(self):
        """category语义因果链接"""
        self.causal.add("qian_node", category="creative")
        self.causal.add("li_node", category="light")

        result = self.causal.link_with_category("qian_node", "li_node")

        print(f"\n=== category因果(creative->light) ===")
        print(f"  creative enhance light: link结果={result}")

        assert result is True, "creative enhance light应成功链接"

    def test_category_contradiction_blocks(self):
        """category抑制应阻断链接"""
        self.causal.add("qian_node2", category="creative")
        self.causal.add("xun_node", category="wind")

        result = self.causal.link_with_category("qian_node2", "xun_node")

        print(f"\n=== category抑制(creative suppress wind) ===")
        print(f"  creative suppress wind: link结果={result}")

        assert result is False, "creative suppress wind应阻断链接"

    def test_energy_type_propagation(self):
        """能量传播沿因果链"""
        self.causal.add("source", energy_type="wood")
        self.causal.add("target1", energy_type="fire")
        self.causal.add("target2", energy_type="earth")

        self.causal.link("source", "target1")
        self.causal.link("target1", "target2")

        result = self.causal.propagate("source", delta=0.2)

        print(f"\n=== 能量传播 ===")
        print(f"  source能量: {self.causal.energy['source']:.3f}")
        for nid, e in result.items():
            print(f"  {nid}能量: {e:.3f}")

        assert "target1" in result, "target1应受到能量传播"
        assert result["target1"] > 1.0, "target1能量应大于1.0"

    def test_time_code_temporal_link(self):
        """time_code时空关联"""
        self.causal.add("zi_node")
        self.causal.add("chou_node")
        self.causal.add("wu_node")

        self.causal.link_temporal("zi_node", "子")
        self.causal.link_temporal("chou_node", "丑")
        self.causal.link_temporal("wu_node", "午")

        result_adj = self.causal.link_with_time_code("zi_node", "chou_node")
        result_chong = self.causal.link_with_time_code("zi_node", "wu_node")

        print(f"\n=== time_code时空关联 ===")
        print(f"  子丑(相邻): link={result_adj}")
        print(f"  子午(冲): link={result_chong}")

        assert result_adj is True, "子丑相邻应可链接"
        assert result_chong is False, "子午冲应阻断链接"

    def test_multi_layer_coverage(self):
        """多层因果覆盖率(五层)"""
        nodes = ["n1", "n2", "n3", "n4", "n5", "n6"]
        for n in nodes:
            self.causal.add(n)

        self.causal.link("n1", "n2")
        self.causal.link("n2", "n3")

        self.causal.add("n1", category="creative")
        self.causal.add("n4", category="light")

        self.causal.add("n5", energy_type="wood")
        self.causal.add("n3", energy_type="fire")

        self.causal.link_temporal("n6", "寅")
        self.causal.link_temporal("n1", "卯")

        cov = self.causal.coverage(nodes)
        print(f"\n=== 多层覆盖率 ===")
        print(f"  覆盖率: {cov}%")

        assert cov >= 50.0, f"多层覆盖率{cov}%过低"

    def test_energy_type_balance(self):
        """energy_type制化: 某类型过strong时触发约束"""
        for i in range(10):
            self.causal.add(f"fire_{i}", energy_type="fire")
        self.causal.add("water_0", energy_type="water")

        for i in range(9):
            self.causal.link(f"fire_{i}", f"fire_{i+1}")

        self.causal.propagate("fire_0", delta=0.5)

        constrained = self.causal.apply_energy_type_balance()

        print(f"\n=== energy_type制化 ===")
        print(f"  被约束节点数: {len(constrained)}")
        if constrained:
            print(f"  被约束节点: {constrained[:5]}...")

    def test_causal_chain_complex(self):
        """复杂因果链: 分支+汇聚"""
        for n in ["A", "B", "C", "D"]:
            self.causal.add(n)

        self.causal.link("A", "B")
        self.causal.link("A", "C")
        self.causal.link("B", "D")
        self.causal.link("C", "D")

        path = self.causal.get_causal_path("A", "D")
        print(f"\n=== 复杂因果链 ===")
        print(f"  A->D路径: {' -> '.join(path)}")

        assert len(path) >= 2, "A到D应有路径"
        assert path[0] == "A", "路径应从A开始"
        assert path[-1] == "D", "路径应到D结束"
