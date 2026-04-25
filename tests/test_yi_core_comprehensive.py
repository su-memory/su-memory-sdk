"""
yi_core / su_core 全面单元测试套件

覆盖：
  - 八卦分类器（_c1.py）
  - 五行生克引擎（_c2.py）
  - 天干地支编码器（ganzhi.py）
  - 64卦编码器（yijing.py + encoders.py）
  - 语义编码器（encoders.py）
  - 多视图检索融合（fusion.py）
  - 信念追踪（states.py）
  - 时序系统（chrono.py）
"""

import sys
import time
import pytest

sys.path.insert(0, "/Users/mac/.openclaw/workspace/su-memory")


# ============================================================
# 辅助：性能计时
# ============================================================

def measure_ms(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed



# ============================================================
# 3.1 八卦分类器测试
# ============================================================

class TestBaguaClassifier:
    """八卦推断函数 infer_bagua_from_content 全面测试"""

    def setup_method(self):
        from su_core._sys._c1 import infer_bagua_from_content, Bagua, BAGUA_ASSOCIATIONS
        self.infer = infer_bagua_from_content
        self.Bagua = Bagua
        self.BAGUA_ASSOCIATIONS = BAGUA_ASSOCIATIONS

    def test_bagua_count(self):
        assert len(list(self.Bagua)) == 8

    def test_bagua_all_names(self):
        expected = {"乾", "兑", "离", "震", "巽", "坎", "艮", "坤"}
        got = {b.name_zh for b in self.Bagua}
        assert got == expected

    def test_bagua_symbols(self):
        expected = {"☰", "☱", "☲", "☳", "☴", "☵", "☶", "☷"}
        got = {b.symbol for b in self.Bagua}
        assert got == expected

    def test_bagua_wuxing_values(self):
        valid = {"金", "木", "水", "火", "土"}
        for b in self.Bagua:
            assert b.wuxing in valid

    def test_qian_attributes(self):
        b = self.Bagua.QIAN
        assert b.name_zh == "乾"
        assert b.wuxing == "金"
        assert b.direction == "西北"
        assert b.symbol == "☰"

    def test_kun_attributes(self):
        b = self.Bagua.KUN
        assert b.name_zh == "坤"
        assert b.wuxing == "土"
        assert b.direction == "西南"

    def test_li_fire(self):
        assert self.Bagua.LI.wuxing == "火"

    def test_kan_water(self):
        assert self.Bagua.KAN.wuxing == "水"

    def test_zhen_xun_mu(self):
        assert self.Bagua.ZHEN.wuxing == "木"
        assert self.Bagua.XUN.wuxing == "木"

    def test_gen_kun_tu(self):
        assert self.Bagua.GEN.wuxing == "土"
        assert self.Bagua.KUN.wuxing == "土"

    def test_qian_dui_jin(self):
        assert self.Bagua.QIAN.wuxing == "金"
        assert self.Bagua.DUI.wuxing == "金"

    def test_from_trigram_by_name(self):
        assert self.Bagua.from_trigram("乾") == self.Bagua.QIAN
        assert self.Bagua.from_trigram("坤") == self.Bagua.KUN

    def test_from_trigram_by_symbol(self):
        assert self.Bagua.from_trigram("☰") == self.Bagua.QIAN
        assert self.Bagua.from_trigram("☷") == self.Bagua.KUN

    def test_from_trigram_invalid(self):
        with pytest.raises(ValueError):
            self.Bagua.from_trigram("UNKNOWN_TRIGRAM")

    def test_get_associations_keys(self):
        assoc = self.Bagua.QIAN.get_associations()
        for key in ["卦名", "符号", "五行", "方位", "性情", "类别"]:
            assert key in assoc

    def test_infer_default_empty(self):
        result = self.infer("")
        assert result == self.Bagua.DUI

    def test_infer_pure_number(self):
        result = self.infer("12345678")
        assert result in list(self.Bagua)

    def test_infer_long_text(self):
        result = self.infer("知识" * 1000)
        assert result in list(self.Bagua)

    def test_infer_non_chinese(self):
        result = self.infer("hello world this is a test string")
        assert result in list(self.Bagua)

    def test_infer_knowledge_type(self):
        result = self.infer("这是一段文字", metadata={"type": "knowledge"})
        assert result == self.Bagua.LI

    def test_infer_goal_type(self):
        result = self.infer("设定目标", metadata={"type": "goal"})
        assert result == self.Bagua.GEN

    def test_infer_danger_type(self):
        result = self.infer("风险评估", metadata={"type": "danger"})
        assert result == self.Bagua.KAN

    def test_infer_twenty_texts(self):
        texts = [
            "用户喜欢吃苹果", "项目存在严重风险", "制定下季度目标",
            "研究报告已经完成", "发生了紧急事件", "团队协作效率提升",
            "基础设施建设完成", "父亲身体健康", "市场存在不确定性",
            "教育资源需要补充", "突然发现系统漏洞", "关系网络扩展中",
            "停止无效投资", "水资源匮乏问题", "连接中断需要修复",
            "目标完成进度80%", "确定的技术方案", "团队满意度高",
            "雷击导致断电事故", "风向改变了策略",
        ]
        for t in texts:
            r = self.infer(t)
            assert r in list(self.Bagua)

    def test_infer_kan_keywords(self):
        result = self.infer("问题和危险风险很大有困难失败可能")
        assert result == self.Bagua.KAN

    def test_infer_returns_bagua_enum(self):
        result = self.infer("任意文本")
        assert isinstance(result, self.Bagua)


# ============================================================
# 3.2 五行生克引擎测试
# ============================================================

class TestWuxingEngine:

    def setup_method(self):
        from su_core._sys._c2 import (
            Wuxing, WUXING_SHENG, WUXING_KE,
            WuxingEnergyNetwork, WuxingState, wuxing_from_bagua
        )
        self.Wuxing = Wuxing
        self.SHENG = WUXING_SHENG
        self.KE = WUXING_KE
        self.Network = WuxingEnergyNetwork
        self.WuxingState = WuxingState
        self.from_bagua = wuxing_from_bagua

    def test_five_elements_count(self):
        assert len(list(self.Wuxing)) == 5

    def test_element_names(self):
        names = {w.element for w in self.Wuxing}
        assert names == {"木", "火", "土", "金", "水"}

    def test_seasons_present(self):
        seasons = {w.season for w in self.Wuxing}
        assert "春" in seasons and "秋" in seasons

    def test_directions_present(self):
        directions = {w.direction for w in self.Wuxing}
        assert "东" in directions and "西" in directions

    def test_sheng_chain_complete(self):
        chain = [
            (self.Wuxing.MU, self.Wuxing.HUO),
            (self.Wuxing.HUO, self.Wuxing.TU),
            (self.Wuxing.TU, self.Wuxing.JIN),
            (self.Wuxing.JIN, self.Wuxing.SHUI),
            (self.Wuxing.SHUI, self.Wuxing.MU),
        ]
        for src, target in chain:
            assert self.SHENG[src] == target

    def test_sheng_dict_size(self):
        assert len(self.SHENG) == 5

    def test_ke_chain_complete(self):
        chain = [
            (self.Wuxing.MU, self.Wuxing.TU),
            (self.Wuxing.TU, self.Wuxing.SHUI),
            (self.Wuxing.SHUI, self.Wuxing.HUO),
            (self.Wuxing.HUO, self.Wuxing.JIN),
            (self.Wuxing.JIN, self.Wuxing.MU),
        ]
        for src, target in chain:
            assert self.KE[src] == target

    def test_ke_dict_size(self):
        assert len(self.KE) == 5

    def test_no_self_sheng(self):
        for w in self.Wuxing:
            assert self.SHENG[w] != w

    def test_no_self_ke(self):
        for w in self.Wuxing:
            assert self.KE[w] != w

    def test_sheng_ke_different_targets(self):
        for w in self.Wuxing:
            assert self.SHENG[w] != self.KE[w]

    def test_wuxing_state_default_intensity(self):
        state = self.WuxingState(wuxing=self.Wuxing.MU)
        assert state.intensity == 1.0

    def test_effective_intensity_sheng_boost(self):
        state = self.WuxingState(wuxing=self.Wuxing.MU)
        env = self.WuxingState(wuxing=self.Wuxing.HUO)
        eff = state.get_effective_intensity(env)
        assert eff == pytest.approx(1.0, rel=1e-3)

    def test_effective_intensity_ke_penalty(self):
        state = self.WuxingState(wuxing=self.Wuxing.MU)
        env = self.WuxingState(wuxing=self.Wuxing.TU)
        eff = state.get_effective_intensity(env)
        assert eff == pytest.approx(0.5, rel=1e-3)

    def test_effective_intensity_no_env(self):
        state = self.WuxingState(wuxing=self.Wuxing.SHUI, intensity=2.0)
        assert state.get_effective_intensity() == 2.0

    def test_network_dominant_wuxing(self):
        net = self.Network()
        net.register_memory("m1", self.Wuxing.MU)
        net.register_memory("m2", self.Wuxing.MU)
        net.register_memory("m3", self.Wuxing.SHUI)
        assert net.get_dominant_wuxing() == self.Wuxing.MU

    def test_network_empty_default_tu(self):
        net = self.Network()
        assert net.get_dominant_wuxing() == self.Wuxing.TU

    def test_network_propagate_energy(self):
        net = self.Network()
        net.register_memory("src", self.Wuxing.MU)
        net.register_memory("tgt", self.Wuxing.HUO)
        before = net.memory_states["tgt"].intensity
        net.propagate_energy("src", 0.5)
        after = net.memory_states["tgt"].intensity
        assert after > before

    def test_network_missing_source_no_crash(self):
        net = self.Network()
        net.propagate_energy("nonexistent", 1.0)

    def test_wuxing_from_bagua_qian(self):
        assert self.from_bagua("乾") == self.Wuxing.JIN

    def test_wuxing_from_bagua_li(self):
        assert self.from_bagua("离") == self.Wuxing.HUO

    def test_wuxing_from_bagua_kan(self):
        assert self.from_bagua("坎") == self.Wuxing.SHUI

    def test_wuxing_from_bagua_unknown_default(self):
        assert self.from_bagua("UNKNOWN") == self.Wuxing.TU


# ============================================================
# 3.3 天干地支编码器测试
# ============================================================

class TestGanzhi:

    def setup_method(self):
        from su_core._sys.ganzhi import (
            Tiangan, Dizhi, Jiazi,
            TIANGAN_HE, TIANGAN_CHONG,
            create_ganzhi, get_jiagan, get_dizhi, get_jiazi
        )
        self.Tiangan = Tiangan
        self.Dizhi = Dizhi
        self.Jiazi = Jiazi
        self.TIANGAN_HE = TIANGAN_HE
        self.TIANGAN_CHONG = TIANGAN_CHONG
        self.create_ganzhi = create_ganzhi
        self.get_jiagan = get_jiagan
        self.get_dizhi = get_dizhi
        self.get_jiazi = get_jiazi

    def test_tiangan_count(self):
        assert len(list(self.Tiangan)) == 10

    def test_tiangan_names_complete(self):
        expected = {"甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"}
        got = {t.name for t in self.Tiangan}
        assert got == expected

    def test_tiangan_yin_yang(self):
        for t in self.Tiangan:
            expected_yy = "阳" if t.value % 2 == 0 else "阴"
            assert t.yin_yang == expected_yy

    def test_tiangan_elements(self):
        pairs = [(0,"木"),(1,"木"),(2,"火"),(3,"火"),(4,"土"),
                 (5,"土"),(6,"金"),(7,"金"),(8,"水"),(9,"水")]
        tg_list = list(self.Tiangan)
        for idx, wx in pairs:
            assert tg_list[idx].element == wx

    def test_tiangan_he_count(self):
        assert len(self.TIANGAN_HE) == 5

    def test_tiangan_chong_count(self):
        assert len(self.TIANGAN_CHONG) == 5

    def test_dizhi_count(self):
        assert len(list(self.Dizhi)) == 12

    def test_dizhi_names_complete(self):
        expected = {"子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥"}
        got = {d.name for d in self.Dizhi}
        assert got == expected

    def test_dizhi_zi_water(self):
        assert list(self.Dizhi)[0].element == "水"

    def test_dizhi_yin_wood(self):
        assert list(self.Dizhi)[2].element == "木"

    def test_dizhi_wu_fire(self):
        assert list(self.Dizhi)[6].element == "火"

    def test_dizhi_shen_metal(self):
        assert list(self.Dizhi)[8].element == "金"

    def test_dizhi_zi_hidden_gui(self):
        dz = self.Dizhi.ZI_YANG
        hidden = dz.hidden_stems
        assert any(t.name == "癸" for t in hidden)

    def test_dizhi_yin_hidden(self):
        dz = self.Dizhi.YIN_YANG
        hidden = dz.hidden_stems
        names = {t.name for t in hidden}
        assert {"甲", "丙", "戊"}.issubset(names)

    def test_jiazi_singleton(self):
        j1 = self.Jiazi()
        j2 = self.Jiazi()
        assert j1 is j2

    def test_jiazi_name_0(self):
        assert self.Jiazi().get_name(0) == "甲子"

    def test_jiazi_name_1(self):
        assert self.Jiazi().get_name(1) == "乙丑"

    def test_jiazi_name_59(self):
        assert self.Jiazi().get_name(59) == "癸亥"

    def test_jiazi_cycle_60(self):
        j = self.Jiazi()
        assert j.get_name(60) == j.get_name(0)

    def test_jiazi_sixty_unique(self):
        j = self.Jiazi()
        names = [j.get_name(i) for i in range(60)]
        assert len(set(names)) == 60

    def test_jiazi_wuxing_valid(self):
        j = self.Jiazi()
        valid = {"木", "火", "土", "金", "水"}
        for i in range(60):
            wx = j.get_wuxing(i)
            assert wx in valid

    def test_get_jiagan_jiazi(self):
        assert self.get_jiagan(0) == "甲"
        assert self.get_jiagan(10) == "甲"
        assert self.get_jiagan(9) == "癸"

    def test_get_dizhi_cycle(self):
        assert self.get_dizhi(0) == "子"
        assert self.get_dizhi(12) == "子"

    def test_get_jiazi_func(self):
        assert self.get_jiazi(0) == "甲子"
        assert self.get_jiazi(59) == "癸亥"

    def test_create_ganzhi_jiazi(self):
        gz = self.create_ganzhi(0, 0)
        assert gz.tiangan.name == "甲"
        assert gz.dizhi.name == "子"
        assert gz.element == "木"
        assert gz.name == "甲子"

    def test_create_ganzhi_wuxing_equals_element(self):
        gz = self.create_ganzhi(0, 0)
        assert gz.wuxing == gz.element

    def test_create_ganzhi_overflow_no_crash(self):
        gz = self.create_ganzhi(10, 12)
        valid_names = {"甲","乙","丙","丁","戊","己","庚","辛","壬","癸"}
        assert gz.tiangan.name in valid_names


# ============================================================
# 3.4 64卦编码器测试
# ============================================================

class TestHexagramEncoding:

    def setup_method(self):
        from su_core._sys.yijing import (
            HexagramType, create_hexagram, HEXAGRAM_NAMES, get_jianggong, YiJingRule
        )
        from su_core._sys.encoders import (
            SemanticEncoder, EncoderCore, EncodingInfo, HEXAGRAM_NAMES as ENC_NAMES
        )
        self.HexagramType = HexagramType
        self.create_hexagram = create_hexagram
        self.HEXAGRAM_NAMES = HEXAGRAM_NAMES
        self.YiJingRule = YiJingRule
        self.SemanticEncoder = SemanticEncoder
        self.EncoderCore = EncoderCore
        self.EncodingInfo = EncodingInfo
        self.ENC_NAMES = ENC_NAMES

    def test_hexagram_type_count(self):
        assert len(list(self.HexagramType)) == 8

    def test_hexagram_type_names(self):
        names = {h.name_zh for h in self.HexagramType}
        assert names == {"乾", "兑", "离", "震", "巽", "坎", "艮", "坤"}

    def test_hexagram_wuxing(self):
        ht = self.HexagramType
        assert ht.QIAN.wuxing == "金"
        assert ht.LI.wuxing == "火"
        assert ht.KAN.wuxing == "水"
        assert ht.ZHEN.wuxing == "木"
        assert ht.KUN.wuxing == "土"

    def test_hexagram_sheng_no_crash(self):
        for ht in self.HexagramType:
            assert isinstance(ht.sheng, self.HexagramType)

    def test_hexagram_ke_no_crash(self):
        for ht in self.HexagramType:
            assert isinstance(ht.ke, self.HexagramType)

    def test_hexagram_names_64(self):
        assert len(self.HEXAGRAM_NAMES) == 64

    def test_hexagram_first_qian(self):
        assert self.HEXAGRAM_NAMES[0] == "乾"

    def test_hexagram_second_kun(self):
        assert self.HEXAGRAM_NAMES[1] == "坤"

    def test_hexagram_last_weiji(self):
        assert self.HEXAGRAM_NAMES[63] == "未济"

    def test_hexagram_names_unique(self):
        assert len(set(self.HEXAGRAM_NAMES)) == 64

    def test_create_qian(self):
        h = self.create_hexagram(0, 0)
        info = h.get_base_info()
        assert info["upper"] == "乾"
        assert info["lower"] == "乾"
        assert info["name"] == "乾"

    def test_create_hexagram_gua_xiang(self):
        h = self.create_hexagram(1, 0)
        assert "兑" in h.gua_xiang
        assert "乾" in h.gua_xiang

    def test_create_hexagram_wuxing_from_upper(self):
        h = self.create_hexagram(2, 0)  # 离上乾下
        assert h.wuxing == "火"

    def test_create_hexagram_all_no_crash(self):
        for upper in range(8):
            for lower in range(8):
                h = self.create_hexagram(upper, lower)
                assert h.upper is not None

    def test_yijing_bu_yi(self):
        result = self.YiJingRule.bu_yi()
        assert isinstance(result, str) and len(result) > 0

    def test_yijing_bian_yi_moving(self):
        result = self.YiJingRule.bian_yi("木", True)
        assert isinstance(result, str)

    def test_yijing_bian_yi_still(self):
        result = self.YiJingRule.bian_yi("火", False)
        assert isinstance(result, str)

    def test_yijing_jian_yi(self):
        result = self.YiJingRule.jian_yi("水")
        assert "水" in result

    def test_encoding_info_64(self):
        for i in range(64):
            info = self.EncodingInfo.from_index(i)
            assert info.index == i
            assert 0 <= info.hu_gua <= 63
            assert 0 <= info.zong_gua <= 63
            assert info.cuo_gua == 63 - i

    def test_encoding_info_wuxing_valid(self):
        valid = {"金", "木", "水", "火", "土"}
        for i in range(64):
            info = self.EncodingInfo.from_index(i)
            assert info.wuxing in valid

    def test_encoding_info_direction_nonempty(self):
        for i in range(64):
            info = self.EncodingInfo.from_index(i)
            assert info.direction != ""

    def test_semantic_encoder_returns_info(self):
        enc = self.SemanticEncoder()
        result = enc.encode("用户有高血压病史", "fact")
        assert isinstance(result, self.EncodingInfo)

    def test_semantic_encoder_deterministic(self):
        enc = self.SemanticEncoder()
        r1 = enc.encode("固定文本ABC", "fact")
        r2 = enc.encode("固定文本ABC", "fact")
        assert r1.index == r2.index

    def test_semantic_encoder_index_range(self):
        enc = self.SemanticEncoder()
        for text in ["hello", "你好", "12345", "数据科学研究"]:
            info = enc.encode(text)
            assert 0 <= info.index <= 63

    def test_semantic_encoder_batch(self):
        enc = self.SemanticEncoder()
        items = [{"content": f"内容{i}", "type": "fact"} for i in range(5)]
        results = enc.batch_encode(items)
        assert len(results) == 5

    def test_semantic_encoder_empty_no_crash(self):
        enc = self.SemanticEncoder()
        result = enc.encode("")
        assert 0 <= result.index <= 63

    def test_semantic_encoder_performance(self):
        enc = self.SemanticEncoder()
        enc.encode("预热", "fact")
        _, elapsed = measure_ms(enc.encode, "性能测试文本" * 10, "fact")
        assert elapsed < 80.0, f"编码耗时 {elapsed:.2f}ms 超过80ms"

    def test_encoder_core_views(self):
        core = self.EncoderCore()
        views = core.get_holographic_views(0)
        assert set(views.keys()) == {"本卦", "互卦", "综卦", "错卦"}
        assert views["本卦"] == 0

    def test_encoder_core_cuo_formula(self):
        core = self.EncoderCore()
        for i in range(64):
            views = core.get_holographic_views(i)
            assert views["错卦"] == 63 - i

    def test_encoder_core_self_retrieval_first(self):
        core = self.EncoderCore()
        scored = core.retrieve_holographic(0, list(range(8)), top_k=8)
        assert scored[0][0] == 0
        assert scored[0][1] == 1.0

    def test_encoder_core_top_k(self):
        core = self.EncoderCore()
        for k in [1, 3, 5]:
            scored = core.retrieve_holographic(0, list(range(20)), top_k=k)
            assert len(scored) <= k

    def test_encoder_core_scores_descending(self):
        core = self.EncoderCore()
        scored = core.retrieve_holographic(5, list(range(20)), top_k=10)
        scores = [s for _, s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_cuo_gua_double_inverse(self):
        core = self.EncoderCore()
        for i in range(64):
            views = core.get_holographic_views(i)
            cuo = views["错卦"]
            views2 = core.get_holographic_views(cuo)
            assert views2["错卦"] == i

    def test_four_views_range(self):
        core = self.EncoderCore()
        for i in range(64):
            for key, val in core.get_holographic_views(i).items():
                assert 0 <= val <= 63


# ============================================================
# 3.5 信念追踪测试
# ============================================================

class TestBeliefTracker:

    def setup_method(self):
        from su_core._sys.states import BeliefTracker, BeliefState, BeliefStage
        self.BeliefTracker = BeliefTracker
        self.BeliefState = BeliefState
        self.BeliefStage = BeliefStage

    def test_initialize_stage_cognition(self):
        t = self.BeliefTracker()
        state = t.initialize("m1")
        assert state.stage == self.BeliefStage.COGNITION

    def test_initialize_confidence_half(self):
        t = self.BeliefTracker()
        state = t.initialize("m1")
        assert state.confidence == pytest.approx(0.5)

    def test_initialize_counts_zero(self):
        t = self.BeliefTracker()
        state = t.initialize("m1")
        assert state.reinforce_count == 0
        assert state.shake_count == 0

    def test_reinforce_increases_confidence(self):
        t = self.BeliefTracker()
        t.initialize("m1")
        before = t.get_state("m1").confidence
        t.reinforce("m1")
        assert t.get_state("m1").confidence > before

    def test_reinforce_threshold_transition(self):
        t = self.BeliefTracker()
        t.initialize("m1")
        for _ in range(3):
            t.reinforce("m1")
        state = t.get_state("m1")
        assert state.stage in [self.BeliefStage.CONFIRM, self.BeliefStage.REINFORCE]

    def test_shake_decreases_confidence(self):
        t = self.BeliefTracker()
        t.initialize("m1")
        for _ in range(4):
            t.reinforce("m1")
        before = t.get_state("m1").confidence
        t.shake("m1")
        assert t.get_state("m1").confidence < before

    def test_get_state_none_for_unknown(self):
        t = self.BeliefTracker()
        assert t.get_state("nonexistent") is None

    def test_stage_distribution(self):
        t = self.BeliefTracker()
        t.initialize("m1")
        t.initialize("m2")
        dist = t.get_stage_distribution()
        assert self.BeliefStage.COGNITION in dist
        assert dist[self.BeliefStage.COGNITION] == 2

    def test_should_forget_false_initial(self):
        t = self.BeliefTracker()
        t.initialize("m1")
        assert t.should_forget("m1") is False

    def test_should_forget_unknown(self):
        t = self.BeliefTracker()
        assert t.should_forget("unknown") is False

    def test_transitions_recorded(self):
        t = self.BeliefTracker()
        state = t.initialize("m1")
        assert "认知" in state.transitions

    def test_reinforce_without_init(self):
        t = self.BeliefTracker()
        state = t.reinforce("auto_init")
        assert state.reinforce_count >= 1

    def test_shake_with_conflict_id(self):
        t = self.BeliefTracker()
        t.initialize("m1")
        state = t.shake("m1", conflict_with="m2")
        assert state is not None


# ============================================================
# 3.6 多视图检索融合测试
# ============================================================

class TestMultiViewRetriever:

    def setup_method(self):
        from su_core._sys.fusion import MultiViewRetriever
        from su_core._sys.encoders import SemanticEncoder, EncodingInfo
        self.MultiViewRetriever = MultiViewRetriever
        self.SemanticEncoder = SemanticEncoder
        self.EncodingInfo = EncodingInfo

    def _candidates(self, query_idx):
        return [
            {"id": "c0", "content": "完全匹配", "hexagram_index": query_idx, "vector_score": 0.9},
            {"id": "c1", "content": "相关1", "hexagram_index": (query_idx+5)%64, "vector_score": 0.6},
            {"id": "c2", "content": "相关2", "hexagram_index": (query_idx+10)%64, "vector_score": 0.5},
            {"id": "c3", "content": "无关", "hexagram_index": (query_idx+30)%64, "vector_score": 0.3},
        ]

    def test_retrieve_returns_list(self):
        enc = self.SemanticEncoder()
        ret = self.MultiViewRetriever()
        qh = enc.encode("用户有高血压", "fact")
        results = ret.retrieve("高血压", qh, self._candidates(qh.index), top_k=4)
        assert isinstance(results, list)

    def test_retrieve_top_k_limit(self):
        enc = self.SemanticEncoder()
        ret = self.MultiViewRetriever()
        qh = enc.encode("测试查询", "fact")
        for k in [1, 2, 3]:
            cands = self._candidates(qh.index)
            results = ret.retrieve("查询", qh, cands, top_k=k)
            assert len(results) <= k

    def test_retrieve_holographic_score(self):
        enc = self.SemanticEncoder()
        ret = self.MultiViewRetriever()
        qh = enc.encode("知识测试", "fact")
        results = ret.retrieve("知识", qh, self._candidates(qh.index), top_k=4)
        for r in results:
            assert "holographic_score" in r

    def test_retrieve_exact_match_highest(self):
        enc = self.SemanticEncoder()
        ret = self.MultiViewRetriever()
        qh = enc.encode("健康问题", "fact")
        results = ret.retrieve("健康", qh, self._candidates(qh.index), top_k=4)
        assert results[0]["id"] == "c0"

    def test_retrieve_scores_descending(self):
        enc = self.SemanticEncoder()
        ret = self.MultiViewRetriever()
        qh = enc.encode("用户目标", "goal")
        results = ret.retrieve("目标", qh, self._candidates(qh.index), top_k=4)
        scores = [r["holographic_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_empty_candidates(self):
        enc = self.SemanticEncoder()
        ret = self.MultiViewRetriever()
        qh = enc.encode("测试", "fact")
        assert ret.retrieve("测试", qh, [], top_k=5) == []

    def test_retrieve_holo_detail_keys(self):
        enc = self.SemanticEncoder()
        ret = self.MultiViewRetriever()
        qh = enc.encode("知识存储", "knowledge")
        results = ret.retrieve("知识", qh, self._candidates(qh.index), top_k=4)
        for r in results:
            detail = r.get("fusion_detail", {})
            for key in ["semantic", "bagua_soft", "wuxing_energy", "holographic", "causal"]:
                assert key in detail

    def test_retrieve_score_range(self):
        enc = self.SemanticEncoder()
        ret = self.MultiViewRetriever()
        qh = enc.encode("任意文本", "fact")
        results = ret.retrieve("文本", qh, self._candidates(qh.index), top_k=4)
        for r in results:
            assert 0.0 <= r["holographic_score"] <= 1.0


# ============================================================
# 3.7 时序系统测试
# ============================================================

class TestTemporalSystem:

    def setup_method(self):
        from su_core._sys.chrono import TemporalSystem, TemporalInfo, DynamicPriority
        from datetime import date
        self.TemporalSystem = TemporalSystem
        self.TemporalInfo = TemporalInfo
        self.DynamicPriority = DynamicPriority
        self.date = date

    def test_get_current_ganzhi(self):
        ts = self.TemporalSystem()
        info = ts.get_current_ganzhi()
        assert isinstance(info, self.TemporalInfo)

    def test_current_ganzhi_fields(self):
        ts = self.TemporalSystem()
        info = ts.get_current_ganzhi()
        assert info.ganzhi
        assert info.season in ["春", "夏", "秋", "冬", "四季"]
        assert info.wuxing in ["金", "木", "水", "火", "土"]
        assert info.yin_yang in ["阴", "阳"]

    def test_spring_season(self):
        ts = self.TemporalSystem()
        info = ts.date_to_ganzhi(self.date(2024, 4, 1))
        assert info.season == "春"

    def test_summer_season(self):
        ts = self.TemporalSystem()
        info = ts.date_to_ganzhi(self.date(2024, 7, 1))
        assert info.season == "夏"

    def test_autumn_season(self):
        ts = self.TemporalSystem()
        info = ts.date_to_ganzhi(self.date(2024, 10, 1))
        assert info.season == "秋"

    def test_winter_season(self):
        ts = self.TemporalSystem()
        info_12 = ts.date_to_ganzhi(self.date(2024, 12, 15))
        assert info_12.season == "冬"
        info_1 = ts.date_to_ganzhi(self.date(2024, 1, 15))
        assert info_1.season == "四季"
        info_2 = ts.date_to_ganzhi(self.date(2024, 2, 15))
        assert info_2.season == "春"

    def test_1984_jiazi_year(self):
        ts = self.TemporalSystem()
        info = ts.date_to_ganzhi(self.date(1984, 6, 15))
        assert info.ganzhi.startswith("甲")

    def test_priority_range(self):
        ts = self.TemporalSystem()
        info = ts.get_current_ganzhi()
        for base in [0, 3, 5, 7, 10]:
            dp = ts.calculate_priority(base, info, "木")
            assert 0.0 <= dp.final_priority <= 1.0

    def test_priority_spring_wood_boost(self):
        ts = self.TemporalSystem()
        info = ts.date_to_ganzhi(self.date(2024, 4, 1))
        dp = ts.calculate_priority(5, info, "木")
        assert dp.season_boost == pytest.approx(-0.08)

    def test_priority_spring_metal_penalty(self):
        ts = self.TemporalSystem()
        info = ts.date_to_ganzhi(self.date(2024, 4, 1))
        dp = ts.calculate_priority(5, info, "金")
        assert dp.season_boost == pytest.approx(0.1)

    def test_priority_structure(self):
        ts = self.TemporalSystem()
        info = ts.get_current_ganzhi()
        dp = ts.calculate_priority(5, info, "火")
        assert hasattr(dp, "base_priority")
        assert hasattr(dp, "season_boost")
        assert hasattr(dp, "time_boost")
        assert hasattr(dp, "final_priority")


# ============================================================
# 3.8 性能基准测试
# ============================================================

class TestPerformanceBenchmark:

    def test_semantic_encoder_latency(self):
        from su_core._sys.encoders import SemanticEncoder
        enc = SemanticEncoder()
        enc.encode("预热", "fact")
        times = []
        for i in range(20):
            _, elapsed = measure_ms(enc.encode, f"性能测试内容{i}", "fact")
            times.append(elapsed)
        avg = sum(times) / len(times)
        assert avg < 20.0, f"平均编码 {avg:.2f}ms > 20ms"
        assert max(times) < 60.0, f"最大编码 {max(times):.2f}ms 异常"

    def test_batch_encode_20_items(self):
        from su_core._sys.encoders import SemanticEncoder
        enc = SemanticEncoder()
        items = [{"content": f"批量{i}", "type": "fact"} for i in range(20)]
        _, elapsed = measure_ms(enc.batch_encode, items)
        assert elapsed < 250.0, f"批量编码 {elapsed:.2f}ms > 250ms"

    def test_holographic_retrieval_latency(self):
        from su_core._sys.encoders import EncoderCore
        core = EncoderCore()
        candidates = list(range(64))
        _, elapsed = measure_ms(core.retrieve_holographic, 0, candidates, 8)
        assert elapsed < 5.0, f"全息检索 {elapsed:.2f}ms > 5ms"

    def test_infer_bagua_latency(self):
        from su_core._sys._c1 import infer_bagua_from_content
        _, elapsed = measure_ms(infer_bagua_from_content, "测试文本内容")
        assert elapsed < 20.0, f"八卦推断 {elapsed:.2f}ms > 20ms"


# ============================================================
# 3.9 公开 API 接口完整性
# ============================================================

class TestPublicAPI:

    def test_import_semantic_encoder(self):
        from su_core import SemanticEncoder
        assert SemanticEncoder is not None

    def test_import_encoder_core(self):
        from su_core import EncoderCore
        assert EncoderCore is not None

    def test_import_encoding_info(self):
        from su_core import EncodingInfo
        assert EncodingInfo is not None

    def test_import_multi_view_retriever(self):
        from su_core import MultiViewRetriever
        assert MultiViewRetriever is not None

    def test_import_su_compressor(self):
        from su_core import SuCompressor
        assert SuCompressor is not None

    def test_import_temporal_system(self):
        from su_core import TemporalSystem
        assert TemporalSystem is not None

    def test_import_belief_tracker(self):
        from su_core import BeliefTracker
        assert BeliefTracker is not None

    def test_import_meta_cognition(self):
        from su_core import MetaCognition
        assert MetaCognition is not None

    def test_version(self):
        import su_core
        assert hasattr(su_core, "__version__")
        assert su_core.__version__ == "1.0.0"

    def test_all_exports(self):
        import su_core
        for name in su_core.__all__:
            obj = getattr(su_core, name, None)
            assert obj is not None, f"su_core.{name} 未导出"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
