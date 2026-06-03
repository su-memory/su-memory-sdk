"""
su-memory v3.6.0 — MCIWorldModel 集成测试
==========================================

测试覆盖:
- 初始化
- 因果发现
- 因果预测
- 因果解释
- 健康检查
- 干预接口桩
"""


import pytest

pytestmark = pytest.mark.e2e

from su_memory.sdk._world_model import (
    CausalWorldModelState,
    MCIWorldModel,
)


class TestCausalWorldModelState:
    """世界模型状态测试。"""

    def test_empty_state(self):
        state = CausalWorldModelState.empty()
        assert state.n_memories == 0
        assert state.n_confirmed == 0
        assert state.n_novel == 0
        assert state.n_suppressed == 0
        assert state.parametric_enhanced is False

    def test_to_dict(self):
        state = CausalWorldModelState(
            n_memories=10,
            n_confirmed=5,
            n_novel=2,
            n_suppressed=1,
        )
        d = state.to_dict()
        assert d["n_memories"] == 10
        assert d["n_confirmed"] == 5


class TestMCIWorldModel:
    """MCIWorldModel 集成测试。"""

    def test_init_without_lite_pro(self):
        """无 lite_pro 时也能初始化。"""
        wm = MCIWorldModel()
        assert wm.state is not None
        assert wm.state.n_memories == 0

    def test_initialize(self):
        """初始化报告生成。"""
        wm = MCIWorldModel()
        report = wm.initialize()
        assert "modules" in report
        assert "warnings" in report
        assert "ready" in report
        # 核心模块应可用
        assert report["modules"]["causal_pipeline"] in ("available", "unavailable")

    def test_health_check_initial(self):
        """初始健康检查。"""
        wm = MCIWorldModel()
        health = wm.health_check()
        assert health["version"] == "4.0.0"
        assert health["code_name"] == "MCI World Model v4.0.0 JEPA"
        assert "causal_pipeline" in health
        assert "jepa_predictor" in health
        assert "roadmap" in health

    def test_health_check_status(self):
        """健康状态判定。"""
        wm = MCIWorldModel()
        health = wm.health_check()
        # 未初始化 → not_initialized
        assert health["status"] in ("not_initialized", "degraded", "no_causal_data")

    def test_discover_no_memories(self):
        """无记忆时的因果发现。"""
        wm = MCIWorldModel()
        state = wm.discover(memories=[])
        assert state.n_memories == 0
        assert len(state.causal_edges) == 0

    def test_discover_with_minimal_memories(self):
        """最小记忆集因果发现。"""
        wm = MCIWorldModel()
        wm.initialize()
        memories = [
            {"id": "1", "content": "价格上涨导致需求下降"},
            {"id": "2", "content": "需求下降促使企业减产"},
            {"id": "3", "content": "企业减产引发供应紧张"},
        ]
        state = wm.discover(memories=memories)
        # 因果关系可能被检测到
        assert state.n_memories == 3

    def test_predict_effect_no_memories(self):
        """无记忆时的预测。"""
        wm = MCIWorldModel()
        results = wm.predict_effect("价格上涨", memories=[])
        assert results == []

    def test_predict_effect_with_memories(self):
        """有记忆时的预测。"""
        wm = MCIWorldModel()
        memories = [
            {"id": "1", "content": "暴雨导致城市内涝"},
            {"id": "2", "content": "城市内涝促使排水系统升级"},
            {"id": "3", "content": "排水系统升级改善防汛能力"},
        ]
        results = wm.predict_effect("暴雨", memories=memories)
        # 可能检测到效应
        assert isinstance(results, list)

    def test_fused_predict(self):
        """融合预测（检索 + 参数化）。"""
        wm = MCIWorldModel()
        memories = [
            {"id": "1", "content": "原材料价格上涨导致生产成本增加"},
            {"id": "2", "content": "生产成本增加促使产品调价"},
        ]
        results = wm.fused_predict("原材料价格上涨", memories=memories)
        assert isinstance(results, list)

    def test_parametric_predict_no_model(self):
        """无参数化模型时回退到检索路径。"""
        wm = MCIWorldModel()
        results = wm.parametric_predict("价格上涨")
        # 应返回检索路径结果
        assert isinstance(results, list)

    def test_explain_without_data(self):
        """无因果数据时解释。"""
        wm = MCIWorldModel()
        result = wm.explain("为什么价格下降?")
        assert "query" in result
        assert "chains" in result
        assert "summary" in result
        assert len(result["chains"]) == 0

    def test_explain_with_causal_data(self):
        """有因果数据时解释。"""
        wm = MCIWorldModel()
        wm.initialize()
        memories = [
            {"id": "1", "content": "气温升高导致用电量增加"},
            {"id": "2", "content": "用电量增加导致电力紧张"},
            {"id": "3", "content": "电力紧张促使限电措施"},
        ]
        wm.discover(memories=memories)
        result = wm.explain("用电量增加")
        assert "query" in result
        assert "summary" in result

    def test_intervene_full_implementation(self):
        """干预分析完整实现（v3.7.0）。"""
        wm = MCIWorldModel()
        wm.initialize()
        # 构建因果图后执行干预
        memories = [
            {"id": "1", "content": "价格变动导致需求变化"},
            {"id": "2", "content": "需求变化导致供应调整"},
            {"id": "3", "content": "供应调整导致价格变动"},
        ]
        wm.discover(memories=memories)
        result = wm.intervene(
            state="current",
            do_x={"V0": 1.5},
            target="V1",
        )
        assert "status" in result
        assert result["status"] == "ok"
        assert "ate" in result
        assert "method" in result
        # 验证非桩返回
        assert "implementation_status" not in result

    def test_intervene_insufficient_input(self):
        """干预缺少输入。"""
        wm = MCIWorldModel()
        result = wm.intervene()
        assert result["status"] == "insufficient_input"

    def test_train_parametric_no_model(self):
        """无参数化模型时训练失败。"""
        wm = MCIWorldModel()
        result = wm.train_parametric([])
        assert "error" in result

    def test_state_property(self):
        """状态属性可访问。"""
        wm = MCIWorldModel()
        state = wm.state
        assert isinstance(state, CausalWorldModelState)

    def test_repr(self):
        """字符串表示。"""
        wm = MCIWorldModel()
        rep = repr(wm)
        assert "MCIWorldModel" in rep
        assert "v4.0.0" in rep


class TestMCIWorldModelIntegration:
    """v3.6.0 端到端集成测试。"""

    def test_full_pipeline(self):
        """完整流水线：初始化 → 发现 → 预测 → 解释 → 健康检查。"""
        wm = MCIWorldModel()
        init_report = wm.initialize()
        assert "modules" in init_report

        memories = [
            {"id": "m1", "content": "市场需求激增导致原材料短缺"},
            {"id": "m2", "content": "原材料短缺导致生产成本上升"},
            {"id": "m3", "content": "生产成本上升导致产品涨价"},
            {"id": "m4", "content": "产品涨价导致销量下滑"},
            {"id": "m5", "content": "销量下滑导致库存积压"},
        ]

        # 因果发现
        state = wm.discover(memories=memories)
        assert state.n_memories == 5

        # 因果预测
        predictions = wm.predict_effect("市场需求激增", memories=memories)
        assert isinstance(predictions, list)

        # 因果解释
        explanation = wm.explain("成本上升")
        assert "summary" in explanation

        # 健康检查
        health = wm.health_check()
        assert health["version"] == "4.0.0"

    def test_discover_then_explain_chain(self):
        """发现后解释因果链。"""
        wm = MCIWorldModel()
        wm.initialize()
        memories = [
            {"id": "a", "content": "暴雨导致洪水"},
            {"id": "b", "content": "洪水导致道路中断"},
            {"id": "c", "content": "道路中断导致物流延迟"},
        ]
        wm.discover(memories=memories)
        explanation = wm.explain("洪水", max_depth=2)
        assert "query" in explanation
        # 应有因果链
        chains = explanation.get("chains", [])
        if chains:
            assert "path" in chains[0]


class TestJEPAIntegration:
    """v4.0.0 JEPA 集成验证。"""

    def test_jepa_predict_graceful_fallback(self):
        """JEPA 预测在无 lite_pro 时优雅回退到检索路径。"""
        from su_memory.sdk._world_model import MCIWorldModel
        wm = MCIWorldModel()
        wm.initialize()
        # 无记忆时 JEPA 预测应回退到检索
        result = wm.jepa_predict("价格上涨")
        assert isinstance(result, list)

    def test_jepa_predict_with_memories(self):
        """JEPA 预测使用编码器+预测器。"""
        from su_memory.sdk._world_model import MCIWorldModel
        wm = MCIWorldModel()
        wm.initialize()
        memories = [
            {"id": "1", "content": "成本上升导致价格上涨"},
            {"id": "2", "content": "价格上涨导致需求下降"},
            {"id": "3", "content": "需求下降导致库存积压"},
        ]
        result = wm.jepa_predict("价格上涨", memories=memories, top_k=3)
        assert isinstance(result, list)

    def test_train_jepa_no_data(self):
        """train_jepa 无数据时返回错误。"""
        from su_memory.sdk._world_model import MCIWorldModel
        wm = MCIWorldModel()
        wm.initialize()
        result = wm.train_jepa()
        assert "error" in result

    def test_health_check_v4_jepa(self):
        """health_check 报告 v4.0.0 JEPA 状态。"""
        from su_memory.sdk._world_model import MCIWorldModel
        wm = MCIWorldModel()
        wm.initialize()
        health = wm.health_check()
        assert health["version"] == "4.0.0"
        assert "jepa_predictor" in health
        assert health["jepa_predictor"]["available"]
        assert health["jepa_predictor"]["encoder_available"]
        assert "v4.0.0" in health["roadmap"]
        assert health["roadmap"]["v4.0.0"] == "jepa_world_model_closed_loop ✓"

    def test_jepa_encoder_availability(self):
        """JEPA 编码器在 initialize 后可用。"""
        from su_memory.sdk._world_model import MCIWorldModel
        wm = MCIWorldModel()
        report = wm.initialize()
        assert "jepa_encoder" in report
        assert report.get("jepa_encoder") == "initialized"

    def test_jepa_predictor_availability(self):
        """JEPA 预测器在 initialize 后可用（BeliefPropagation 基线）。"""
        from su_memory.sdk._world_model import MCIWorldModel
        wm = MCIWorldModel()
        report = wm.initialize()
        assert "jepa_predictor" in report
        assert report.get("jepa_predictor") == "initialized"

    def test_no_transformer_dependency(self):
        """v4.0.0: 验证 MCIWorldModel 不依赖 ParametricMemory/Transformer。"""
        from su_memory.sdk._world_model import MCIWorldModel
        wm = MCIWorldModel()
        wm.initialize()
        # _parametric 不再自动初始化
        assert wm._parametric is None
        # JEPA 编码器和预测器已初始化
        assert wm._jepa_encoder is not None
        assert wm._jepa_predictor is not None

    def test_predict_effect_still_works(self):
        """检索路径 predict_effect 在 v4.0.0 中保留不变。"""
        from su_memory.sdk._world_model import MCIWorldModel
        wm = MCIWorldModel()
        wm.initialize()
        memories = [
            {"id": "1", "content": "洪灾导致供应中断"},
            {"id": "2", "content": "供应中断导致价格飙升"},
            {"id": "3", "content": "价格飙升导致需求萎缩"},
        ]
        result = wm.predict_effect("供应中断", memories=memories, top_k=3)
        assert isinstance(result, list)

    def test_parametric_predict_delegates_to_jepa(self):
        """parametric_predict 已降级为 jepa_predict 别名。"""
        from su_memory.sdk._world_model import MCIWorldModel
        wm = MCIWorldModel()
        wm.initialize()
        result = wm.parametric_predict("价格上涨")
        assert isinstance(result, list)

    def test_train_parametric_delegates_to_train_jepa(self):
        """train_parametric 已降级为 train_jepa 别名。"""
        from su_memory.sdk._world_model import MCIWorldModel
        wm = MCIWorldModel()
        wm.initialize()
        result = wm.train_parametric(qa_pairs=[])
        assert isinstance(result, dict)
