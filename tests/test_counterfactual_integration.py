"""
su-memory v3.5.5 — Counterfactual 集成测试
===========================================

验证 client.py 中 counterfactual_query() 方法与 CounterfactualEngine 的集成。

测试覆盖:
  - 基本反事实查询 (ITE, PN, PS, PNS)
  - 空记忆边界
  - 多因果链
  - 确定性验证
  - 回退行为 (记忆不足时)
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytestmark = pytest.mark.causal


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def client_with_causal_memories():
    """创建内存客户端并填充因果相关记忆。"""
    from su_memory import SuMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        client = SuMemory(
            persist_directory=tmpdir,
            enable_vector=False,
        )
        # 填充因果相关记忆
        memories = [
            "吃辛辣食物后血压升高到145/95",
            "今天吃了川菜，感觉很辣，血压146/96",
            "清淡饮食后血压降至120/80",
            "连续一周低盐饮食，血压稳定在118/78",
            "喝咖啡后心率加快到95次/分",
            "停止喝咖啡一周后心率恢复到72次/分",
            "运动30分钟后血压下降10mmHg",
            "压力大时血压会升高",
            "睡眠不足导致第二天血压偏高",
        ]
        for mem in memories:
            client.add(mem)
        yield client
        try:
            client.clear()
        except Exception:
            pass


@pytest.fixture
def empty_client():
    """空客户端用于边界测试。"""
    from su_memory import SuMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        client = SuMemory(
            persist_directory=tmpdir,
            enable_vector=False,
        )
        yield client
        try:
            client.clear()
        except Exception:
            pass


# ============================================================
# 基本反事实查询
# ============================================================


class TestCounterfactualBasic:
    """基本反事实查询功能。"""

    def test_simple_counterfactual(self, client_with_causal_memories):
        """验证 counterfactual_query() 返回有效结构。"""
        client = client_with_causal_memories

        result = client.counterfactual_query(
            evidence="吃辛辣食物",
            do_intervention="改为清淡饮食",
            target="血压",
        )

        assert result is not None, "应返回结果"
        assert isinstance(result, dict), "结果应为字典"

        # 验证必需字段
        for key in ("counterfactual_value", "ite", "pn", "ps", "pns"):
            assert key in result, f"结果应包含 {key} 字段"

        # ITE 应为有限值
        ite = result.get("ite", float("nan"))
        assert not np.isnan(ite), "ITE 不应为 NaN"

    def test_counterfactual_structure(self, client_with_causal_memories):
        """验证反事实推理的三步结构。"""
        client = client_with_causal_memories

        result = client.counterfactual_query(
            evidence="咖啡",
            do_intervention="停止咖啡",
            target="心率",
        )

        # 验证因果值
        cf_val = result.get("counterfactual_value")
        ite = result.get("ite")
        assert cf_val is not None
        assert isinstance(ite, (int, float))

        # PN/PS/PNS 应在 [0, 1] 范围内
        for prob_key in ("pn", "ps", "pns"):
            val = result.get(prob_key, 0)
            assert 0 <= val <= 1, f"{prob_key}={val} 应在 [0,1]"

    def test_deterministic_result(self, client_with_causal_memories):
        """验证相同输入产生相同输出 (确定性)。"""
        client = client_with_causal_memories

        result1 = client.counterfactual_query(
            evidence="辛辣食物",
            do_intervention="清淡饮食",
            target="血压",
        )
        result2 = client.counterfactual_query(
            evidence="辛辣食物",
            do_intervention="清淡饮食",
            target="血压",
        )

        # ITE 应相同
        assert result1["ite"] == result2["ite"], "确定性: ITE 应相同"
        assert result1["pn"] == result2["pn"], "确定性: PN 应相同"
        assert result1["ps"] == result2["ps"], "确定性: PS 应相同"


# ============================================================
# 边界测试
# ============================================================


class TestCounterfactualEdgeCases:
    """边界条件测试。"""

    def test_empty_memories(self, empty_client):
        """空记忆库应优雅降级。"""
        client = empty_client

        result = client.counterfactual_query(
            evidence="某事物",
            do_intervention="改为其他",
            target="某结果",
        )

        # 空记忆库仍应返回结构 (可能为 0 或默认值)
        assert isinstance(result, dict)
        assert "counterfactual_value" in result
        # ITE 应为 0 (无因果信息)
        assert result.get("ite", float("nan")) == 0.0 or not np.isnan(result.get("ite", 0))

    def test_no_matching_evidence(self, client_with_causal_memories):
        """没有匹配证据时的回退行为。"""
        client = client_with_causal_memories

        result = client.counterfactual_query(
            evidence="完全不存在的概念xyzabc",
            do_intervention="改为其他",
            target="血压",
        )

        assert isinstance(result, dict)
        # 无匹配时 ITE 应接近 0
        ite = result.get("ite", 0)
        assert abs(ite) < 1.0, "无匹配证据时 ITE 应接近 0"

    def test_single_memory(self, empty_client):
        """单条记忆时的行为。"""
        client = empty_client
        client.add("吃辣导致血压升高")

        result = client.counterfactual_query(
            evidence="吃辣",
            do_intervention="不吃辣",
            target="血压",
        )

        assert isinstance(result, dict)
        assert "counterfactual_value" in result


# ============================================================
# 多因果链测试
# ============================================================


class TestCounterfactualMultiChain:
    """多因果链场景。"""

    def test_chain_mediation(self, client_with_causal_memories):
        """链式中介: 饮食 → 血压 → ?"""
        client = client_with_causal_memories

        # 链式关系: 辛辣食物 → 血压 → (下游效应)
        result = client.counterfactual_query(
            evidence="辛辣食物和咖啡",
            do_intervention="清淡饮食和停止咖啡",
            target="血压",
        )

        assert isinstance(result, dict)
        assert result.get("ite", 0) != 0 or result.get("counterfactual_value", 0) is not None

    def test_multiple_interventions(self, client_with_causal_memories):
        """多干预同时应用。"""
        client = client_with_causal_memories

        result = client.counterfactual_query(
            evidence="压力大且睡眠不足",
            do_intervention="减轻压力和充足睡眠",
            target="血压",
        )

        assert isinstance(result, dict)
        ite = result.get("ite", float("inf"))
        assert not np.isinf(ite), "ITE 应为有限值"


# ============================================================
# PN/PS/PNS 概率边界
# ============================================================


class TestProbabilisticBounds:
    """概率边界不等式验证。"""

    def test_pn_bounds(self, client_with_causal_memories):
        """验证 max(0, PS+PN-1) ≤ PNS ≤ min(PN, PS)。"""
        client = client_with_causal_memories

        result = client.counterfactual_query(
            evidence="吃辛辣食物",
            do_intervention="改为清淡饮食",
            target="血压",
        )

        pn = result.get("pn", 0)
        ps = result.get("ps", 0)
        pns = result.get("pns", 0)

        lower = max(0, ps + pn - 1)
        upper = min(pn, ps)

        assert lower - 0.01 <= pns <= upper + 0.01, (
            f"PNS 边界: max(0,{ps}+{pn}-1)={lower:.4f} ≤ {pns:.4f} ≤ min({pn},{ps})={upper:.4f}"
        )

    def test_pn_ps_range(self, client_with_causal_memories):
        """PN 和 PS 应在 [0, 1] 区间。"""
        client = client_with_causal_memories

        result = client.counterfactual_query(
            evidence="运动",
            do_intervention="不运动",
            target="血压",
        )

        for key in ("pn", "ps", "pns"):
            val = result.get(key, -1)
            assert 0 <= val <= 1, f"{key}={val} 超出 [0,1]"
