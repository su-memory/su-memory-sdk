"""L0 不变式 + L1 动力学 oracle 测试 — 能量中心矩阵动力学。

验证 EnergyCore 的能量流是否满足其数学定义内含的不变式:
- L0: 转移矩阵列随机(列和=1, 守恒), 非负, 谱半径<=1
- L1: 总能量随步数守恒, 稳态非零且唯一

这些测试设计为"先红后绿": 修复前应失败(暴露守恒bug), 修复后通过。
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.invariant

import numpy as np
import pytest

from su_memory._sys._energy_core import EnergyCore
from su_memory._sys._terms import ENERGY_ENHANCE, ENERGY_SUPPRESS

ORDER = ["semantic", "causal", "spacetime", "generative", "trust"]
IDX = {n: i for i, n in enumerate(ORDER)}


def _build_transition_matrix(core: EnergyCore) -> np.ndarray:
    """从 EnergyCore 的流速率重建等价转移矩阵 A (x_{t+1} = A x_t)。"""
    n = len(ORDER)
    A = np.eye(n)
    enh = core.ENHANCE_FLOW_RATE
    sup = core.SUPPRESS_FLOW_RATE
    # 生: src -> tgt
    for src, tgt in ENERGY_ENHANCE.items():
        if src in IDX and tgt in IDX:
            A[IDX[tgt], IDX[src]] += enh
            A[IDX[src], IDX[src]] -= enh
    # 克: 被克方(tgt) -> 克方(src) (抑制=能量夺取, 守恒)
    sup_rate = abs(sup) if sup < 0 else sup
    for src, tgt in ENERGY_SUPPRESS.items():
        if src in IDX and tgt in IDX:
            A[IDX[src], IDX[tgt]] += sup_rate
            A[IDX[tgt], IDX[tgt]] -= sup_rate
    return A


@pytest.fixture
def core() -> EnergyCore:
    return EnergyCore()


class TestTransitionMatrixInvariants:
    """L0: 转移矩阵的数学不变式。"""

    def test_matrix_nonnegative(self, core):
        """转移矩阵必须非负(克关系不能消灭能量,只能重分配)。"""
        A = _build_transition_matrix(core)
        assert A.min() >= -1e-12, f"转移矩阵有负元素: min={A.min()}"

    def test_column_stochastic(self, core):
        """列和必须=1(能量守恒)。当前bug: 列和=0.975。"""
        A = _build_transition_matrix(core)
        col_sums = A.sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-9), f"列和≠1: {col_sums}"

    def test_spectral_radius_le_one(self, core):
        """谱半径<=1(系统不发散)。"""
        A = _build_transition_matrix(core)
        rho = np.max(np.abs(np.linalg.eigvals(A)))
        assert rho <= 1.0 + 1e-9, f"谱半径>1, 系统发散: rho={rho}"


class TestEnergyConservation:
    """L1: 能量随动力学步数守恒。"""

    def test_total_energy_constant_over_steps(self, core):
        """总能量在多步迭代后必须守恒(不耗散、不发散)。"""
        x0 = {"semantic": 0.9, "causal": 0.025, "spacetime": 0.025,
              "generative": 0.025, "trust": 0.025}
        history = core.simulate_energy_flow(x0, steps=50)
        initial_total = sum(x0.values())
        for i, state in enumerate(history):
            total = sum(state.values())
            assert abs(total - initial_total) < 1e-6, (
                f"第{i}步总能量{total:.6f}≠初始{initial_total:.6f}, 守恒被破坏"
            )

    def test_steady_state_nonzero(self, core):
        """长时间后稳态必须非零(不坍缩到零向量)。"""
        x0 = {"semantic": 0.9, "causal": 0.025, "spacetime": 0.025,
              "generative": 0.025, "trust": 0.025}
        history = core.simulate_energy_flow(x0, steps=200)
        final = history[-1]
        total = sum(final.values())
        assert total > 0.01, f"稳态坍缩到零: total={total}"

    def test_steady_state_independent_of_initial(self, core):
        """不同初值应收敛到同一稳态(不可约非负矩阵的Perron唯一性)。"""
        x_a = {"semantic": 0.9, "causal": 0.025, "spacetime": 0.025,
               "generative": 0.025, "trust": 0.025}
        x_b = {"semantic": 0.025, "causal": 0.9, "spacetime": 0.025,
               "generative": 0.025, "trust": 0.025}
        ha = core.simulate_energy_flow(x_a, steps=500)
        hb = core.simulate_energy_flow(x_b, steps=500)
        for key in x_a:
            assert abs(ha[-1][key] - hb[-1][key]) < 1e-3, (
                f"初值敏感: {key} 稳态A={ha[-1][key]:.4f} ≠ 稳态B={hb[-1][key]:.4f}"
            )


class TestFlowMatrixDynamics:
    """L0 不变式 — 矩阵化能量动力学的线性代数性质。"""

    def test_flow_matrix_column_sum_zero(self, core):
        """Flow 矩阵每列和必须 = 0 (能量不灭, 守恒律)。"""
        F = core.flow_matrix()
        col_sums = F.sum(axis=0)
        assert np.allclose(col_sums, 0.0, atol=1e-12), (
            f"Flow 列和≠0: {col_sums}, 能量不守恒"
        )

    def test_flow_matrix_equivalent_to_iteration(self, core):
        """矩阵乘法 x_{t+1}=clip(x+Fx) 必须等价于 simulate_energy_flow 单步。"""
        x0 = {"semantic": 0.7, "causal": 0.1, "spacetime": 0.1,
              "generative": 0.05, "trust": 0.05}
        # 通过 simulate_energy_flow 走 1 步
        sim_result = core.simulate_energy_flow(x0, steps=1)[1]
        # 通过矩阵直接算
        v = core._to_vector(x0)
        v_next = np.maximum(v + core._flow_matrix @ v, 0.0)
        mat_result = core._to_dict(v_next)
        for key in x0:
            assert abs(sim_result[key] - mat_result[key]) < 1e-12, (
                f"矩阵与迭代不一致: {key} sim={sim_result[key]:.6f} mat={mat_result[key]:.6f}"
            )

    def test_stationary_distribution_uniform(self, core):
        """对称生克环的平稳分布应为均匀分布 [0.2×5]。"""
        stat = core.stationary_distribution()
        assert np.allclose(stat, 0.2, atol=0.01), (
            f"平稳分布非均匀: {stat}, 对称五环应趋于均匀"
        )

    def test_balance_deviation_orders_distributions(self, core):
        """失衡度应有区分度: 极端分布 > 中度 > 均匀。"""
        uniform = {k: 0.2 for k in ["semantic", "causal", "spacetime", "generative", "trust"]}
        extreme = {"semantic": 0.96, "causal": 0.01, "spacetime": 0.01,
                   "generative": 0.01, "trust": 0.01}
        moderate = {"semantic": 0.4, "causal": 0.3, "spacetime": 0.15,
                    "generative": 0.1, "trust": 0.05}
        d_uniform = core.balance_deviation(uniform)
        d_moderate = core.balance_deviation(moderate)
        d_extreme = core.balance_deviation(extreme)
        assert d_uniform < d_moderate < d_extreme, (
            f"失衡度无区分度: uniform={d_uniform:.3f} moderate={d_moderate:.3f} extreme={d_extreme:.3f}"
        )
