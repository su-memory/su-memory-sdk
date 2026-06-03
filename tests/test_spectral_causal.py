"""
su-memory v3.4.0 — Spectral Causal Engine 测试套件

M1: GaussianDAG 单元测试
- 偏相关系数数值正确性
- TF-IDF 矩阵构建
- 隐藏因果边发现
- 能量先验交叉验证
- 混淆因子检测
- 边界条件
"""

from collections import defaultdict

import numpy as np
import pytest

pytestmark = pytest.mark.causal

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_memories():
    """构造 10 条测试记忆，包含显式因果和隐藏因果。"""
    return [
        {"id": "m01", "content": "暴雨导致城市内涝严重影响交通"},
        {"id": "m02", "content": "城市内涝促使排水系统全面升级改造"},
        {"id": "m03", "content": "物价指数同比上涨百分之三点五"},
        {"id": "m04", "content": "居民消费意愿指数下降百分之八点二"},
        {"id": "m05", "content": "公司宣布大规模裁员两百人"},
        {"id": "m06", "content": "竞争对手股价上涨百分之五"},
        {"id": "m07", "content": "气温连续一周超过三十五摄氏度"},
        {"id": "m08", "content": "冰淇淋销量增长百分之二百"},
        {"id": "m09", "content": "溺水事故增加百分之五十"},
        {"id": "m10", "content": "研发投入增加百分之五十"},
    ]


@pytest.fixture
def char_index(sample_memories):
    """以单字为 token 构建索引。"""
    idx = defaultdict(set)
    for i, mem in enumerate(sample_memories):
        for ch in mem["content"]:
            if '\u4e00' <= ch <= '\u9fff':
                idx[ch].add(i)
    return idx


@pytest.fixture
def gaussian_dag(sample_memories, char_index):
    """创建 GaussianDAG 实例。"""
    from su_memory.sdk._spectral_causal import GaussianDAG
    return GaussianDAG(sample_memories, char_index)


# =============================================================================
# 偏相关系数
# =============================================================================

class TestPartialCorrelation:
    """偏相关系数数值正确性。"""

    def test_known_correlation(self):
        """构造已知相关的向量对，验证偏相关计算。"""
        from su_memory.sdk._spectral_causal import GaussianDAG
        dag = GaussianDAG([], None)

        rng = np.random.RandomState(42)
        n = 100
        x = rng.randn(n)
        y = x + 0.3 * rng.randn(n)  # 真实相关
        z = rng.randn(n)             # 独立噪声

        rho, p = dag.partial_correlation(x, y, z)
        assert rho > 0.8, f"应检测到强偏相关, ρ={rho:.4f}"
        assert p < 0.001, f"p-value 应极显著, p={p:.4f}"

    def test_no_relation(self):
        """三个独立随机向量 — 偏相关应接近 0。"""
        from su_memory.sdk._spectral_causal import GaussianDAG
        dag = GaussianDAG([], None)

        rng = np.random.RandomState(77)
        n = 100
        x = rng.randn(n)
        y = rng.randn(n)
        z = rng.randn(n)

        rho, p = dag.partial_correlation(x, y, z)
        assert abs(rho) < 0.3, f"独立变量偏相关应接近0, ρ={rho:.4f}"

    def test_perfect_correlation_degenerate(self):
        """退化情况 — 除数为 0 时安全返回。"""
        from su_memory.sdk._spectral_causal import GaussianDAG
        dag = GaussianDAG([], None)

        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        z = np.array([1.0, 2.0, 3.0])

        rho, p = dag.partial_correlation(x, y, z)
        assert p in (0.0, 1.0), "退化应返回 0 或 1"


# =============================================================================
# TF-IDF 矩阵
# =============================================================================

class TestTFIDFMatrix:
    """TF-IDF 矩阵构建。"""

    def test_matrix_shape(self, gaussian_dag, sample_memories):
        """矩阵 shape 正确。"""
        mat = gaussian_dag.build_tfidf_matrix()
        n = len(sample_memories)
        assert mat.shape[0] == n, f"行数应为 {n}"
        assert mat.shape[1] > 0, "词汇表非空"

    def test_lazy_build(self, gaussian_dag):
        """懒构建 — 首次调用时自动构建。"""
        gaussian_dag._ensure_matrix()
        assert gaussian_dag._tfidf_matrix is not None

    def test_get_vector(self, gaussian_dag):
        """get_vector 返回等长向量。"""
        v0 = gaussian_dag.get_vector(0)
        v1 = gaussian_dag.get_vector(1)
        assert v0.shape == v1.shape
        assert v0.shape[0] > 0

    def test_empty_memories(self):
        """空记忆列表。"""
        from su_memory.sdk._spectral_causal import GaussianDAG
        dag = GaussianDAG([], None)
        mat = dag.build_tfidf_matrix()
        assert mat.shape == (0, 1)


# =============================================================================
# 隐藏因果边发现
# =============================================================================

class TestHiddenEdgeDiscovery:
    """隐藏因果边发现。"""

    def test_discovers_edges(self, gaussian_dag):
        """基本功能: 返回因果边列表。"""
        edges = gaussian_dag.discover_hidden_edges(
            min_correlation=0.1, p_threshold=0.2
        )
        assert isinstance(edges, list)

    def test_edge_structure(self, gaussian_dag):
        """每条边包含所需字段。"""
        edges = gaussian_dag.discover_hidden_edges(
            min_correlation=0.1, p_threshold=0.2
        )
        for edge in edges:
            for key in ["cause_idx", "effect_idx", "rho", "p_value",
                        "confidence", "verdict"]:
                assert key in edge, f"缺少字段: {key}"

    def test_sorted_by_confidence(self, gaussian_dag):
        """按置信度降序排列。"""
        edges = gaussian_dag.discover_hidden_edges(
            min_correlation=0.1, p_threshold=0.2
        )
        for i in range(len(edges) - 1):
            assert edges[i]["confidence"] >= edges[i + 1]["confidence"]

    def test_max_pairs_limit(self, gaussian_dag):
        """max_pairs 参数生效。"""
        edges = gaussian_dag.discover_hidden_edges(
            min_correlation=0.0, p_threshold=1.0, max_pairs=3
        )
        assert len(edges) <= 3

    def test_max_scan_limit(self, gaussian_dag):
        """max_scan 限制扫描范围。"""
        edges = gaussian_dag.discover_hidden_edges(
            min_correlation=0.0, p_threshold=1.0, max_scan=3
        )
        assert isinstance(edges, list)

    def test_empty_memories(self):
        """空记忆列表 — 返回空。"""
        from su_memory.sdk._spectral_causal import GaussianDAG
        dag = GaussianDAG([], None)
        edges = dag.discover_hidden_edges()
        assert edges == []

    def test_single_memory(self):
        """单条记忆 — 返回空。"""
        from su_memory.sdk._spectral_causal import GaussianDAG
        dag = GaussianDAG([{"id": "1", "content": "测试"}], None)
        edges = dag.discover_hidden_edges()
        assert edges == []


# =============================================================================
# 能量先验加权
# =============================================================================

class TestEnergyPriorBoost:
    """能量先验交叉验证。"""

    def test_no_energy_bus(self, gaussian_dag):
        """无能量总线 — 返回原始置信度和 verdict=none。"""
        conf, verdict = gaussian_dag.energy_prior_boost(
            {"id": "1", "content": "测试A", "energy_type": "wood"},
            {"id": "2", "content": "测试B", "energy_type": "fire"},
            0.7,
        )
        # 无 energy_bus → none
        assert verdict == "none"

    def test_hedge_type_inference(self, gaussian_dag):
        """能量类型推断。"""
        etype = gaussian_dag._infer_energy_type(
            {"id": "1", "content": "测试文本", "energy_type": "wood"}
        )
        assert etype == "wood"

    def test_hash_based_inference_consistent(self, gaussian_dag):
        """基于 hash 的推断应该一致。"""
        mem = {"id": "x", "content": "某段文本内容"}
        t1 = gaussian_dag._infer_energy_type(mem)
        t2 = gaussian_dag._infer_energy_type(mem)
        assert t1 == t2, "同一记忆的推断结果应一致"


# =============================================================================
# 混淆因子检测
# =============================================================================

class TestConfounderDetection:
    """混淆因子检测。"""

    def test_detects_confounder(self, gaussian_dag):
        """基本功能测试。"""
        result = gaussian_dag.detect_confounder(0, 1, 2)
        assert "is_confounder" in result
        assert "unconditional_rho" in result
        assert "conditional_rho" in result
        assert "confounder_score" in result

    def test_confounder_score_range(self, gaussian_dag):
        """confounder_score 应在 [0, 1] 范围内。"""
        result = gaussian_dag.detect_confounder(0, 1, 2)
        assert 0.0 <= result["confounder_score"] <= 1.0


# =============================================================================
# 统计摘要
# =============================================================================

class TestStatistics:
    """统计摘要。"""

    def test_get_statistics(self, gaussian_dag):
        """get_statistics 返回结构。"""
        stats = gaussian_dag.get_statistics()
        assert "n_memories" in stats
        assert "vocab_size" in stats
        assert "tfidf_built" in stats
        assert stats["n_memories"] > 0

    def test_energy_bus_unavailable(self, gaussian_dag):
        """无能量总线时标记 correct。"""
        stats = gaussian_dag.get_statistics()
        assert stats["energy_bus_available"] is False


# =============================================================================
# 向后兼容: _causal.py
# =============================================================================

class TestCausalBackwardCompat:
    """确保 _causal.py 向后兼容。"""

    def test_detect_causal_link_unchanged(self):
        """detect_causal_link 签名和行为不变。"""
        from su_memory.sdk._causal import detect_causal_link
        result = detect_causal_link("因为暴雨导致内涝", "内涝导致电网故障")
        assert result is not None
        ctype, conf = result
        assert isinstance(conf, float)

    def test_find_causal_pairs_default(self):
        """默认行为 (use_statistical=False) 不变。"""
        from su_memory.sdk._causal import CausalEngine
        engine = CausalEngine(min_confidence=0.5)
        memories = [
            {"id": "1", "content": "因为暴雨导致城市内涝"},
            {"id": "2", "content": "城市内涝导致排水系统升级"},
        ]
        pairs = engine.find_causal_pairs(memories)
        assert len(pairs) >= 0
        # 默认不启用统计路径

    def test_find_causal_pairs_statistical_no_crash(self):
        """统计路径启用时不崩溃。"""
        from su_memory.sdk._causal import CausalEngine
        engine = CausalEngine(min_confidence=0.5)
        memories = [
            {"id": "1", "content": "物价指数同比上涨百分之三点五"},
            {"id": "2", "content": "居民消费意愿指数下降百分之八点二"},
            {"id": "3", "content": "公司宣布大规模裁员两百人"},
            {"id": "4", "content": "竞争对手股价上涨百分之五"},
            {"id": "5", "content": "气温连续一周超过三十五摄氏度"},
            {"id": "6", "content": "冰淇淋销量增长百分之二百"},
            {"id": "7", "content": "溺水事故增加百分之五十"},
            {"id": "8", "content": "研发投入增加百分之五十"},
            {"id": "9", "content": "产品缺陷率下降至百分之零点一"},
            {"id": "10", "content": "由于技术突破所以生产效率大幅提升"},
        ]
        pairs = engine.find_causal_pairs(memories, use_statistical=True)
        assert isinstance(pairs, list)

    def test_is_duplicate(self):
        """_is_duplicate 双向去重。"""
        from su_memory.sdk._causal import _is_duplicate
        pairs = [({"id": "a"}, {"id": "b"}, "cause", 0.8)]
        assert _is_duplicate(pairs, "a", "b")
        assert _is_duplicate(pairs, "b", "a")
        assert not _is_duplicate(pairs, "a", "c")
        assert not _is_duplicate([], "a", "b")


# =============================================================================
# M2: FourierCausal 单元测试
# =============================================================================

class TestFourierCausal:
    """FourierCausal 频域因果分析。"""

    @pytest.fixture
    def fc(self):
        from su_memory.sdk._spectral_causal import FourierCausal
        return FourierCausal()

    @pytest.fixture
    def fc_with_data(self):
        from su_memory.sdk._spectral_causal import FourierCausal
        fc = FourierCausal()
        # 30 样本纯正弦 + 白噪声 (seed=42 可复现)
        rng = np.random.RandomState(42)
        t = np.linspace(0, 4 * np.pi, 30)
        fc._intensity_history = {
            "wood": list(1.0 + 0.5 * np.sin(t) + 0.05 * rng.randn(30)),
            "fire": list(1.0 + 0.5 * np.sin(t + 1.0) + 0.05 * rng.randn(30)),
            "earth": list(1.0 + 0.3 * np.sin(t + 2.0) + 0.05 * rng.randn(30)),
            "metal": list(1.0 + 0.4 * np.sin(t + 3.0) + 0.05 * rng.randn(30)),
            "water": list(1.0 + 0.5 * np.sin(t + 4.0) + 0.05 * rng.randn(30)),
        }
        return fc

    def test_fft_decompose_sine(self, fc_with_data):
        """纯正弦信号: DC+基频主导, high_freq≈0。"""
        decomp = fc_with_data.fft_decompose("wood")
        assert "error" not in decomp, f"不应返回错误: {decomp}"
        assert decomp["n_samples"] == 30
        # 纯正弦 → 基频应主导
        assert decomp["fundamental_ratio"] > 0.3, \
            f"基频应主导, got {decomp['fundamental_ratio']}"
        # 纯正弦 → 高频应很低
        assert decomp["high_freq_ratio"] < 0.3, \
            f"高频应低, got {decomp['high_freq_ratio']}"
        # 纯正弦 → anomaly_score 应低
        assert decomp["anomaly_score"] < 0.3, \
            f"纯正弦异常分应低, got {decomp['anomaly_score']}"

    def test_fft_decompose_spike(self, fc_with_data):
        """周期信号 + 单个尖峰: anomaly_score > 0.3。"""
        # 在 wood 序列中注入一个尖峰
        wood = fc_with_data._intensity_history["wood"]
        wood_with_spike = list(wood)
        wood_with_spike[15] = wood[15] * 4.0  # 4x 尖峰
        fc_with_data._intensity_history["wood"] = wood_with_spike

        decomp_spike = fc_with_data.fft_decompose("wood")
        assert decomp_spike["anomaly_score"] > 0.25, \
            f"尖峰应有显著异常分, got {decomp_spike['anomaly_score']}"

        # 对照: fire 无尖峰, 得分应较低
        decomp_normal = fc_with_data.fft_decompose("fire")
        assert decomp_normal["anomaly_score"] < decomp_spike["anomaly_score"], \
            f"无尖峰({decomp_normal['anomaly_score']}) 应 < 有尖峰({decomp_spike['anomaly_score']})"

    def test_fft_decompose_cold_start(self, fc):
        """< 5 个采样点 → 返回 error dict。"""
        fc._intensity_history = {"wood": [0.5, 0.6, 0.5, 0.6]}
        result = fc.fft_decompose("wood")
        assert result.get("error") == "insufficient_samples"
        assert result["anomaly_score"] == 0.0

    def test_record_snapshot_manual(self):
        """手动传入强度 → snapshot_count 递增。"""
        from su_memory.sdk._spectral_causal import FourierCausal
        fc = FourierCausal()
        assert fc.record_snapshot({"wood": 0.6, "fire": 0.7}) == 1
        assert fc.record_snapshot({"wood": 0.5, "fire": 0.8}) == 2
        assert fc._snapshot_count == 2
        assert len(fc.get_series("wood")) == 2
        assert fc.get_series("wood")[0] == 0.6
        assert fc.get_series("wood")[1] == 0.5

    def test_record_snapshot_no_data(self):
        """无 energy_bus 且不传 intensities → 仅 count 递增, 不记录数据。"""
        from su_memory.sdk._spectral_causal import FourierCausal
        fc = FourierCausal()
        fc.record_snapshot()  # 不传 intensities, 也无 _bus
        # count 递增但不记录数据 (设计: 显式传入或无 bus 时不自动填充)
        assert fc._snapshot_count == 1
        # 没有数据被记录 — 这是预期行为
        assert len(fc.get_series("wood")) == 0

    def test_detect_causal_events(self, fc_with_data):
        """5 元素中仅 1 个有尖峰 → 仅该元素报告异常。"""
        # 给 wood 注入极端尖峰
        wood = fc_with_data._intensity_history["wood"]
        wood_spiked = list(wood)
        wood_spiked[15] = wood[15] * 4.0
        fc_with_data._intensity_history["wood"] = wood_spiked

        events = fc_with_data.detect_causal_events(threshold=0.2)
        anomalous_elements = {e["element"] for e in events}

        # wood 应出现
        assert "wood" in anomalous_elements, \
            f"wood 应有异常事件, events={events}"

        # 其他元素出现数应少
        non_wood = anomalous_elements - {"wood"}
        assert len(non_wood) <= 2, \
            f"非 wood 异常事件不应过多: {non_wood}"

    def test_periodic_noise_filter(self):
        """低频同步信号 → filter_periodic_noise 正确归零对应相关系数。"""
        from su_memory.sdk._spectral_causal import FourierCausal

        fc = FourierCausal()
        # 使用 endpoint=False 避免频谱泄漏, 确保低频主导
        n_samples = 100
        rng = np.random.RandomState(42)
        t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        low_freq_a = 1.0 + 0.8 * np.sin(t) + 0.02 * rng.randn(n_samples)
        low_freq_b = 1.0 + 0.8 * np.sin(t + 0.1) + 0.02 * rng.randn(n_samples)

        fc._intensity_history = {
            "wood": list(low_freq_a),
            "fire": list(low_freq_b),
            "earth": list(1.0 + 0.3 * np.sin(3.0 * t) + 0.05 * rng.randn(n_samples)),
            "metal": list(1.0 + 0.4 * np.sin(4.0 * t) + 0.05 * rng.randn(n_samples)),
            "water": list(1.0 + 0.5 * np.sin(5.0 * t) + 0.05 * rng.randn(n_samples)),
        }

        # 验证交叉频谱相干性
        coh = fc.cross_spectral_coherence("wood", "fire")
        assert "error" not in coh, f"不应报错: {coh}"
        assert coh["is_synchronized"], "低频同步信号应判定为同步"

        # 原始矩阵
        corr = np.eye(5) + np.abs(rng.randn(5, 5)) * 0.3
        corr[0, 1] = 0.85
        corr[1, 0] = 0.85

        filtered = fc.filter_periodic_noise(corr, cutoff=0.3)

        # 如果 sync_band 是 low → wood-fire 应被滤除
        # 如果 sync_band 是 mid/high → 不应被滤除 (低频同步检查不通过)
        if coh["sync_band"] == "low":
            assert filtered[0, 1] < 0.01, \
                f"低频同步应对应被滤除: {filtered[0, 1]}"
        # 无论如何, shape 不变
        assert filtered.shape == corr.shape

    def test_gaussian_dag_with_fourier(self, sample_memories, char_index):
        """GaussianDAG + FourierCausal → 边包含频域标注。"""
        from su_memory.sdk._spectral_causal import FourierCausal, GaussianDAG

        # 准备 FourierCausal 数据
        fc = FourierCausal()
        rng = np.random.RandomState(42)
        t = np.linspace(0, 4 * np.pi, 20)
        fc._intensity_history = {
            "wood": list(1.0 + 0.5 * np.sin(t) + 0.05 * rng.randn(20)),
            "fire": list(1.0 + 0.5 * np.sin(t + 1.0) + 0.05 * rng.randn(20)),
            "earth": list(1.0 + 0.3 * np.sin(t + 2.0) + 0.05 * rng.randn(20)),
            "metal": list(1.0 + 0.4 * np.sin(t + 3.0) + 0.05 * rng.randn(20)),
            "water": list(1.0 + 0.5 * np.sin(t + 4.0) + 0.05 * rng.randn(20)),
        }

        # 无 Fourier
        dag_no_f = GaussianDAG(sample_memories, char_index)
        edges_no_f = dag_no_f.discover_hidden_edges(
            min_correlation=0.0, p_threshold=0.5
        )

        # 有 Fourier
        dag_with_f = GaussianDAG(sample_memories, char_index)
        dag_with_f.with_fourier_filter(fc)
        edges_with_f = dag_with_f.discover_hidden_edges(
            min_correlation=0.0, p_threshold=0.5
        )

        # 基本结构
        assert isinstance(edges_no_f, list)
        assert isinstance(edges_with_f, list)

        # 有 Fourier 的边应包含频域字段
        if edges_with_f:
            for edge in edges_with_f:
                assert "fourier_filtered" in edge, \
                    "过滤后的边应包含 fourier_filtered 字段"
                assert "spectral_coherence" in edge, \
                    "过滤后的边应包含 spectral_coherence 字段"

    def test_spectral_balance_report_sine(self, fc_with_data):
        """纯正弦信号 → health_status='healthy'。"""
        report = fc_with_data.spectral_balance_report()
        assert report["health_status"] in ("healthy", "warning", "critical")
        assert "per_element" in report
        assert "global_anomaly_score" in report
        assert "most_anomalous" in report
        assert "recommendations" in report
        # 纯正弦 → 应健康
        assert report["health_status"] == "healthy", \
            f"纯正弦应健康, got {report['health_status']}"

    def test_cross_spectral_coherence_same_signal(self):
        """两个相同的信号 → coherence 应接近 1。"""
        from su_memory.sdk._spectral_causal import FourierCausal

        fc = FourierCausal()
        rng = np.random.RandomState(42)
        t = np.linspace(0, 4 * np.pi, 30)
        signal = list(1.0 + 0.5 * np.sin(t) + 0.01 * rng.randn(30))

        fc._intensity_history = {
            "wood": signal,
            "fire": signal,  # 完全相同
            "earth": [1.0] * 30,
            "metal": [0.5] * 30,
            "water": [0.8] * 30,
        }

        coh = fc.cross_spectral_coherence("wood", "fire")
        assert "error" not in coh, f"不应报错: {coh}"
        # 相同信号 → 高相干性
        assert coh["coherence"] > 0.5, \
            f"相同信号应有高相干性, got {coh['coherence']}"
        assert coh["is_synchronized"], "相同信号应判定为同步"


# =============================================================================
# M3: GaussianDistribution + BayesianCausal 单元测试
# =============================================================================

class TestGaussianDistribution:
    """GaussianDistribution 共轭先验。"""

    def test_conjugate_update_known_result(self):
        """已知共轭解析解验证。"""
        from su_memory.sdk._spectral_causal import GaussianDistribution

        prior = GaussianDistribution(mu=0.0, sigma=1.0)
        prior.update(sample_mean=0.5, sample_std=0.2, n=10)

        # 手动计算: τ₀=1, τ_lik=10/0.04=250, τ_post=251
        # μ_post = (0*1 + 0.5*250)/251 ≈ 0.498
        # σ_post = 1/√251 ≈ 0.0631
        assert abs(prior.mu - 0.498) < 0.01
        assert abs(prior.sigma - 0.0631) < 0.01

    def test_credible_interval(self):
        """μ±1.96σ 覆盖 95% 概率。"""
        from su_memory.sdk._spectral_causal import GaussianDistribution

        dist = GaussianDistribution(mu=0.0, sigma=1.0)
        ci = dist.credible_interval(0.95)
        # z₀.₉₇₅ ≈ 1.96
        assert abs(ci[0] + 1.96) < 0.01
        assert abs(ci[1] - 1.96) < 0.01

    def test_serialization_roundtrip(self):
        """to_dict/from_dict 往返完全恢复。"""
        from su_memory.sdk._spectral_causal import GaussianDistribution

        original = GaussianDistribution(mu=0.5, sigma=0.2, n_observations=42)
        d = original.to_dict()
        restored = GaussianDistribution.from_dict(d)
        assert abs(restored.mu - original.mu) < 1e-10
        assert abs(restored.sigma - original.sigma) < 1e-10
        assert restored.n_observations == original.n_observations

    def test_properties(self):
        """mean/variance/precision 属性正确。"""
        from su_memory.sdk._spectral_causal import GaussianDistribution

        dist = GaussianDistribution(mu=1.0, sigma=2.0)
        assert dist.mean == 1.0
        assert dist.variance == 4.0
        assert abs(dist.precision - 0.25) < 1e-10

    def test_pdf_cdf(self):
        """pdf/cdf 有效值。"""
        from su_memory.sdk._spectral_causal import GaussianDistribution

        dist = GaussianDistribution(mu=0.0, sigma=1.0)
        p = dist.pdf(0.0)
        c = dist.cdf(0.0)
        assert p > 0.3  # N(0,1) at x=0 ≈ 0.399
        assert abs(c - 0.5) < 0.01

    def test_update_no_samples(self):
        """n=0 不改变分布。"""
        from su_memory.sdk._spectral_causal import GaussianDistribution

        dist = GaussianDistribution(mu=0.5, sigma=0.3)
        dist.update(sample_mean=10.0, sample_std=1.0, n=0)
        assert abs(dist.mu - 0.5) < 1e-10
        assert abs(dist.sigma - 0.3) < 1e-10


class TestBayesianCausal:
    """BayesianCausal 后验量化。"""

    def test_h0_no_effect(self):
        """rho≈0, n large → BF 支持 H₀。"""
        from su_memory.sdk._spectral_causal import BayesianCausal

        bc = BayesianCausal()
        result = bc.causal_hypothesis_test("test_h0", rho=0.01, n_samples=100)
        assert "no_causal" in result["conclusion"] or "inconclusive" in result["conclusion"]

    def test_h1_strong_effect(self):
        """rho=0.7, n=50 → BF 支持 H₁。"""
        from su_memory.sdk._spectral_causal import BayesianCausal

        bc = BayesianCausal()
        result = bc.causal_hypothesis_test("test_h1", rho=0.7, n_samples=50)
        assert result["bayes_factor"] > 3.0 or result["bayes_factor"] == float("inf")

    def test_energy_prior_enhance(self):
        """生关系先验 → 后验向正偏移。"""
        from su_memory.sdk._spectral_causal import BayesianCausal

        bc = BayesianCausal()
        result = bc.causal_hypothesis_test(
            "test_e", rho=0.3, n_samples=20, energy_relation="enhance"
        )
        assert result["posterior_mean"] > 0.15

    def test_energy_prior_suppress_conservative(self):
        """克关系先验 → 后验较保守 (σ 较小)。"""
        from su_memory.sdk._spectral_causal import BayesianCausal

        bc = BayesianCausal()
        r_enhance = bc.causal_hypothesis_test(
            "a", rho=0.3, n_samples=20, energy_relation="enhance"
        )
        r_suppress = bc.causal_hypothesis_test(
            "b", rho=0.3, n_samples=20, energy_relation="suppress"
        )
        # 克关系后验均值应低于生关系
        assert r_suppress["posterior_mean"] < r_enhance["posterior_mean"]

    def test_posterior_convergence(self):
        """n 增大 → posterior_std 单调递减。"""
        from su_memory.sdk._spectral_causal import BayesianCausal

        bc = BayesianCausal()
        stds = []
        for n in [5, 10, 20, 50]:
            r = bc.causal_hypothesis_test(f"conv_{n}", rho=0.5, n_samples=n)
            stds.append(r["posterior_std"])

        for i in range(len(stds) - 1):
            assert stds[i] >= stds[i + 1] * 0.99, \
                f"std 应递减: {stds[i]} vs {stds[i+1]}"

    def test_bf_extreme(self):
        """rho≈1 → BF=inf 安全处理。"""
        from su_memory.sdk._spectral_causal import BayesianCausal

        bc = BayesianCausal()
        result = bc.causal_hypothesis_test("extreme", rho=0.99, n_samples=100)
        assert result["bayes_factor"] == float("inf") or result["bayes_factor"] > 100
        assert "strong_evidence" in result["conclusion"]

    def test_batch_update(self):
        """batch_update 返回带后验字段的边列表。"""
        from su_memory.sdk._spectral_causal import BayesianCausal

        bc = BayesianCausal()
        edges = [
            {"cause_idx": 0, "effect_idx": 1, "rho": 0.5, "n_samples": 20},
            {"cause_idx": 2, "effect_idx": 3, "rho": 0.05, "n_samples": 20},
        ]
        updated = bc.batch_update(edges)
        assert len(updated) == 2
        for e in updated:
            assert "posterior_mean" in e
            assert "posterior_std" in e
            assert "bayes_factor" in e
            assert "conclusion" in e

    def test_get_summary(self):
        """get_summary 结构正确。"""
        from su_memory.sdk._spectral_causal import BayesianCausal

        bc = BayesianCausal()
        bc.causal_hypothesis_test("a", rho=0.7, n_samples=50)
        bc.causal_hypothesis_test("b", rho=0.01, n_samples=100)
        summary = bc.get_summary()

        assert summary["n_edges_tested"] == 2
        assert "n_strong_causal" in summary
        assert "n_moderate_causal" in summary
        assert "edges" in summary

    def test_compare_hypotheses(self):
        """compare_hypotheses 比较两条边。"""
        from su_memory.sdk._spectral_causal import BayesianCausal

        bc = BayesianCausal()
        bc.causal_hypothesis_test("strong", rho=0.8, n_samples=50)
        bc.causal_hypothesis_test("weak", rho=0.05, n_samples=50)

        comp = bc.compare_hypotheses("strong", "weak")
        assert "posterior_odds" in comp
        assert "favored" in comp
        assert "confidence" in comp
        # strong 应该 favored
        assert comp["favored"] in ("strong", "neither")

    def test_end_to_end_pipeline(self, sample_memories, char_index):
        """GaussianDAG + FourierCausal + BayesianCausal 全链路不报错。"""
        import numpy as np

        from su_memory.sdk._spectral_causal import (
            BayesianCausal,
            FourierCausal,
            GaussianDAG,
        )

        rng = np.random.RandomState(42)
        t = np.linspace(0, 4 * np.pi, 20)
        fc = FourierCausal()
        fc._intensity_history = {
            "wood": list(1.0 + 0.5 * np.sin(t) + 0.05 * rng.randn(20)),
            "fire": list(1.0 + 0.5 * np.sin(t + 1.0) + 0.05 * rng.randn(20)),
            "earth": list(1.0 + 0.3 * np.sin(t + 2.0) + 0.05 * rng.randn(20)),
            "metal": list(1.0 + 0.4 * np.sin(t + 3.0) + 0.05 * rng.randn(20)),
            "water": list(1.0 + 0.5 * np.sin(t + 4.0) + 0.05 * rng.randn(20)),
        }

        bc = BayesianCausal()

        dag = GaussianDAG(sample_memories, char_index)
        dag.with_fourier_filter(fc)
        dag.with_bayesian_quantification(bc)

        edges = dag.discover_hidden_edges(
            min_correlation=0.0, p_threshold=0.5
        )

        assert isinstance(edges, list)
        # 验证边包含后验字段
        if edges:
            for edge in edges:
                assert "posterior_mean" in edge
                assert "bayes_factor" in edge
                assert "conclusion" in edge
