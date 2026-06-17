"""
su-memory v3.6.0 — Spectral Causal Engine

core-centric · 4-layer quantization · from syntactic to mathematically verifiable causality

包含三个核心类:
- GaussianDAG:     偏相关系数因果发现 + 能量先验交叉验证
- FourierCausal:   频域因果分析 (v3.4.0-p1)
- BayesianCausal:  贝叶斯后验量化 (v3.4.0-p2)

v3.6.0 新增:
- GaussianDAG.with_parametric_prior(): 参数化模型先验矩阵注入
- discover_hidden_edges(): 三路径融合 (统计 0.5 + reflection 0.3 + parametric 0.2)

设计原则:
- zero-intrusion core — EnergyBus / EnergyCore untouched
- 零新依赖 — scipy + numpy 已有
- 向后兼容 — 因果 API 签名不变
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import norm, pearsonr

logger = logging.getLogger(__name__)

# =============================================================================
# M1: GaussianDAG — 偏相关系数因果发现 + 能量先验交叉验证
# =============================================================================


class GaussianDAG:
    """
    基于高斯假设的偏相关因果发现。

    核心数学:
      ρ_{XY|Z} = (ρ_{XY} - ρ_{XZ}·ρ_{YZ}) / sqrt((1-ρ_{XZ}²)(1-ρ_{YZ}²))
      ρ_{XY|Z} = 0 ⟺ X ⟂ Y | Z  (高斯假设下条件独立性)

    能量先验交叉验证 — 三重判定:
      - 统计+能量 都指向 → confirmed  (confidence × 1.2)
      - 统计有、能量无 → novel       (新发现因果)
      - 统计无、能量有 → suppressed  (克关系压制)
      - 两者都无      → none
    """

    # Energy Types能量原始名称列表
    FIVE_ELEMENTS = ["wood", "fire", "earth", "metal", "water"]

    def __init__(
        self,
        memories: list[dict],
        tfidf_index: dict[str, set] | None = None,
        energy_bus=None,
    ):
        """
        Args:
            memories: 记忆列表，每项含 "id" 和 "content"
            tfidf_index: TF-IDF 倒排索引用作向量化基础 (可选)
            energy_bus: EnergyBus 实例 (可选，用于能量先验交叉验证)
        """
        self.memories = memories
        self._energy_bus = energy_bus

        # ── TF-IDF 词汇表 ──
        if tfidf_index and len(tfidf_index) > 0:
            self._vocab = sorted(tfidf_index.keys())
        else:
            # 从记忆内容中提取词汇（支持中英文混合）
            vocab_set = set()
            for mem in memories:
                content = mem.get("content", "")
                # 中文字符提取
                for ch in content:
                    if '\u4e00' <= ch <= '\u9fff':
                        vocab_set.add(ch)
                # 英文单词提取 (至少 2 个字符)
                import re
                words = re.findall(r'[a-zA-Z]{2,}', content.lower())
                vocab_set.update(words)
            self._vocab = sorted(vocab_set)

        self._vocab_map = {w: i for i, w in enumerate(self._vocab)}
        self._tfidf_matrix: np.ndarray | None = None
        self._corr_matrix: np.ndarray | None = None
        self._n_effective: int = len(memories)

        # ── 能量先验统计 ──
        self._energy_stats: dict[str, int] = defaultdict(int)

        # ── v3.5.0: Reflection QA 因果先验矩阵 ──
        self._reflection_prior: np.ndarray | None = None

        # ── v3.6.0: 参数化模型先验矩阵 ──
        self._parametric_prior: np.ndarray | None = None

    # -----------------------------------------------------------------
    # v3.5.0: Reflection Prior 注入
    # -----------------------------------------------------------------

    def with_reflection_prior(self, prior_matrix: np.ndarray):
        """
        注入 Reflection QA 合成的因果先验矩阵。

        先验矩阵 P[i][j] ∈ [0, 1]:
        - 0 = 无先验 → 使用原始偏相关
        - 1 = 强因果 → 高权重优先
        """
        self._reflection_prior = np.asarray(prior_matrix, dtype=np.float32)

    # -----------------------------------------------------------------
    # v3.6.0: Parametric Prior 注入
    # -----------------------------------------------------------------

    def with_parametric_prior(self, prior):
        """
        注入参数化模型 (QLoRA) 的因果先验。

        可从 TopologicalEnergyMatrix 或任意 np.ndarray 输入。
        参数化先验在三路径融合中权重为 0.2。

        Args:
            prior: TopologicalEnergyMatrix 或 np.ndarray
        """
        if hasattr(prior, 'to_flat_vector'):
            self._parametric_prior = prior.to_flat_vector().reshape(5, 5)
        elif hasattr(prior, 'matrix'):
            self._parametric_prior = np.asarray(prior.matrix, dtype=np.float32)
        else:
            self._parametric_prior = np.asarray(prior, dtype=np.float32)

    # -----------------------------------------------------------------
    # TF-IDF 矩阵构建
    # -----------------------------------------------------------------

    def build_tfidf_matrix(self) -> np.ndarray:
        """
        将记忆集合转换为 TF-IDF 稀疏矩阵。

        使用与 SuMemoryLite 一致的中文 n-gram (2-3 字符) 分词策略。
        n_memories × n_vocab。

        Returns:
            np.ndarray, shape=(n_memories, n_vocab)
        """
        n = len(self.memories)
        d = len(self._vocab)
        if n == 0 or d == 0:
            self._tfidf_matrix = np.zeros((n, max(d, 1)), dtype=np.float32)
            return self._tfidf_matrix

        # 统计每篇记忆的词频
        mat = lil_matrix((n, d), dtype=np.float32)
        doc_freq = np.zeros(d, dtype=np.float32)

        for i, mem in enumerate(self.memories):
            content = mem.get("content", "")
            # 中文 n-gram (2-3 char) + 英文单词 分词
            tokens = set()
            for k in [2, 3]:
                for j in range(len(content) - k + 1):
                    tokens.add(content[j:j + k])
            # 中文字符
            for ch in content:
                if '\u4e00' <= ch <= '\u9fff':
                    tokens.add(ch)
            # 英文单词 (至少 2 个字符)
            import re
            words = re.findall(r'[a-zA-Z]{2,}', content.lower())
            tokens.update(words)

            for token in tokens:
                if token in self._vocab_map:
                    col = self._vocab_map[token]
                    mat[i, col] += 1

        # 转为 dense 并计算 TF-IDF
        tf = mat.toarray()
        doc_freq = np.count_nonzero(tf, axis=0)
        df = np.maximum(doc_freq, 1)
        idf = np.log(n / df) + 1.0  # smooth IDF
        tfidf = tf.astype(np.float32) * idf.astype(np.float32)

        # L2 归一化
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._tfidf_matrix = tfidf / norms
        return self._tfidf_matrix

    def _ensure_matrix(self):
        """确保 TF-IDF 矩阵已构建。"""
        if self._tfidf_matrix is None:
            self.build_tfidf_matrix()

    def get_vector(self, memory_idx: int) -> np.ndarray:
        """获取记忆的 TF-IDF 向量。"""
        self._ensure_matrix()
        return self._tfidf_matrix[memory_idx].copy()

    # -----------------------------------------------------------------
    # 偏相关系数
    # -----------------------------------------------------------------

    def partial_correlation(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[float, float]:
        """
        计算偏相关系数 ρ_{XY|Z} + Fisher z-transform p-value。

        ρ_{XY|Z} = (ρ_{XY} - ρ_{XZ}·ρ_{YZ}) / sqrt((1-ρ_{XZ}²)(1-ρ_{YZ}²))

        Args:
            x, y: 两个变量的观测向量
            z: 条件变量向量

        Returns:
            (rho, p_value) — p < 0.05 表示在控制 Z 后 X 与 Y 仍显著相关
        """
        # 简单相关系数
        r_xy, _ = pearsonr(x, y)
        r_xz, _ = pearsonr(x, z)
        r_yz, _ = pearsonr(y, z)

        # 偏相关系数
        num = r_xy - r_xz * r_yz
        denom = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))

        if abs(denom) < 1e-10:
            return (0.0, 1.0)  # 退化情况

        rho = np.clip(num / denom, -1.0, 1.0)

        # Fisher z-transform → p-value
        n = len(x)
        if abs(rho) >= 1.0 - 1e-10:
            return (rho, 0.0)  # 完美相关

        z_stat = 0.5 * np.log((1 + rho) / (1 - rho)) * np.sqrt(n - 3)
        p_value = 2.0 * (1.0 - norm.cdf(abs(z_stat)))

        return (float(rho), float(p_value))

    def _partial_corr_vec(
        self, x: np.ndarray, y: np.ndarray, z_vec: np.ndarray
    ) -> tuple[float, float]:
        """
        以全局均值向量作为条件集的代理 (简化版)。
        当无法穷举所有条件组合时使用。
        """
        return self.partial_correlation(x, y, z_vec)

    # -----------------------------------------------------------------
    # 能量先验加权
    # -----------------------------------------------------------------

    def energy_prior_boost(
        self, mem_a: dict, mem_b: dict, base_conf: float
    ) -> tuple[float, str]:
        """
        能量先验交叉验证。

        根据记忆中隐含的能量类型 (或内容推断)，
        使用Energy Types enhance/suppress关系调整因果置信度。

        Args:
            mem_a: 记忆 A
            mem_b: 记忆 B
            base_conf: 统计偏相关给出的基础置信度

        Returns:
            (adjusted_confidence, verdict)
            verdict: "confirmed" | "novel" | "suppressed" | "none"
        """
        if self._energy_bus is None:
            return (base_conf, "none")

        # 从记忆中提取或推断能量类型
        etype_a = self._infer_energy_type(mem_a)
        etype_b = self._infer_energy_type(mem_b)

        if etype_a is None or etype_b is None:
            return (base_conf, "none")

        try:
            from su_memory._sys._energy_relations import (
                RelationType,
                analyze_relation,
            )
            relation = analyze_relation(etype_a, etype_b)

            if relation.relation == RelationType.ENHANCE:
                if base_conf > 0.3:
                    return (min(base_conf * 1.2, 0.98), "confirmed")
                return (base_conf * 0.8, "suppressed")
            elif relation.relation == RelationType.SUPPRESS:
                return (base_conf * 0.8, "suppressed")
            elif relation.relation in (RelationType.SAME,):
                return (base_conf * 1.1, "confirmed")
            else:
                return (base_conf, "novel")
        except ImportError:
            return (base_conf, "none")

    def _infer_energy_type(self, mem: dict) -> str | None:
        """从记忆中推断能量类型。"""
        # 1. 直接标注
        etype = mem.get("energy_type")
        if etype and etype in self.FIVE_ELEMENTS:
            return etype

        # 2. 从 energy_bus 节点查找
        if self._energy_bus:
            mem_id = mem.get("id", "")
            for etype in self.FIVE_ELEMENTS:
                for prefix in ["element_", "wuxing_"]:
                    try:
                        node = self._energy_bus.get_node(f"{prefix}{etype}")
                        if node and mem_id in str(getattr(node, 'memory_ids', [])):
                            return etype
                    except Exception as e:
                        logger.debug(
                            "_infer_energy_type: failed to query energy_bus node %s%s: %s",
                            prefix, etype, e,
                        )

        # 3. 基于内容的启发式推断 (hash 映射保证一致性)
        content = mem.get("content", "")
        if content:
            idx = hash(content) % 5
            return self.FIVE_ELEMENTS[idx]

        return None

    # -----------------------------------------------------------------
    # 隐藏因果边发现
    # -----------------------------------------------------------------

    def discover_hidden_edges(
        self,
        min_correlation: float = 0.3,
        p_threshold: float = 0.05,
        max_pairs: int = 200,
        max_scan: int = 50,
    ) -> list[dict]:
        """
        三重验证因果发现:

        1. 偏相关系数 (统计) — 不需要关键词标记的因果检测
        2. 能量先验交叉验证 (结构) — 生克关系交叉验证
        3. p-value 显著性 (推断) — Fisher z-transform

        Args:
            min_correlation: 最低偏相关系数阈值
            p_threshold: p-value 阈值
            max_pairs: 最大返回因果对数
            max_scan: 最大扫描的记忆数 (防止 O(n²) 爆炸, R1 缓解)

        Returns:
            [{"cause_idx", "effect_idx", "rho", "p_value",
              "confidence", "verdict", "energy_relation"}, ...]
            按 confidence 降序排列
        """
        self._ensure_matrix()
        mat = self._tfidf_matrix
        n = mat.shape[0]

        if n < 2:
            return []

        # 限制扫描范围 (R1 缓解)
        scan_n = min(n, max_scan)
        edges: list[dict] = []

        # 全局均值向量作为条件集 Z 的代理
        z_global = mat[:scan_n].mean(axis=0)

        for i in range(scan_n):
            x = mat[i]
            for j in range(i + 1, scan_n):
                y = mat[j]

                # 偏相关系数
                rho, p_value = self._partial_corr_vec(x, y, z_global)

                if p_value >= p_threshold or abs(rho) < min_correlation:
                    continue

                # 基础置信度
                base_conf = min(abs(rho), 0.95)

                # 能量先验交叉验证
                conf, verdict = self.energy_prior_boost(
                    self.memories[i], self.memories[j], base_conf
                )

                # 尝试获取能量关系描述
                energy_rel = None
                mem_etype_a = self._infer_energy_type(self.memories[i])
                mem_etype_b = self._infer_energy_type(self.memories[j])
                if mem_etype_a and mem_etype_b:
                    try:
                        from su_memory._sys._energy_relations import (
                            analyze_relation,
                        )
                        rel = analyze_relation(mem_etype_a, mem_etype_b)
                        energy_rel = rel.relation.value
                    except ImportError:
                        pass

                edges.append({
                    "cause_idx": i if rho > 0 else j,
                    "effect_idx": j if rho > 0 else i,
                    "rho": round(rho, 4),
                    "p_value": round(p_value, 4),
                    "confidence": round(conf, 4),
                    "verdict": verdict,
                    "energy_relation": energy_rel,
                })

        # 按置信度降序
        edges.sort(key=lambda e: e["confidence"], reverse=True)
        edges = edges[:max_pairs]

        # ── v3.5.0: Reflection Prior 增强 ──
        if self._reflection_prior is not None:
            n_prior = self._reflection_prior.shape[0]
            for edge in edges:
                i, j = edge["cause_idx"], edge["effect_idx"]
                if i < n_prior and j < n_prior:
                    prior_val = self._reflection_prior[i, j]
                    if prior_val > 0.1:
                        # 先验与偏相关加权融合
                        edge["confidence"] = round(
                            edge["confidence"] * 0.7 + prior_val * 0.3, 4
                        )
                        edge["reflection_boosted"] = True
                        edge["reflection_prior"] = round(float(prior_val), 4)

        # ── v3.6.0: 参数化先验增强 (第三路径融合) ──
        if self._parametric_prior is not None:
            n_param = min(self._parametric_prior.shape[0], 5)
            for edge in edges:
                # 基于内容 hash 映射到 5×5 矩阵
                i_mod = edge["cause_idx"] % n_param
                j_mod = edge["effect_idx"] % n_param
                param_val = float(self._parametric_prior[i_mod, j_mod])
                if param_val > 0.1:
                    # 三路径加权融合: 统计 0.5 + reflection 0.3 + parametric 0.2
                    current_conf = edge.get("confidence", 0.5)
                    reflection_weight = 0.3 if edge.get("reflection_boosted") else 0.0
                    stat_weight = 1.0 - reflection_weight - 0.2
                    edge["confidence"] = round(
                        current_conf * stat_weight
                        + edge.get("reflection_prior", 0.0) * reflection_weight
                        + param_val * 0.2,
                        4,
                    )
                    edge["parametric_boosted"] = True
                    edge["parametric_prior"] = round(param_val, 4)

        # Fourier 频域过滤 (如果启用)
        if self._fourier is not None:
            edges = self._fourier.filter_periodic_edges(edges, self)

        # Bayesian 后验量化 (如果启用)
        if self._bayesian is not None:
            for edge in edges:
                eid = f"{edge['cause_idx']}_{edge['effect_idx']}"
                posterior = self._bayesian.causal_hypothesis_test(
                    eid, edge["rho"], self._n_effective,
                    edge.get("energy_relation"),
                )
                edge["posterior_mean"] = posterior["posterior_mean"]
                edge["posterior_std"] = posterior["posterior_std"]
                edge["credible_interval_95"] = posterior["credible_interval_95"]
                edge["bayes_factor"] = posterior["bayes_factor"]
                edge["conclusion"] = posterior["conclusion"]

        return edges

    # -----------------------------------------------------------------
    # 混淆因子检测
    # -----------------------------------------------------------------

    def detect_confounder(
        self, mem_a_idx: int, mem_b_idx: int, candidate_z_idx: int
    ) -> dict:
        """
        检测候选变量 Z 是否为 X↔Y 的混淆因子。

        逻辑: 如果 ρ_{XY} ≠ 0 但 ρ_{XY|Z} ≈ 0，则 Z 是混淆因子。
        (经典案例: "冰淇淋销量"↔"溺水事故" 由"气温"驱动)

        Returns:
            {"is_confounder": bool, "unconditional_rho": float,
             "conditional_rho": float, "confounder_score": float}
        """
        self._ensure_matrix()
        mat = self._tfidf_matrix

        x = mat[mem_a_idx]
        y = mat[mem_b_idx]
        z = mat[candidate_z_idx]

        # 无条件相关系数
        r_xy, p_xy = pearsonr(x, y)

        # 条件相关系数
        r_xy_z, p_xy_z = self.partial_correlation(x, y, z)

        # 混淆因子得分: 无条件下相关 → 条件下不相关
        # confounder_score ∈ [0, 1], 越高越说明 Z 是混淆因子
        if abs(r_xy) > 0.1 and abs(r_xy_z) < 0.1:
            conf_score = 1.0
        elif abs(r_xy) > 0.1:
            reduction = max(0, abs(r_xy) - abs(r_xy_z)) / abs(r_xy)
            conf_score = round(reduction, 4)
        else:
            conf_score = 0.0

        return {
            "is_confounder": conf_score > 0.5,
            "unconditional_rho": round(r_xy, 4),
            "unconditional_p": round(p_xy, 4),
            "conditional_rho": round(r_xy_z, 4),
            "conditional_p": round(p_xy_z, 4),
            "confounder_score": conf_score,
        }

    # -----------------------------------------------------------------
    # 统计摘要
    # -----------------------------------------------------------------

    def get_statistics(self) -> dict:
        """获取当前因果发现引擎的统计摘要。"""
        return {
            "n_memories": len(self.memories),
            "vocab_size": len(self._vocab),
            "tfidf_built": self._tfidf_matrix is not None,
            "energy_bus_available": self._energy_bus is not None,
            "n_effective": self._n_effective,
        }

    # -----------------------------------------------------------------
    # FourierCausal 联动 (v3.4.0-p1)
    # -----------------------------------------------------------------

    _fourier: FourierCausal | None = None

    def with_fourier_filter(self, fourier: FourierCausal):
        """注入 FourierCausal 实例以启用频域预过滤。"""
        self._fourier = fourier

    def discover_hidden_edges_spectral(
        self,
        min_correlation: float = 0.3,
        p_threshold: float = 0.05,
        max_pairs: int = 200,
        max_scan: int = 50,
    ) -> list[dict]:
        """带频域过滤的隐藏因果发现 (discover_hidden_edges + Fourier 滤波)。"""
        edges = self.discover_hidden_edges(
            min_correlation=min_correlation,
            p_threshold=p_threshold,
            max_pairs=max_pairs,
            max_scan=max_scan,
        )
        if self._fourier is not None:
            edges = self._fourier.filter_periodic_edges(edges, self)
        return edges

    # -----------------------------------------------------------------
    # BayesianCausal 联动 (v3.4.0-p2)
    # -----------------------------------------------------------------

    _bayesian: BayesianCausal | None = None

    def with_bayesian_quantification(self, bayesian: BayesianCausal):
        """注入 BayesianCausal 以启用后验量化输出。"""
        self._bayesian = bayesian


# =============================================================================
# M2: FourierCausal — 频域因果分析 + 周期混淆过滤
# =============================================================================


class FourierCausal:
    """
    Energy Bus 信号历史的频域因果分析。

    三个注入点 (零侵入 EnergyBus):
    ① 能量传播谱分析: 对 _signal_history 做 FFT
    ② 能量平衡频谱诊断: 五维强度联合频谱
    ③ 周期混淆因子过滤: 滤除共享周期后重算偏相关
    """

    FIVE_ELEMENTS = ["wood", "fire", "earth", "metal", "water"]

    # 内置Energy Types周期先验 (归一化频率)
    ELEMENT_PERIOD_HINTS = {
        "wood": 0.25,     # T=4 → f=0.25 (春生)
        "fire": 0.25,     # T=4 → f=0.25 (夏长)
        "earth": 0.1667,  # T=6 → f≈0.1667 (长夏/过渡)
        "metal": 0.25,    # T=4 → f=0.25 (秋收)
        "water": 0.25,    # T=4 → f=0.25 (冬藏)
    }

    def __init__(self, energy_bus=None):
        """
        Args:
            energy_bus: EnergyBus 实例 (可选，用于 record_snapshot 自动采集)
        """
        self._bus = energy_bus
        self._intensity_history: dict[str, list[float]] = defaultdict(list)
        self._snapshot_count: int = 0

    # -----------------------------------------------------------------
    # 信号采集
    # -----------------------------------------------------------------

    def record_snapshot(self, intensities: dict[str, float] | None = None) -> int:
        """
        从 EnergyBus 采集当前五元素强度快照，或手动传入。

        Args:
            intensities: 手动传入的元素强度字典 {etype: value}，
                         若为 None 则从 _bus.energy_core 读取

        Returns:
            当前快照总数
        """
        self._snapshot_count += 1

        if intensities is not None:
            for etype, val in intensities.items():
                self._intensity_history[etype].append(float(val))
        elif self._bus is not None:
            try:
                core = self._bus._energy_core
                for etype in self.FIVE_ELEMENTS:
                    try:
                        node = getattr(core, f"_node_{etype}", None)
                        if node is not None:
                            val = getattr(node, "intensity", 0.5)
                            self._intensity_history[etype].append(float(val))
                    except Exception:
                        self._intensity_history[etype].append(0.5)
            except Exception:
                # 无法读取 EnergyBus → 填充默认值
                for etype in self.FIVE_ELEMENTS:
                    self._intensity_history[etype].append(0.5)

        return self._snapshot_count

    def get_series(self, etype: str) -> np.ndarray:
        """获取某元素的强度时间序列。"""
        return np.array(self._intensity_history.get(etype, []), dtype=np.float64)

    # -----------------------------------------------------------------
    # FFT 频谱分解
    # -----------------------------------------------------------------

    def fft_decompose(self, etype: str) -> dict:
        """
        FFT 分解 → 频谱成分分析 + 异常检测。

        将信号分解为:
          - DC (直流分量)
          - 基频 (主要周期)
          - 二次谐波
          - 高频残差

        Returns:
            {
                "n_samples": int,
                "dc_ratio": float,           # 直流能量占比
                "fundamental_ratio": float,  # 基频能量占比
                "harmonic2_ratio": float,    # 二次谐波占比
                "high_freq_ratio": float,    # 高频残差占比
                "anomaly_score": float,      # [0, 1] 越高越异常
                "dominant_freq": float,      # 主导频率
                "freqs": list[float],        # 频率轴 (前 10)
                "magnitudes": list[float],   # 幅值 (前 10)
            }
        """
        series = self.get_series(etype)
        n = len(series)

        if n < 5:
            return {
                "n_samples": n,
                "error": "insufficient_samples",
                "dc_ratio": 0.0,
                "fundamental_ratio": 0.0,
                "harmonic2_ratio": 0.0,
                "high_freq_ratio": 0.0,
                "anomaly_score": 0.0,
                "dominant_freq": 0.0,
                "freqs": [],
                "magnitudes": [],
            }

        # 零均值化
        centered = series - np.mean(series)

        # FFT
        fft_result = np.fft.rfft(centered)
        magnitudes = np.abs(fft_result)
        power = magnitudes ** 2
        total_power = np.sum(power) + 1e-10

        # 频率轴
        freqs = np.fft.rfftfreq(n)

        # ── 能量分区 ──
        # DC (f=0)
        dc_power = power[0] if len(power) > 0 else 0

        # 基频区 (0 < f <= 0.25)
        fund_mask = (freqs > 0) & (freqs <= 0.25)
        fund_power = np.sum(power[fund_mask]) if np.any(fund_mask) else 0

        # 二次谐波 (0.25 < f <= 0.4)
        harm2_mask = (freqs > 0.25) & (freqs <= 0.4)
        harm2_power = np.sum(power[harm2_mask]) if np.any(harm2_mask) else 0

        # 高频残差 (f > 0.4, 包含 Nyquist)
        high_mask = freqs > 0.4
        high_power = np.sum(power[high_mask]) if np.any(high_mask) else 0

        # ── 异常得分 ──
        # 双维度: 频域能量扩散度 × 时域幅值因子
        #
        # 设计原理:
        #   - 纯频域指标会把 tiny noise 误判为异常 (例如平稳信号 ±0.01 的
        #     微小波动在 FFT 中表现为宽带能量, spectral_spread 反高于尖峰信号)
        #   - 引入 CV (变异系数) 作为幅值因子: CV < 0.3 → 波动可忽略
        #   - 最终: anomaly = spectral_spread × amplitude_factor × 3
        #
        # ① 频域能量扩散度: 能量在前 2 主频外的比例
        #    0 = 全集中在 2 个频率 (周期规律) → 正常
        #    1 = 能量完全分散 (宽带噪声) → 异常
        sorted_power = np.sort(power[1:])[::-1] if len(power) > 1 else np.array([0])
        top2_power = np.sum(sorted_power[:2]) if len(sorted_power) >= 2 else (sorted_power[0] if len(sorted_power) > 0 else 0)
        spectral_spread = max(0.0, 1.0 - top2_power / (total_power - dc_power + 1e-10))

        # ② 幅值因子: 只有有意义的波动才值得关注
        #    CV = σ/μ → CV > 0.3 时 amp_factor = 1.0 (全权重)
        orig_std = float(np.std(series))
        orig_mean = float(np.mean(series))
        cv = orig_std / (orig_mean + 1e-10)
        amplitude_factor = float(np.clip(cv / 0.3, 0.0, 1.0))

        # 综合: 频域扩散 × 幅值因子 × 放大系数
        anomaly_score = float(np.clip(spectral_spread * amplitude_factor * 3.0, 0.0, 1.0))

        # ── 主导频率 ──
        if len(magnitudes) > 1:
            dominant_idx = int(np.argmax(magnitudes[1:])) + 1  # 跳过 DC
            dominant_freq = float(freqs[dominant_idx])
        else:
            dominant_freq = 0.0

        # 前 10 频率/幅值 (按幅值降序, 跳过 DC)
        sorted_indices = np.argsort(magnitudes)[::-1]

        return {
            "n_samples": n,
            "dc_ratio": round(dc_power / total_power, 4),
            "fundamental_ratio": round(fund_power / total_power, 4),
            "harmonic2_ratio": round(harm2_power / total_power, 4),
            "high_freq_ratio": round(high_power / total_power, 4),
            "anomaly_score": round(anomaly_score, 4),
            "dominant_freq": round(dominant_freq, 4),
            "freqs": [round(float(freqs[i]), 4) for i in sorted_indices],
            "magnitudes": [round(float(magnitudes[i]), 4) for i in sorted_indices],
        }

    # -----------------------------------------------------------------
    # 因果异常事件检测
    # -----------------------------------------------------------------

    def detect_causal_events(self, threshold: float = 0.3) -> list[dict]:
        """
        检测高频异常 → 解释为因果干预事件。

        当某元素的频谱异常得分超过阈值时，该采样时刻可能是
        外部因果事件导致能量场扰动。

        Args:
            threshold: anomaly_score 阈值

        Returns:
            [{"element": str, "anomaly_score": float,
              "dominant_freq": float, "n_samples": int}, ...]
            按 anomaly_score 降序
        """
        events = []
        for etype in self.FIVE_ELEMENTS:
            decomp = self.fft_decompose(etype)
            if decomp.get("error"):
                continue
            if decomp["anomaly_score"] >= threshold:
                events.append({
                    "element": etype,
                    "anomaly_score": decomp["anomaly_score"],
                    "dominant_freq": decomp["dominant_freq"],
                    "high_freq_ratio": decomp["high_freq_ratio"],
                    "n_samples": decomp["n_samples"],
                })
        events.sort(key=lambda e: e["anomaly_score"], reverse=True)
        return events

    # -----------------------------------------------------------------
    # 双元素频域相干性
    # -----------------------------------------------------------------

    def cross_spectral_coherence(
        self, etype_a: str, etype_b: str
    ) -> dict:
        """
        双元素频域相干性分析。

        计算两元素强度序列在不同频段的相干性:
          C_{AB}(f) = |P_{AB}(f)|² / (P_{AA}(f) · P_{BB}(f))
        其中 P 为互功率谱密度。

        Returns:
            {"coherence": float, "dominant_shared_freq": float,
             "is_synchronized": bool, "sync_band": str}
        """
        series_a = self.get_series(etype_a)
        series_b = self.get_series(etype_b)
        n = min(len(series_a), len(series_b))

        if n < 5:
            return {
                "error": "insufficient_samples",
                "coherence": 0.0,
                "dominant_shared_freq": 0.0,
                "is_synchronized": False,
                "sync_band": "none",
            }

        # 零均值化
        a = series_a[:n] - np.mean(series_a[:n])
        b = series_b[:n] - np.mean(series_b[:n])

        # FFT
        fa = np.fft.rfft(a)
        fb = np.fft.rfft(b)
        freqs = np.fft.rfftfreq(n)

        # 互功率谱
        p_ab = fa * np.conj(fb)
        p_aa = np.abs(fa) ** 2
        p_bb = np.abs(fb) ** 2

        # 相干性 (跳过 DC)
        coherence_vals = np.zeros_like(freqs[1:])
        for i_f in range(1, len(freqs)):
            denom = p_aa[i_f] * p_bb[i_f]
            if denom > 1e-10:
                coherence_vals[i_f - 1] = np.abs(p_ab[i_f]) ** 2 / denom
            else:
                coherence_vals[i_f - 1] = 0.0

        mean_coherence = float(np.mean(coherence_vals)) if len(coherence_vals) > 0 else 0.0

        # 主导共享频率
        if len(coherence_vals) > 0:
            max_idx = int(np.argmax(coherence_vals))
            dominant_shared_freq = float(freqs[max_idx + 1])
        else:
            dominant_shared_freq = 0.0

        # 同步频段判定
        if dominant_shared_freq <= 0.1:
            sync_band = "low"    # 低频同步 → 周期混淆风险
        elif dominant_shared_freq <= 0.3:
            sync_band = "mid"
        else:
            sync_band = "high"   # 高频同步 → 可能是因果共振

        return {
            "coherence": round(mean_coherence, 4),
            "dominant_shared_freq": round(dominant_shared_freq, 4),
            "is_synchronized": mean_coherence > 0.5,
            "sync_band": sync_band,
        }

    # -----------------------------------------------------------------
    # 周期混淆过滤
    # -----------------------------------------------------------------

    def filter_periodic_noise(
        self, corr_matrix: np.ndarray, cutoff: float = 0.8
    ) -> np.ndarray:
        """
        滤除周期混淆的相关系数矩阵。

        当两个元素低频相干性 > cutoff 但高频不同 → 周期混淆。
        对应矩阵元素置零。

        Args:
            corr_matrix: n×n 相关系数矩阵
            cutoff: 周期混淆判定阈值

        Returns:
            过滤后的相关系数矩阵
        """
        n_elem = min(len(self.FIVE_ELEMENTS), corr_matrix.shape[0])
        filtered = corr_matrix.copy()

        for i in range(n_elem):
            for j in range(i + 1, n_elem):
                coherence = self.cross_spectral_coherence(
                    self.FIVE_ELEMENTS[i], self.FIVE_ELEMENTS[j]
                )
                if coherence.get("error"):
                    continue

                # 低频同步 + 高相干性 = 周期混淆
                if (
                    coherence["sync_band"] == "low"
                    and coherence["coherence"] > cutoff
                ):
                    filtered[i, j] = 0.0
                    filtered[j, i] = 0.0

        return filtered

    def filter_periodic_edges(
        self, edges: list[dict], dag: GaussianDAG
    ) -> list[dict]:
        """
        对 GaussianDAG 发现的因果边进行频域过滤。

        逻辑:
        - 计算每对记忆关联元素的频域相干性
        - 如果低频相干性高 (周期混淆)，则降低该边置信度
        - 如果高频相干性高 (因果共振)，则提升该边置信度

        Args:
            edges: discover_hidden_edges 的输出
            dag: GaussianDAG 实例 (用于推断元素类型)

        Returns:
            调整后的 edges
        """
        for edge in edges:
            i, j = edge["cause_idx"], edge["effect_idx"]
            if i >= len(dag.memories) or j >= len(dag.memories):
                continue

            etype_a = dag._infer_energy_type(dag.memories[i])
            etype_b = dag._infer_energy_type(dag.memories[j])

            if etype_a is None or etype_b is None:
                continue

            coherence = self.cross_spectral_coherence(etype_a, etype_b)
            if coherence.get("error"):
                continue

            # 低频同步 → 疑似周期混淆 → 降权
            if coherence["sync_band"] == "low" and coherence["coherence"] > 0.7:
                edge["confidence"] = round(edge["confidence"] * 0.7, 4)
                edge["verdict"] = "suppressed"
                edge["fourier_filtered"] = True
            # 高频同步 → 可能因果共振 → 升权
            elif coherence["sync_band"] == "high" and coherence["coherence"] > 0.5:
                edge["confidence"] = round(min(edge["confidence"] * 1.15, 0.98), 4)
                edge["fourier_filtered"] = True
            else:
                edge["fourier_filtered"] = False

            edge["spectral_coherence"] = coherence["coherence"]
            edge["sync_band"] = coherence["sync_band"]

        # 重排序
        edges.sort(key=lambda e: e["confidence"], reverse=True)
        return edges

    # -----------------------------------------------------------------
    # 五元素频谱平衡报告
    # -----------------------------------------------------------------

    def spectral_balance_report(self) -> dict:
        """
        五元素频谱平衡诊断报告。

        对每个元素做 FFT 分解，汇总成整体频谱健康度评估。

        Returns:
            {
                "per_element": {etype: decomp_dict, ...},
                "global_anomaly_score": float,
                "health_status": str,   # "healthy"|"warning"|"critical"
                "most_anomalous": str,  # 最异常的元素
                "recommendations": [str, ...],
            }
        """
        per_element = {}
        global_anomaly = 0.0
        most_anomalous = None
        max_anomaly = 0.0

        for etype in self.FIVE_ELEMENTS:
            decomp = self.fft_decompose(etype)
            per_element[etype] = decomp
            score = decomp.get("anomaly_score", 0)
            global_anomaly += score
            if score > max_anomaly:
                max_anomaly = score
                most_anomalous = etype

        # 均分
        global_anomaly /= 5.0

        # 健康判定
        if global_anomaly < 0.2:
            health = "healthy"
            recs = ["能量频谱正常，无异常波动"]
        elif global_anomaly < 0.5:
            health = "warning"
            recs = [
                f"⚠️  {most_anomalous} 元素存在高频扰动 (score={max_anomaly:.2f})",
                "建议检查是否有外部因果事件干扰能量场",
            ]
        else:
            health = "critical"
            recs = [
                f"🚨 {most_anomalous} 元素严重异常 (score={max_anomaly:.2f})",
                "建议暂停自动决策，人工审核因果链路",
            ]

        return {
            "per_element": per_element,
            "global_anomaly_score": round(global_anomaly, 4),
            "health_status": health,
            "most_anomalous": most_anomalous,
            "recommendations": recs,
        }


# =============================================================================
# M3: GaussianDistribution — 连续因果效应的共轭先验
# =============================================================================


@dataclass
class GaussianDistribution:
    """
    高斯分布 — 连续因果效应的共轭先验。

    平行于已有 BetaDistribution:
    - BetaDistribution → 二项概率 (事件是否发生)
    - GaussianDistribution → 连续效应 (效应有多大)

    Normal-Normal 共轭:
      μ_post = (μ₀/σ₀² + n·x̄/σ²) / (1/σ₀² + n/σ²)
      σ²_post = 1 / (1/σ₀² + n/σ²)
    """

    mu: float = 0.0
    sigma: float = 1.0
    n_observations: int = 0

    @property
    def mean(self) -> float:
        """后验均值。"""
        return self.mu

    @property
    def variance(self) -> float:
        """后验方差。"""
        return self.sigma ** 2

    @property
    def precision(self) -> float:
        """后验精度 τ = 1/σ²。"""
        return 1.0 / (self.sigma ** 2 + 1e-10)

    def pdf(self, x: float) -> float:
        """概率密度函数值。"""
        return float(norm.pdf(x, loc=self.mu, scale=self.sigma))

    def cdf(self, x: float) -> float:
        """累积分布函数值。"""
        return float(norm.cdf(x, loc=self.mu, scale=self.sigma))

    def credible_interval(self, prob: float = 0.95) -> tuple[float, float]:
        """
        等尾可信区间。

        μ ± z·σ, z = Φ⁻¹(1 - (1-prob)/2)

        Args:
            prob: 可信度 (默认 0.95)

        Returns:
            (lower, upper)
        """
        alpha = (1.0 - prob) / 2.0
        z = norm.ppf(1.0 - alpha)
        lower = self.mu - z * self.sigma
        upper = self.mu + z * self.sigma
        return (float(lower), float(upper))

    def update(
        self,
        sample_mean: float,
        sample_std: float,
        n: int,
    ) -> GaussianDistribution:
        """
        Normal-Normal 共轭更新 (自共轭)。

        数学:
          先验: N(μ₀, σ₀²)
          似然: N(x̄, σ²/n)
          后验: N(μ_post, σ²_post)

          τ₀ = 1/σ₀²  (先验精度)
          τ_lik = n/σ²  (似然精度)
          τ_post = τ₀ + τ_lik
          μ_post = (μ₀·τ₀ + x̄·τ_lik) / τ_post
          σ²_post = 1 / τ_post

        Args:
            sample_mean: 样本均值 x̄
            sample_std: 样本标准差 σ (已知/估计)
            n: 样本量

        Returns:
            self (链式调用)
        """
        if n <= 0:
            return self

        prior_precision = 1.0 / (self.sigma ** 2 + 1e-10)
        likelihood_precision = n / (sample_std ** 2 + 1e-10)
        posterior_precision = prior_precision + likelihood_precision

        self.mu = (self.mu * prior_precision + sample_mean * likelihood_precision) / posterior_precision
        self.sigma = 1.0 / math.sqrt(posterior_precision)
        self.n_observations += n

        return self

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "n_observations": self.n_observations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GaussianDistribution:
        """从字典反序列化。"""
        return cls(
            mu=d.get("mu", 0.0),
            sigma=d.get("sigma", 1.0),
            n_observations=d.get("n_observations", 0),
        )


# =============================================================================
# M3: BayesianCausal — 贝叶斯因果后验量化
# =============================================================================


class BayesianCausal:
    """
    贝叶斯因果推断 — 能量先验增强版。

    H₀: 无因果效应 (ρ = 0)
    H₁: 有因果效应 (ρ ≠ 0)

    prior fused into core engine:
    - 生关系 → μ₀ = 0.3, σ₀ = 0.5 (正效应预期)
    - 克关系 → μ₀ = 0.0, σ₀ = 0.3 (保守)
    - 无关系 → μ₀ = 0.0, σ₀ = 1.0 (无信息先验)

    Bayes Factor:
      BF₁₀ = P(data | H₁) / P(data | H₀)
      使用 Savage-Dickey density ratio 近似:
      BF₁₀ ≈ prior(ρ=0) / posterior(ρ=0)
    """

    # 能量关系 → 先验参数映射
    ENERGY_PRIORS = {
        "enhance":    {"mu": 0.3, "sigma": 0.5},   # 生 → 正向预期
        "suppress":   {"mu": 0.0, "sigma": 0.3},   # 克 → 保守
        "same":       {"mu": 0.15, "sigma": 0.5},   # 同 → 弱正向
        "neutral":    {"mu": 0.0, "sigma": 1.0},    # 无 → 无信息
    }

    BF_THRESHOLDS = {
        "strong_h1": 10.0,
        "moderate_h1": 3.0,
        "weak_h1": 1.0,
    }

    def __init__(self, energy_bus=None):
        """
        Args:
            energy_bus: EnergyBus 实例 (可选, 用于能量先验推断)
        """
        self._bus = energy_bus
        self._posteriors: dict[str, GaussianDistribution] = {}
        self._test_history: list[dict] = []

    # -----------------------------------------------------------------
    # 先验选择
    # -----------------------------------------------------------------

    def _select_prior(self, energy_relation: str | None = None) -> GaussianDistribution:
        """根据能量关系选择先验分布。"""
        if energy_relation and energy_relation in self.ENERGY_PRIORS:
            p = self.ENERGY_PRIORS[energy_relation]
            return GaussianDistribution(mu=p["mu"], sigma=p["sigma"])
        # 默认: 无信息先验
        return GaussianDistribution(mu=0.0, sigma=1.0)

    # -----------------------------------------------------------------
    # 贝叶斯因果假设检验
    # -----------------------------------------------------------------

    def causal_hypothesis_test(
        self,
        edge_id: str,
        rho: float,
        n_samples: int,
        energy_relation: str | None = None,
        prior_strength: str = "informative",
    ) -> dict:
        """
        贝叶斯因果假设检验。

        H₀: ρ = 0 (无因果效应)
        H₁: ρ ≠ 0 (有因果效应)

        使用 Savage-Dickey density ratio 近似计算 BF:
          BF₁₀ ≈ prior(ρ=0) / posterior(ρ=0)

        Args:
            edge_id: 因果边 ID
            rho: 观测偏相关系数 (效应量)
            n_samples: 有效样本数
            energy_relation: 能量关系类型 (enhance/suppress/same/neutral)
            prior_strength: 先验强度 ("informative" | "weak")

        Returns:
            {
                "posterior_mean": float,
                "posterior_std": float,
                "credible_interval_95": (float, float),
                "bayes_factor": float,
                "conclusion": str,
                "energy_prior_used": str,
            }
        """
        # 选择先验
        if prior_strength == "weak":
            prior = GaussianDistribution(mu=0.0, sigma=2.0)
        else:
            prior = self._select_prior(energy_relation)

        # Fisher z-transform: ρ → z 近似正态
        # z = 0.5 * ln((1+ρ)/(1-ρ)), se(z) ≈ 1/√(n-3)
        if abs(rho) >= 1.0 - 1e-10:
            z_score = 10.0 if rho > 0 else -10.0
        else:
            z_score = 0.5 * math.log((1.0 + rho) / (1.0 - rho))
        z_se = 1.0 / math.sqrt(max(n_samples - 3, 1))

        # 后验更新 (Normal-Normal 共轭)
        posterior = GaussianDistribution(mu=prior.mu, sigma=prior.sigma)
        posterior.update(sample_mean=z_score, sample_std=z_se, n=1)

        # Savage-Dickey Bayes Factor: BF₁₀ = prior(ρ=0) / posterior(ρ=0)
        # 在 z-scale 上: prior(z=0) / posterior(z=0)
        prior_at_zero = prior.pdf(0.0)
        post_at_zero = posterior.pdf(0.0)

        if post_at_zero > 1e-15:
            bayes_factor = prior_at_zero / post_at_zero
        else:
            bayes_factor = float("inf")

        # 防止 BF 过大溢出
        if bayes_factor > 1e6:
            bayes_factor = float("inf")

        # 结论
        if bayes_factor > self.BF_THRESHOLDS["strong_h1"]:
            conclusion = "strong_evidence_for_causal"
        elif bayes_factor > self.BF_THRESHOLDS["moderate_h1"]:
            conclusion = "moderate_evidence_for_causal"
        elif bayes_factor > self.BF_THRESHOLDS["weak_h1"]:
            conclusion = "weak_evidence_for_causal"
        elif bayes_factor > 1.0 / self.BF_THRESHOLDS["weak_h1"]:
            conclusion = "inconclusive"
        else:
            conclusion = "evidence_for_no_causal"

        ci = posterior.credible_interval(0.95)

        # 存储后验
        self._posteriors[edge_id] = posterior

        result = {
            "posterior_mean": round(posterior.mu, 6),
            "posterior_std": round(posterior.sigma, 6),
            "credible_interval_95": (round(ci[0], 6), round(ci[1], 6)),
            "bayes_factor": bayes_factor if bayes_factor == float("inf") else round(bayes_factor, 4),
            "conclusion": conclusion,
            "energy_prior_used": energy_relation or "none",
        }

        self._test_history.append({"edge_id": edge_id, **result})
        return result

    # -----------------------------------------------------------------
    # 批量更新
    # -----------------------------------------------------------------

    def batch_update(self, edges: list[dict]) -> list[dict]:
        """
        批量更新多条因果边的后验。

        Args:
            edges: discover_hidden_edges 的输出列表

        Returns:
            附加了后验字段的 edges
        """
        for _i, edge in enumerate(edges):
            eid = f"{edge['cause_idx']}_{edge['effect_idx']}"
            posterior = self.causal_hypothesis_test(
                edge_id=eid,
                rho=edge["rho"],
                n_samples=edge.get("n_samples", 10),
                energy_relation=edge.get("energy_relation"),
            )
            edge["posterior_mean"] = posterior["posterior_mean"]
            edge["posterior_std"] = posterior["posterior_std"]
            edge["credible_interval_95"] = posterior["credible_interval_95"]
            edge["bayes_factor"] = posterior["bayes_factor"]
            edge["conclusion"] = posterior["conclusion"]

        # 按后验均值重排序
        edges.sort(key=lambda e: abs(e.get("posterior_mean", 0)), reverse=True)
        return edges

    # -----------------------------------------------------------------
    # 摘要
    # -----------------------------------------------------------------

    def get_summary(self) -> dict:
        """
        所有因果边的概率摘要。

        Returns:
            {
                "n_edges_tested": int,
                "n_strong_causal": int,
                "n_moderate_causal": int,
                "n_inconclusive": int,
                "max_bayes_factor": float,
                "edges": [{"id": ..., "conclusion": ..., "posterior_mean": ...}, ...],
            }
        """
        summary = {
            "n_edges_tested": len(self._test_history),
            "n_strong_causal": 0,
            "n_moderate_causal": 0,
            "n_inconclusive": 0,
            "max_bayes_factor": 0.0,
            "edges": [],
        }

        for test in self._test_history:
            summary["edges"].append({
                "id": test["edge_id"],
                "conclusion": test["conclusion"],
                "posterior_mean": test["posterior_mean"],
                "bayes_factor": (
                    test["bayes_factor"] if test["bayes_factor"] != float("inf") else "inf"
                ),
            })

            if "strong" in test["conclusion"]:
                summary["n_strong_causal"] += 1
            elif "moderate" in test["conclusion"]:
                summary["n_moderate_causal"] += 1
            elif "inconclusive" in test["conclusion"]:
                summary["n_inconclusive"] += 1

            bf = test["bayes_factor"]
            if isinstance(bf, (int, float)) and bf != float("inf"):
                summary["max_bayes_factor"] = max(summary["max_bayes_factor"], bf)

        return summary

    # -----------------------------------------------------------------
    # 假设比较
    # -----------------------------------------------------------------

    def compare_hypotheses(self, edge_id_a: str, edge_id_b: str) -> dict:
        """
        比较两条竞争因果边的后验。

        后验比值 = P(H_a | data) / P(H_b | data)
                  ≈ BF_a / BF_b (等先验概率下)

        Returns:
            {"posterior_odds": float, "favored": str, "confidence": str}
        """
        post_a = self._posteriors.get(edge_id_a)
        post_b = self._posteriors.get(edge_id_b)

        if post_a is None or post_b is None:
            return {
                "posterior_odds": 1.0,
                "favored": "unknown",
                "confidence": "insufficient_data",
            }

        # 计算后验概率比 (使用后验均值的绝对值)
        odds = abs(post_a.mu) / (abs(post_b.mu) + 1e-10)

        if odds > 3.0:
            favored = edge_id_a
            confidence = "strong"
        elif odds > 1.5:
            favored = edge_id_a
            confidence = "moderate"
        elif odds < 1.0 / 3.0:
            favored = edge_id_b
            confidence = "strong"
        elif odds < 1.0 / 1.5:
            favored = edge_id_b
            confidence = "moderate"
        else:
            favored = "neither"
            confidence = "weak"

        return {
            "posterior_odds": round(odds, 4),
            "favored": favored,
            "confidence": confidence,
        }
