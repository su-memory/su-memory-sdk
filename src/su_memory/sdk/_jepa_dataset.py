"""
su-memory v4.0.0 — JEPA Dataset
================================

从记忆时间线构造 JEPA 时序训练数据对 (s_t, s_{t+1})。

核心组件:
- JEPADataset: 将 CausalWorldModelState 序列转换为训练对
- from_memories(): 从原始记忆列表 → 因果发现 → 状态序列 → 训练对

用法:
    from su_memory.sdk._jepa_dataset import JEPADataset

    dataset = JEPADataset.from_memories(world_model, memory_timeline)
    for s_t, s_t1 in dataset.pairs:
        distance = s_t.state_distance(s_t1)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class JEPADataset:
    """
    JEPA 时序训练数据集。

    将按时间排序的因果世界状态序列转换为 (s_t, s_{t+1}) 训练对，
    用于 JEPA 编码器-预测器的端到端训练。

    时间窗口自适应:
    - 最少 3 条记忆才能做因果发现（`discover()` 要求 ≥ 3）
    - 最多 20 条记忆避免因果图过大、发现噪声过高
    - 窗口内记忆按 timestamp 分组
    """

    pairs: list[tuple] = field(default_factory=list)
    # [(s_t, s_{t+1}), ...]

    # ── M3: 端到端可微训练 ──
    memory_pairs: list[tuple] = field(default_factory=list)
    # [(memories_t, memories_{t+1}), ...] — 原始记忆窗口对
    # 用于 GAT 编码器的端到端训练

    state_pairs: list[tuple] = field(default_factory=list)
    # [(state_t, state_{t+1}), ...] — 预计算的 CausalWorldModelState 对
    # 用于避免训练时重复调用 discover()

    n_states: int = 0
    n_pairs: int = 0
    n_memory_pairs: int = 0
    window_size: int = 10

    # ── 统计 ──
    avg_distance: float = 0.0
    min_distance: float = 0.0
    max_distance: float = 0.0

    @classmethod
    def from_states(
        cls,
        states: list,
        window_size: int = 10,
        min_memories_per_window: int = 3,
        max_memories_per_window: int = 20,
    ) -> JEPADataset:
        """
        从已计算好的 CausalWorldModelState 序列构造训练对。

        Args:
            states: CausalWorldModelState 列表（已按 timestamp 排序）
            window_size: 时间窗口大小（多少个状态组成一个滑动窗）
            min_memories_per_window: 每窗口最少记忆数
            max_memories_per_window: 每窗口最多记忆数

        Returns:
            JEPADataset 实例
        """
        from su_memory.sdk._world_model import CausalWorldModelState

        # 过滤无效状态
        valid_states: list[CausalWorldModelState] = [
            s for s in states
            if isinstance(s, CausalWorldModelState) and s.causal_edges
        ]

        if len(valid_states) < 2:
            logger.warning(
                "有效状态不足（%d < 2），无法构造 JEPA 训练对", len(valid_states)
            )
            return cls(pairs=[], n_states=len(valid_states), n_pairs=0)

        pairs: list[tuple] = []
        for i in range(len(valid_states) - 1):
            s_t = valid_states[i]
            s_t1 = valid_states[i + 1]
            # 确保每个状态至少有最小记忆数
            if s_t.n_memories >= min_memories_per_window and s_t1.n_memories >= min_memories_per_window:
                pairs.append((s_t, s_t1))

        distances: list[float] = []
        for s_t, s_t1 in pairs:
            d = s_t.state_distance(s_t1)
            distances.append(d)

        avg_d = float(np.mean(distances)) if distances else 0.0
        min_d = float(np.min(distances)) if distances else 0.0
        max_d = float(np.max(distances)) if distances else 0.0

        logger.info(
            "JEPADataset: %d 状态 → %d 训练对, "
            "avg_distance=%.4f [%.4f, %.4f]",
            len(valid_states), len(pairs), avg_d, min_d, max_d,
        )

        return cls(
            pairs=pairs,
            n_states=len(valid_states),
            n_pairs=len(pairs),
            window_size=window_size,
            avg_distance=avg_d,
            min_distance=min_d,
            max_distance=max_d,
        )

    @classmethod
    def from_memories(
        cls,
        world_model,
        memories: list[dict],
        window_size: int = 10,
        min_memories_per_window: int = 3,
        max_memories_per_window: int = 20,
    ) -> JEPADataset:
        """
        从原始记忆列表构造训练对。

        流程:
        1. 按时间分组记忆
        2. 每组用 world_model.discover() 做因果发现
        3. 产生的 CausalWorldModelState 序列 → from_states()

        并发安全 (P2 加固):
        - 通过 ``copy.deepcopy()`` 拷贝 discover() 返回的状态，
          避免与后续同 world_model 的 discover() 调用产生交叉污染
        - 本身与 ``world_model.discover()`` 的 ``_discover_lock`` 协同保证状态一致性

        Args:
            world_model: MCIWorldModel 实例（已初始化）
            memories: 记忆列表 [{"content": ..., "timestamp": ...}, ...]
            window_size: 时间窗口大小
            min_memories_per_window: 每窗口最少记忆数
            max_memories_per_window: 每窗口最多记忆数

        Returns:
            JEPADataset 实例
        """
        if not memories or len(memories) < 3:
            logger.warning("记忆不足（%d < 3），无法做因果发现", len(memories))
            return cls(pairs=[], n_states=0, n_pairs=0)

        # 按时间排序
        sorted_memories = sorted(
            memories,
            key=lambda m: m.get("timestamp", ""),
        )

        # 滑动窗口：每 window_size 条记忆做一次因果发现
        states = []
        memory_batches = []  # M3: 原始记忆窗口
        step = max(window_size // 2, 1)  # 50% 重叠
        for start in range(0, len(sorted_memories) - min_memories_per_window + 1, step):
            batch = sorted_memories[start : start + window_size]
            batch = batch[:max_memories_per_window]
            if len(batch) < min_memories_per_window:
                continue
            try:
                state = world_model.discover(batch, use_parametric=False, verbose=False)
                # P2 并发加固: 拷贝状态以避免后续 world_model.discover()
                # 调用篡改本线程已存储的 state
                # (discover() 返回的是 world_model._state 引用)
                state = copy.deepcopy(state)
                if state.causal_edges:
                    state.n_memories = len(batch)
                    states.append(state)
                    memory_batches.append(batch)  # M3: 保存原始记忆
            except Exception as e:
                logger.warning("因果发现失败 (start=%d): %s", start, e)

        if len(states) < 2:
            logger.warning("产生的状态不足（%d < 2）", len(states))
            return cls(pairs=[], n_states=len(states), n_pairs=0)

        # ── 构造状态对 + 记忆对 (M3) ──
        result = cls.from_states(
            states,
            window_size=window_size,
            min_memories_per_window=min_memories_per_window,
            max_memories_per_window=max_memories_per_window,
        )

        # M3: 构造记忆对 + 状态对 (与状态对对齐)
        memory_pairs: list[tuple] = []
        state_pairs: list[tuple] = []
        for i in range(len(states) - 1):
            if states[i].n_memories >= min_memories_per_window and \
               states[i + 1].n_memories >= min_memories_per_window:
                memory_pairs.append((memory_batches[i], memory_batches[i + 1]))
                state_pairs.append((states[i], states[i + 1]))
        result.memory_pairs = memory_pairs
        result.state_pairs = state_pairs
        result.n_memory_pairs = len(memory_pairs)

        return result

    def to_dict(self) -> dict:
        """返回数据集统计摘要。"""
        return {
            "n_states": self.n_states,
            "n_pairs": self.n_pairs,
            "n_memory_pairs": self.n_memory_pairs,
            "window_size": self.window_size,
            "avg_distance": round(self.avg_distance, 6),
            "min_distance": round(self.min_distance, 6),
            "max_distance": round(self.max_distance, 6),
        }

    def __len__(self) -> int:
        return self.n_pairs

    def __iter__(self):
        return iter(self.pairs)

    def __repr__(self) -> str:
        return (
            f"JEPADataset(states={self.n_states}, pairs={self.n_pairs}, "
            f"avg_dist={self.avg_distance:.4f})"
        )
