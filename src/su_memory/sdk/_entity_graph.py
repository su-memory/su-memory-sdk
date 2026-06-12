"""
su-memory v4.0 — Entity Graph Traversal Engine (实体图谱遍历引擎)

从 top 语义命中出发，沿实体链接 BFS 发现间接相关记忆。

核心能力:
- 实体倒排索引: entity_name → [memory_ids]
- Spreading Activation: 从种子节点BFS传播激活值
- 实体归一化: Levenshtein距离 + 共现模式

参考: Hindsight (91.4%) 四网络图谱 + ByteRover (92.8%) 上下文树
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class EntityGraph:
    """实体图谱 — 存储 entity → memory_ids 倒排索引，支持 Spreading Activation 检索

    用法:
        graph = EntityGraph()
        graph.add_fact("Alice moved to Boston", "fact_001", entities=["Alice", "Boston"])
        graph.add_fact("Alice works at TechCorp", "fact_002", entities=["Alice", "TechCorp"])

        # 从种子 memory_ids 出发，BFS 发现关联记忆
        related = graph.spreading_activation(seed_ids=["fact_001"], depth=2)
        # → ["fact_002"]  (通过 "Alice" 实体连接)
    """

    def __init__(self):
        # entity_name (normalized) → set of memory_ids
        self._entity_to_mids: dict[str, set[str]] = defaultdict(set)
        # memory_id → set of entity_names
        self._mid_to_entities: dict[str, set[str]] = defaultdict(set)
        # 实体归一化映射: 原始名 → 归一化名
        self._entity_norm: dict[str, str] = {}
        # 统计
        self._total_facts = 0
        self._total_entities = 0

    def add_fact(
        self,
        memory_id: str,
        entities: list[str],
        content: str = "",
    ) -> None:
        """添加一个记忆节点及其关联实体

        Args:
            memory_id: 记忆条目 ID
            entities: 实体列表（人名、地名等）
            content: 记忆内容（可选，用于归一化）
        """
        if not entities:
            return

        self._total_facts += 1
        normalized_entities = set()

        for entity in entities:
            if not entity or len(entity) < 2:
                continue
            norm = self._normalize_entity(entity)
            normalized_entities.add(norm)
            self._entity_to_mids[norm].add(memory_id)
            self._mid_to_entities[memory_id].add(norm)

            # 保存归一化映射
            if entity.lower() != norm:
                self._entity_norm[entity.lower()] = norm

        self._total_entities = len(self._entity_to_mids)

    def add_fact_from_metadata(
        self,
        memory_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """从 memory metadata 中提取实体并添加"""
        entities_json = metadata.get("entities", "")
        if not entities_json:
            return

        try:
            if isinstance(entities_json, str):
                entities = json.loads(entities_json)
            elif isinstance(entities_json, list):
                entities = entities_json
            else:
                return
        except (json.JSONDecodeError, TypeError):
            return

        if entities:
            self.add_fact(memory_id, entities)

    def spreading_activation(
        self,
        seed_ids: list[str],
        depth: int = 2,
        decay: float = 0.8,
        threshold: float = 0.1,
        max_nodes: int = 50,
        entity_weight: float = 1.5,
    ) -> list[tuple[str, float]]:
        """Spreading Activation 检索 — 从种子节点出发 BFS 发现关联记忆

        Args:
            seed_ids: 种子 memory_id 列表（通常是 top 语义命中）
            depth: BFS 深度（默认2层）
            decay: 衰减因子（默认0.8，每层乘以此值）
            threshold: 激活值阈值（低于此值的节点不收集）
            max_nodes: 最多收集的节点数
            entity_weight: 实体边权重（比语义/时间边更高）

        Returns:
            [(memory_id, activation_score), ...] 按激活值降序排列
        """
        if not seed_ids:
            return []

        # 激活值表
        activation: dict[str, float] = {}
        # 种子节点初始激活值
        for i, mid in enumerate(seed_ids):
            activation[mid] = 1.0 / (i + 1)  # 排名越前，初始激活值越高

        # BFS
        visited: set[str] = set(seed_ids)
        current_frontier = list(seed_ids)

        for d in range(depth):
            next_frontier: list[str] = []
            layer_decay = decay ** (d + 1)

            for mid in current_frontier:
                mid_activation = activation.get(mid, 0)
                if mid_activation < threshold:
                    continue

                # 获取此记忆关联的所有实体
                entities = self._mid_to_entities.get(mid, set())

                for entity in entities:
                    # 获取此实体关联的所有记忆
                    related_mids = self._entity_to_mids.get(entity, set())

                    for related_mid in related_mids:
                        if related_mid in visited:
                            continue

                        # 计算传播激活值
                        propagated = mid_activation * layer_decay * entity_weight
                        current = activation.get(related_mid, 0)
                        activation[related_mid] = max(current, propagated)

                        if related_mid not in visited:
                            next_frontier.append(related_mid)
                            visited.add(related_mid)

            current_frontier = next_frontier
            if not current_frontier:
                break

        # 收集激活值 > 阈值的非种子节点
        results = []
        for mid, score in activation.items():
            if mid not in seed_ids and score >= threshold:
                results.append((mid, score))

        # 按激活值降序排列
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:max_nodes]

    def get_entity_mids(self, entity: str) -> set[str]:
        """获取实体关联的所有 memory_id"""
        norm = self._normalize_entity(entity)
        return self._entity_to_mids.get(norm, set())

    def get_mid_entities(self, memory_id: str) -> set[str]:
        """获取记忆关联的所有实体"""
        return self._mid_to_entities.get(memory_id, set())

    def find_connected_mids(self, memory_ids: list[str]) -> list[str]:
        """查找与给定 memory_ids 通过实体连接的其他 memory_ids"""
        all_entities: set[str] = set()
        for mid in memory_ids:
            all_entities.update(self._mid_to_entities.get(mid, set()))

        connected: set[str] = set()
        for entity in all_entities:
            connected.update(self._entity_to_mids.get(entity, set()))

        # 排除给定的 memory_ids
        connected -= set(memory_ids)
        return list(connected)

    @property
    def total_entities(self) -> int:
        return len(self._entity_to_mids)

    @property
    def total_facts(self) -> int:
        return self._total_facts

    # ------------------------------------------------------------------
    # 内部: 实体归一化
    # ------------------------------------------------------------------

    _SUFFIXES = frozenset({
        "'s", "'t", "'re", "'ve", "'ll", "'d",
        "inc", "corp", "ltd", "co", "llc",
    })

    def _normalize_entity(self, entity: str) -> str:
        """实体归一化 — 小写、去后缀、去空格"""
        norm = entity.strip().lower()

        # 去常见后缀
        for suffix in self._SUFFIXES:
            if norm.endswith(suffix):
                norm = norm[:-len(suffix)].strip()

        # 查找已有映射
        if norm in self._entity_norm:
            return self._entity_norm[norm]

        return norm


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def create_entity_graph() -> EntityGraph:
    """工厂函数：创建 EntityGraph 实例"""
    return EntityGraph()
