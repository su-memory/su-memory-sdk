"""
versioning — 临床事实版本链（C6: 诊疗变更可回溯）

同一患者同一事实（如营养方案/诊断/用药）多次更新时，建立版本链：
  - 新版本 prev_version_id 指向旧版本
  - 旧版本 superseded_by 指向新版本
  - get_active() 返回 superseded_by="" 的最新生效版本
  - get_history() 回溯完整版本链

⚠️ 项目区隔：版本链只做「事实变更回溯」，不做「诊断推理」。
   版本判定基于 fact_key（业务侧定义的事实键），不涉及因果。

Example:
  >>> from su_memory.clinical.versioning import ClinicalVersionChain
  >>> chain = ClinicalVersionChain(client._engine)
  >>> chain.update_fact("P001", "nutrition_plan", "方案v2")
  >>> history = chain.get_history("P001", "nutrition_plan")
  >>> active = chain.get_active("P001", "nutrition_plan")
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ClinicalVersionChain:
    """同一患者同一事实的版本链管理。

    Args:
        engine: SuMemoryLitePro 实例（访问 _memories 和 _memory_map）
    """

    def __init__(self, engine: Any):
        self._engine = engine

    def update_fact(
        self,
        patient_id: str,
        fact_key: str,
        new_content: str,
        metadata: dict | None = None,
        source_type: str = "order",
        source_id: str = "",
    ) -> str:
        """更新一个临床事实——创建新版本，链入旧版本。

        Args:
            patient_id: 患者 ID
            fact_key: 事实键（如 "nutrition_plan" / "diagnosis_primary"）
            new_content: 新版本内容
            metadata: 额外元数据
            source_type: 来源类型
            source_id: 来源 ID

        Returns:
            新版本的 memory_id
        """
        # 1. 查找当前活跃版本（superseded_by="" 且 fact_key 匹配）
        active = self._find_active(patient_id, fact_key)
        new_version = (active["version"] + 1) if active else 1  # C6: 基于活跃版本号递增

        full_meta: dict[str, Any] = {
            "patient_id": patient_id,
            "event_type": "fact_update",
            "fact_key": fact_key,
            "fact_version": new_version,
        }
        if metadata:
            full_meta.update(metadata)

        # 2. 创建新版本
        prev_id = active["memory_id"] if active else ""
        new_mid = self._engine.add(
            new_content,
            metadata=full_meta,
            source_type=source_type,
            source_id=source_id,
        )

        # 3. 链入版本链：更新新节点的 prev_version_id + 旧节点 superseded_by
        try:
            new_node = self._get_node(new_mid)
            if new_node is not None:
                new_node.prev_version_id = prev_id
                new_node.version = new_version
            if active:
                old_node = self._get_node(active["memory_id"])
                if old_node is not None:
                    old_node.superseded_by = new_mid
        except Exception as e:
            logger.debug("[VersionChain] 版本链更新降级: %s", e)

        logger.info(
            "[VersionChain] %s/%s 更新到 v%d (%s)",
            patient_id, fact_key, new_version, new_mid[:12],
        )
        return str(new_mid)

    def get_history(
        self, patient_id: str, fact_key: str
    ) -> list[dict[str, Any]]:
        """回溯某事实的完整版本链（从最早到最新）。

        Returns:
            [{"memory_id", "content", "version", "timestamp", "active"}, ...]
        """
        # 找到最新版本
        active = self._find_active(patient_id, fact_key)
        if not active:
            return []

        # 从最新往前回溯
        chain: list[dict[str, Any]] = []
        current: dict[str, Any] | None = active
        seen: set[str] = set()
        while current and current["memory_id"] not in seen:
            seen.add(current["memory_id"])
            chain.append(current)
            prev_id = current.get("prev_version_id", "")
            if not prev_id:
                break
            current = self._node_to_dict(prev_id)

        chain.reverse()  # 从最早到最新
        return chain

    def get_active(
        self, patient_id: str, fact_key: str
    ) -> dict[str, Any] | None:
        """获取某事实的当前生效版本。"""
        return self._find_active(patient_id, fact_key)

    def list_fact_keys(self, patient_id: str) -> list[str]:
        """列出某患者所有有版本记录的事实键。"""
        keys: set[str] = set()
        graph = getattr(self._engine, "_graph", None)
        if graph is None:
            return []
        for _mid, node in getattr(graph, "_nodes", {}).items():
            meta = node.metadata or {}
            if meta.get("patient_id") == patient_id and "fact_key" in meta:
                keys.add(meta["fact_key"])
        return sorted(keys)

    # ── 内部辅助 ──────────────────────────────────────────

    def _find_active(
        self, patient_id: str, fact_key: str
    ) -> dict[str, Any] | None:
        """查找 superseded_by="" 且 fact_key 匹配的最新版本。"""
        graph = getattr(self._engine, "_graph", None)
        if graph is None:
            return None
        candidates: list[tuple[int, Any]] = []
        for _mid, node in getattr(graph, "_nodes", {}).items():
            meta = node.metadata or {}
            if (meta.get("patient_id") == patient_id
                    and meta.get("fact_key") == fact_key
                    and not node.superseded_by):
                candidates.append((node.version, node))
        if not candidates:
            return None
        # 取版本号最大的
        candidates.sort(key=lambda x: x[0], reverse=True)
        node = candidates[0][1]
        return self._node_to_dict_obj(node)

    def _get_node(self, memory_id: str):
        """通过 memory_map 获取 node。"""
        idx_map = getattr(self._engine, "_memory_map", {})
        idx = idx_map.get(memory_id)
        if idx is None:
            return None
        memories = getattr(self._engine, "_memories", [])
        if 0 <= idx < len(memories):
            return memories[idx]
        return None

    def _node_to_dict(self, memory_id: str) -> dict[str, Any] | None:
        node = self._get_node(memory_id)
        if node is None:
            return None
        return self._node_to_dict_obj(node)

    def _node_to_dict_obj(self, node) -> dict[str, Any]:
        return {
            "memory_id": node.id,
            "content": node.content,
            "version": node.version,
            "timestamp": node.timestamp,
            "event_time": node.effective_time,
            "prev_version_id": node.prev_version_id,
            "superseded_by": node.superseded_by,
            "active": not node.superseded_by,
            "metadata": node.metadata or {},
        }
