"""
su-memory SDK — Memory Lifecycle Manager (v3.5.5 P1-3)

统一记忆生命周期管理：自动过期、语义去重、归档策略、健康报告。

核心类：
- MemoryLifecycleManager: 生命周期管理器

集成现有模块：
- client.decay() — 时间衰减
- ForgettingPolicy — 遗忘策略枚举
- SuCompressor — 记忆压缩

使用示例:
    >>> mgr = MemoryLifecycleManager(client)
    >>> mgr.auto_expire(days=90)
    >>> report = mgr.get_report()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ── 可选依赖 ──────────────────────────────────────────────────────────────────

_FORGETTING_POLICY_AVAILABLE = False
try:
    from su_memory._sys._incremental_learning import ForgettingPolicy  # noqa: F401
    _FORGETTING_POLICY_AVAILABLE = True
except ImportError:
    logger.debug("ForgettingPolicy 不可用，使用默认策略")

_SUCOMPRESSOR_AVAILABLE = False
try:
    from su_memory._sys.codec import SuCompressor  # noqa: F401
    _SUCOMPRESSOR_AVAILABLE = True
except ImportError:
    logger.debug("SuCompressor 不可用，压缩功能跳过")


# ── 数据模型 ──────────────────────────────────────────────────────────────────


@dataclass
class LifecycleAction:
    """单次生命周期操作记录"""
    action: str                         # "expire" | "dedup" | "archive" | "compress" | "decay"
    timestamp: str                      # ISO 时间
    affected_count: int                 # 影响的记忆数
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class LifecycleReport:
    """生命周期健康报告"""
    generated_at: str = ""
    total_memories: int = 0
    # 年龄分布
    age_distribution: dict[str, int] = field(default_factory=dict)
    # 重复检测
    duplicate_clusters: int = 0
    duplicate_memories: int = 0
    # 归档状态
    archived_count: int = 0
    active_count: int = 0
    # 衰减状态
    decayed_count: int = 0
    expired_count: int = 0
    # 健康评分 (0-100)
    health_score: float = 0.0
    health_status: str = "unknown"
    # 建议
    recommendations: list[str] = field(default_factory=list)
    # 历史操作
    recent_actions: list[LifecycleAction] = field(default_factory=list)
    # 统计
    category_distribution: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"记忆生命周期报告 [{self.health_status}] "
            f"总量={self.total_memories} "
            f"活跃={self.active_count} "
            f"归档={self.archived_count} "
            f"过期={self.expired_count} "
            f"重复={self.duplicate_memories} "
            f"评分={self.health_score:.0f}/100"
        )


# ── 生命周期管理器 ────────────────────────────────────────────────────────────


class MemoryLifecycleManager:
    """记忆生命周期管理器 (v3.5.5 P1-3)

    统一管理记忆的完整生命周期：自动过期、语义去重、归档策略、健康诊断。

    Args:
        client: SuMemory / SuMemoryLite / SuMemoryLitePro 实例
        max_action_history: 保留最近 N 条操作记录

    Example:
        >>> mgr = MemoryLifecycleManager(client)
        >>> mgr.auto_expire(days=90)  # 过期 90 天前的记忆
        >>> mgr.deduplicate(threshold=0.85)  # 去重相似度 > 85% 的记忆
        >>> print(mgr.get_report().summary())
    """

    def __init__(
        self,
        client,  # SuMemory / SuMemoryLite / SuMemoryLitePro
        max_action_history: int = 50,
    ):
        self._client = client
        self._max_action_history = max_action_history
        self._actions: list[LifecycleAction] = []

    # ── 公开 API ──────────────────────────────────────────────────────────

    def auto_expire(
        self,
        days: int = 90,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """基于时间戳自动过期旧记忆

        调用 client.decay() 进行时间衰减，然后对已衰减的记忆执行过期检查。

        Args:
            days: 超过多少天视为过期（默认 90 天）
            dry_run: 仅模拟，不实际删除

        Returns:
            {"expired": int, "decayed": int, "remaining": int}
        """
        memories = self._get_memories()
        cutoff = datetime.now() - timedelta(days=days)
        expired_ids: list[str] = []
        decayed_count = 0

        # 分析过期记忆
        for m in memories:
            ts = self._get_timestamp(m)
            if ts and ts < cutoff:
                expired_ids.append(m.get("id", ""))
            elif m.get("energy_type") == "decayed":
                decayed_count += 1

        if not dry_run and expired_ids:
            # 执行时间衰减
            try:
                if hasattr(self._client, "decay"):
                    result = self._client.decay(days)
                    decayed_count = result.get("decayed", len(expired_ids))
                else:
                    # 手动删除过期记忆
                    for mid in expired_ids:
                        if mid:
                            self._client.forget(mid)
                    decayed_count = len(expired_ids)
            except Exception as e:
                logger.error(f"自动过期失败: {e}")
                decayed_count = 0

        remaining = len(memories) - decayed_count

        action = LifecycleAction(
            action="expire" if not dry_run else "expire_dry_run",
            timestamp=datetime.now().isoformat(),
            affected_count=decayed_count,
            details={
                "days_threshold": days,
                "candidates": len(expired_ids),
                "remaining": remaining,
                "dry_run": dry_run,
            },
        )
        self._record_action(action)

        logger.info(
            f"自动过期: {decayed_count} 条 (阈值={days}天, 剩余={remaining})"
        )
        return {"expired": decayed_count, "candidates": len(expired_ids), "remaining": remaining}

    def deduplicate(
        self,
        threshold: float = 0.85,
        method: str = "content_hash",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """语义去重：检测并合并内容高度相似的记忆

        Args:
            threshold: 相似度阈值 (0-1)，默认 0.85
            method: 去重方法 "content_hash" | "semantic"
            dry_run: 仅模拟

        Returns:
            {"duplicates_found": int, "removed": int, "clusters": int}
        """
        memories = self._get_memories()
        if len(memories) < 2:
            return {"duplicates_found": 0, "removed": 0, "clusters": 0}

        if method == "semantic":
            return self._semantic_dedup(memories, threshold, dry_run)

        # content_hash: 基于内容哈希的快速去重
        seen: dict[str, list[str]] = {}
        for m in memories:
            content = m.get("content", "").strip()
            # 规范化：去空白、小写
            key = "".join(content.lower().split())
            if key not in seen:
                seen[key] = []
            seen[key].append(m.get("id", ""))

        duplicate_clusters = {k: v for k, v in seen.items() if len(v) > 1}
        total_dupes = sum(len(v) - 1 for v in duplicate_clusters.values())

        removed = 0
        if not dry_run:
            for cluster_ids in duplicate_clusters.values():
                # 保留第一条，删除其余
                for mid in cluster_ids[1:]:
                    try:
                        if mid:
                            self._client.forget(mid)
                            removed += 1
                    except Exception as e:
                        logger.warning(f"去重删除失败 {mid}: {e}")

        action = LifecycleAction(
            action="dedup" if not dry_run else "dedup_dry_run",
            timestamp=datetime.now().isoformat(),
            affected_count=removed,
            details={
                "method": method,
                "threshold": threshold,
                "clusters": len(duplicate_clusters),
                "total_duplicates": total_dupes,
                "dry_run": dry_run,
            },
        )
        self._record_action(action)

        logger.info(
            f"去重: {removed} 条 (方法={method}, 簇={len(duplicate_clusters)})"
        )
        return {
            "duplicates_found": total_dupes,
            "removed": removed,
            "clusters": len(duplicate_clusters),
        }

    def archive(
        self,
        condition: str = "old",
        threshold_days: int = 180,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """归档策略：将旧记忆标记为归档状态

        Args:
            condition: 归档条件 "old" | "low_energy" | "inactive"
            threshold_days: 归档阈值天数
            dry_run: 仅模拟

        Returns:
            {"archived": int, "candidates": int}
        """
        memories = self._get_memories()
        cutoff = datetime.now() - timedelta(days=threshold_days)
        candidates: list[str] = []

        if condition == "old":
            for m in memories:
                ts = self._get_timestamp(m)
                if ts and ts < cutoff:
                    candidates.append(m.get("id", ""))
        elif condition == "low_energy":
            for m in memories:
                energy = m.get("energy_type", "")
                if energy in ("decayed", "low"):
                    candidates.append(m.get("id", ""))
        elif condition == "inactive":
            for m in memories:
                ts = self._get_timestamp(m)
                if ts and ts < cutoff:
                    candidates.append(m.get("id", ""))

        archived = 0
        if not dry_run:
            # 通过 forget + re-add 标记为归档
            for mid in candidates:
                try:
                    memory = self._get_memory_by_id(mid)
                    if memory:
                        content = memory.get("content", "")
                        meta = memory.get("metadata") or {}
                        self._client.forget(mid)
                        self._client.add(
                            content,
                            {
                                **meta,
                                "archived": True,
                                "archived_at": datetime.now().isoformat(),
                                "archive_reason": condition,
                            },
                        )
                        archived += 1
                except Exception as e:
                    logger.warning(f"归档失败 {mid}: {e}")

        action = LifecycleAction(
            action="archive" if not dry_run else "archive_dry_run",
            timestamp=datetime.now().isoformat(),
            affected_count=archived,
            details={
                "condition": condition,
                "threshold_days": threshold_days,
                "candidates": len(candidates),
                "dry_run": dry_run,
            },
        )
        self._record_action(action)

        logger.info(f"归档: {archived} 条 (条件={condition})")
        return {"archived": archived, "candidates": len(candidates)}

    def get_report(self) -> LifecycleReport:
        """生成生命周期健康报告

        Returns:
            LifecycleReport 包含健康评分和建议
        """
        memories = self._get_memories()
        stats = self._get_stats()

        total = len(memories)

        # 年龄分布
        now = datetime.now()
        age_dist: dict[str, int] = {
            "<7天": 0, "7-30天": 0, "30-90天": 0, "90-180天": 0, ">180天": 0,
        }
        for m in memories:
            ts = self._get_timestamp(m)
            if not ts:
                continue
            age_days = (now - ts).days
            if age_days < 7:
                age_dist["<7天"] += 1
            elif age_days < 30:
                age_dist["7-30天"] += 1
            elif age_days < 90:
                age_dist["30-90天"] += 1
            elif age_days < 180:
                age_dist["90-180天"] += 1
            else:
                age_dist[">180天"] += 1

        # 去重检测
        dup_info = self.deduplicate(dry_run=True)

        # 归档/活跃计数
        archived_count = sum(
            1 for m in memories
            if (m.get("metadata") or {}).get("archived")
        )
        active_count = total - archived_count

        # 衰减/过期
        decayed_count = sum(
            1 for m in memories
            if m.get("energy_type") == "decayed"
        )
        expired_count = sum(
            1 for m in memories
            if m.get("energy_type") in ("expired", "forgotten")
        )

        # 分类分布
        category_dist: dict[str, int] = stats.get("category_distribution", {})

        # 健康评分计算
        health_score = self._calculate_health(
            total=total,
            dup_count=dup_info["duplicates_found"],
            expired_count=expired_count,
            age_dist=age_dist,
        )
        if health_score >= 80:
            health_status = "healthy"
        elif health_score >= 60:
            health_status = "fair"
        elif health_score >= 40:
            health_status = "degraded"
        else:
            health_status = "critical"

        # 建议
        recommendations = self._generate_recommendations(
            total=total,
            dup_count=dup_info["duplicates_found"],
            age_dist=age_dist,
            expired_count=expired_count,
            archived_count=archived_count,
            health_score=health_score,
        )

        report = LifecycleReport(
            generated_at=datetime.now().isoformat(),
            total_memories=total,
            age_distribution=age_dist,
            duplicate_clusters=dup_info["clusters"],
            duplicate_memories=dup_info["duplicates_found"],
            archived_count=archived_count,
            active_count=active_count,
            decayed_count=decayed_count,
            expired_count=expired_count,
            health_score=health_score,
            health_status=health_status,
            recommendations=recommendations,
            recent_actions=list(self._actions[-10:]),
            category_distribution=category_dist,
        )

        logger.info(f"生命周期报告: {report.summary()}")
        return report

    def get_action_history(self) -> list[LifecycleAction]:
        """获取操作历史"""
        return list(self._actions)

    def clear_history(self) -> None:
        """清空操作历史"""
        self._actions.clear()

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _get_memories(self) -> list[dict[str, Any]]:
        if hasattr(self._client, "get_all_memories"):
            return self._client.get_all_memories()
        stats = self._client.get_stats()
        return stats.get("recent_memories", []) or []

    def _get_stats(self) -> dict[str, Any]:
        if hasattr(self._client, "get_stats"):
            return self._client.get_stats()
        return {}

    def _get_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
        if hasattr(self._client, "get_memory"):
            return self._client.get_memory(memory_id)
        for m in self._get_memories():
            if m.get("id") == memory_id:
                return m
        return None

    @staticmethod
    def _get_timestamp(memory: dict[str, Any]) -> datetime | None:
        """从记忆中提取时间戳"""
        ts_raw = (
            memory.get("timestamp")
            or memory.get("created_at")
            or memory.get("updated_at")
            or (memory.get("metadata") or {}).get("timestamp")
            or (memory.get("metadata") or {}).get("created_at")
        )
        if not ts_raw:
            return None
        try:
            if isinstance(ts_raw, (int, float)):
                return datetime.fromtimestamp(ts_raw)
            return datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    def _record_action(self, action: LifecycleAction) -> None:
        self._actions.append(action)
        if len(self._actions) > self._max_action_history:
            self._actions = self._actions[-self._max_action_history:]

    def _semantic_dedup(
        self,
        memories: list[dict[str, Any]],
        threshold: float,
        dry_run: bool,
    ) -> dict[str, Any]:
        """基于语义相似度的去重（简化版：使用 Jaccard 相似度）"""
        # 将每条记忆转为词集
        def _word_set(text: str) -> set[str]:
            words = text.lower().split()
            return {w.strip(",.!?;:，。！？；：") for w in words if len(w) >= 2}

        clusters: list[list[str]] = []
        processed: set[int] = set()

        for i in range(len(memories)):
            if i in processed:
                continue
            wi = _word_set(memories[i].get("content", ""))
            if not wi:
                continue

            cluster = [memories[i].get("id", "")]
            for j in range(i + 1, len(memories)):
                if j in processed:
                    continue
                wj = _word_set(memories[j].get("content", ""))
                if not wj:
                    continue

                # Jaccard 相似度
                intersection = wi & wj
                union = wi | wj
                sim = len(intersection) / len(union) if union else 0

                if sim >= threshold:
                    cluster.append(memories[j].get("id", ""))
                    processed.add(j)

            if len(cluster) > 1:
                clusters.append(cluster)
                processed.add(i)

        total_dupes = sum(len(c) - 1 for c in clusters)

        removed = 0
        if not dry_run:
            for cluster_ids in clusters:
                for mid in cluster_ids[1:]:
                    try:
                        if mid:
                            self._client.forget(mid)
                            removed += 1
                    except Exception as e:
                        logger.warning(f"语义去重删除失败 {mid}: {e}")

        return {
            "duplicates_found": total_dupes,
            "removed": removed,
            "clusters": len(clusters),
        }

    def _calculate_health(
        self,
        total: int,
        dup_count: int,
        expired_count: int,
        age_dist: dict[str, int],
    ) -> float:
        """计算健康评分 (0-100)"""
        if total == 0:
            return 100.0

        score = 100.0

        # 重复扣分：每 10% 重复扣 20 分
        dup_ratio = dup_count / max(total, 1)
        score -= min(dup_ratio * 200, 30)

        # 过期扣分：每 10% 过期扣 15 分
        exp_ratio = expired_count / max(total, 1)
        score -= min(exp_ratio * 150, 25)

        # 老化扣分：超过 50% 记忆 >180 天扣 20 分
        old_ratio = age_dist.get(">180天", 0) / max(total, 1)
        score -= min(old_ratio * 40, 20)

        # 新鲜度加分：<7 天 > 30% 则 +5
        fresh_ratio = age_dist.get("<7天", 0) / max(total, 1)
        if fresh_ratio > 0.3:
            score += 5

        return max(0.0, min(100.0, round(score, 1)))

    def _generate_recommendations(
        self,
        total: int,
        dup_count: int,
        age_dist: dict[str, int],
        expired_count: int,
        archived_count: int,
        health_score: float,
    ) -> list[str]:
        """生成优化建议"""
        recs: list[str] = []

        if dup_count > 0:
            recs.append(f"发现 {dup_count} 条重复记忆，建议执行 deduplicate() 清理")

        if age_dist.get(">180天", 0) > total * 0.3 and total > 10:
            recs.append("超过30%的记忆已存放>180天，建议执行 auto_expire() 清理旧数据")

        if expired_count > total * 0.2:
            recs.append(f"{expired_count} 条过期记忆可清理，释放存储空间")

        if archived_count > total * 0.5:
            recs.append("归档记忆占比过高(>50%)，考虑永久删除或导出备份")

        if total < 10:
            recs.append("记忆总量较低，建议导入更多数据以提升分析质量")

        if health_score < 60:
            recs.append("⚠️ 健康评分偏低，建议立即执行 auto_expire + deduplicate")

        if health_score >= 90:
            recs.append("✅ 记忆库状态优秀，继续保持当前管理策略")

        return recs
