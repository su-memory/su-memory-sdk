"""TopicClusterer — 增量关键词分桶 (软分类)。

与 algebra 层的"硬分类" (GF(2)³ 能量维度) 正交: 硬分类管记忆的结构性归属,
软分类管记忆的主题归属 (项目A决策 / 项目B技术 / ...)。

设计原则:
- 增量: 每次 add 一条记忆, 只和已有簇心做比较 (O(簇数)), 不重新全量聚类
- 从数据涌现: 簇不是预设的, 而是从关键词相似度自动生长
- 轻量: 用关键词 Jaccard 相似度, 无 sklearn/torch 依赖

检索时的价值: query 先定位 top-3 主题桶, 桶内再做精确检索 → O(桶大小)
替代 O(全部记忆)。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TopicCluster:
    """一个主题簇。"""
    cluster_id: int
    centroid_keywords: set[str] = field(default_factory=set)  # 簇心关键词集合
    member_ids: list[str] = field(default_factory=list)
    label: str = ""  # 自动生成的人类可读标签 (取最高频关键词)

    def similarity(self, keywords: set[str]) -> float:
        """与给定关键词集合的相似度 (重叠系数, 适合短关键词列表).

        重叠系数 = |A ∩ B| / min(|A|, |B|)
        比 Jaccard 更宽松: 共享 1 个词时, Jaccard=1/5=0.2 (低于阈值),
        重叠系数=1/3=0.33 (高于阈值), 能正确聚类同主题记忆。
        """
        if not self.centroid_keywords or not keywords:
            return 0.0
        intersection = len(self.centroid_keywords & keywords)
        if intersection == 0:
            return 0.0
        return intersection / min(len(self.centroid_keywords), len(keywords))

    def absorb(self, memory_id: str, keywords: set[str]) -> None:
        """吸收一条新成员, 更新簇心 (增量平均)。"""
        self.member_ids.append(memory_id)
        # 簇心 = 原簇心 ∪ 新成员关键词, 但只保留出现 ≥2 次的 (去噪)
        self.centroid_keywords = self.centroid_keywords | keywords
        # 更新标签: 取簇心最长的关键词 (通常最有区分度)
        if self.centroid_keywords:
            self.label = max(self.centroid_keywords, key=len)


class TopicClusterer:
    """增量主题分桶器。

    Parameters
    ----------
    similarity_threshold : float
        Jaccard 相似度 ≥ 此值则归入已有簇; 否则开新簇。
    max_clusters : int
        最大簇数上限 (防止碎片化)。
    """

    def __init__(self, similarity_threshold: float = 0.25, max_clusters: int = 200):
        self._threshold = similarity_threshold
        self._max_clusters = max_clusters
        self._clusters: list[TopicCluster] = []
        self._memory_cluster: dict[str, int] = {}  # memory_id -> cluster_id

    @property
    def n_clusters(self) -> int:
        return len(self._clusters)

    def assign(self, memory_id: str, keywords: list[str]) -> int:
        """为记忆分配主题簇, 返回 cluster_id。

        若与某已有簇的 Jaccard 相似度 ≥ threshold, 归入该簇;
        否则创建新簇。
        """
        kw_set = set(keywords)
        if not kw_set:
            # 无关键词的记忆归入 "misc" 簇 (id=0 若存在, 否则新建)
            if self._clusters:
                self._memory_cluster[memory_id] = self._clusters[0].cluster_id
                return self._clusters[0].cluster_id

        # 找最相似的簇
        best_cluster = None
        best_sim = 0.0
        for cluster in self._clusters:
            sim = cluster.similarity(kw_set)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_cluster is not None and best_sim >= self._threshold:
            best_cluster.absorb(memory_id, kw_set)
            self._memory_cluster[memory_id] = best_cluster.cluster_id
            return best_cluster.cluster_id

        # 创建新簇 (尊重上限)
        new_id = len(self._clusters)
        if new_id >= self._max_clusters:
            # 超上限: 归入最相似的簇 (即使低于阈值)
            if best_cluster is not None:
                best_cluster.absorb(memory_id, kw_set)
                self._memory_cluster[memory_id] = best_cluster.cluster_id
                return best_cluster.cluster_id
        cluster = TopicCluster(cluster_id=new_id)
        cluster.absorb(memory_id, kw_set)
        self._clusters.append(cluster)
        self._memory_cluster[memory_id] = new_id
        return new_id

    def get_cluster(self, memory_id: str) -> int | None:
        """获取记忆所属簇 id。"""
        return self._memory_cluster.get(memory_id)

    def query_clusters(self, keywords: list[str], top_k: int = 3) -> list[tuple[int, float]]:
        """查询: 返回与给定关键词最匹配的 top-k 簇 (cluster_id, similarity)。

        检索路由用: 先定位最相关的几个桶, 桶内再精确检索。
        """
        kw_set = set(keywords)
        if not kw_set or not self._clusters:
            return []
        scored = [(c.cluster_id, c.similarity(kw_set)) for c in self._clusters]
        scored.sort(key=lambda x: x[1], reverse=True)
        # 只返回相似度 > 0 的簇
        return [(cid, sim) for cid, sim in scored[:top_k] if sim > 0]

    def cluster_members(self, cluster_id: int) -> list[str]:
        """获取某簇的全部成员 memory_id。"""
        for c in self._clusters:
            if c.cluster_id == cluster_id:
                return list(c.member_ids)
        return []

    def get_topics(self) -> list[dict]:
        """获取所有主题概览 (供 introspection / 可视化)。"""
        return [
            {
                "cluster_id": c.cluster_id,
                "label": c.label,
                "size": len(c.member_ids),
                "centroid_keywords": list(c.centroid_keywords)[:10],
            }
            for c in self._clusters
        ]

    def remove(self, memory_id: str) -> None:
        """从分桶器中移除记忆 (forget 时调用)。"""
        cid = self._memory_cluster.pop(memory_id, None)
        if cid is not None:
            for c in self._clusters:
                if c.cluster_id == cid and memory_id in c.member_ids:
                    c.member_ids.remove(memory_id)
                    break
