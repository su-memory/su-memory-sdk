"""
贝叶斯网络与概率图模型

用于建模记忆间的因果概率依赖关系，支持：
1. 有向无环图 (DAG) 结构
2. 条件概率表 (CPT) 
3. 因果强度推断
4. 概率传播与推理
5. 信念传播 (Belief Propagation)

对外暴露：BayesianNetwork
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math
import json
import time

from .bayesian import BetaDistribution, BayesianEngine


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ProbabilisticEdge:
    """
    概率边 — 表示节点间的因果关系强度

    存储在条件概率表 (CPT) 中:
    P(child=positive | parent=positive) = strength
    P(child=positive | parent=negative) = baseline

    使用两个独立的 Beta 分布分别估计这两种情况的概率。
    """
    parent_id: str
    child_id: str
    # P(child | parent=positive)
    pos_given_pos: BetaDistribution = field(default_factory=BetaDistribution.uniform)
    # P(child | parent=negative)
    pos_given_neg: BetaDistribution = field(default_factory=BetaDistribution.uniform)

    edge_type: str = "causal"  # "causal" | "correlation" | "association"
    created_at: float = field(default_factory=time.time)
    evidence_count: int = 0

    @property
    def causal_strength(self) -> float:
        """
        因果强度 = P(child=pos | parent=pos) - P(child=pos | parent=neg)

        正值表示正向因果影响，负值表示抑制
        """
        return self.pos_given_pos.mean - self.pos_given_neg.mean

    @property
    def relative_risk(self) -> float:
        """相对风险 = P(child|pos) / P(child|neg)"""
        denom = max(self.pos_given_neg.mean, 1e-10)
        return self.pos_given_pos.mean / denom

    def update(self, parent_state: bool, child_state: bool, weight: float = 1.0):
        """更新条件概率估计"""
        self.evidence_count += 1
        if parent_state:
            if child_state:
                self.pos_given_pos.alpha += weight
            else:
                self.pos_given_pos.beta += weight
        else:
            if child_state:
                self.pos_given_neg.alpha += weight
            else:
                self.pos_given_neg.beta += weight

    def to_dict(self) -> Dict:
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "pos_given_pos": self.pos_given_pos.to_dict(),
            "pos_given_neg": self.pos_given_neg.to_dict(),
            "causal_strength": self.causal_strength,
            "relative_risk": self.relative_risk,
            "edge_type": self.edge_type,
            "evidence_count": self.evidence_count,
        }


@dataclass
class NetworkNode:
    """
    贝叶斯网络节点

    每个节点对应一个记忆或概念，维护：
    - 自身的信念分布 (Beta)
    - 入边/出边列表
    - 马尔可夫毯 (Markov Blanket)
    """
    node_id: str
    label: str = ""
    belief: BetaDistribution = field(default_factory=BetaDistribution.uniform)

    # 图结构
    parents: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)

    # 元信息
    node_type: str = "memory"   # "memory" | "event" | "concept" | "hypothesis"
    metadata: Dict = field(default_factory=dict)

    @property
    def markov_blanket(self) -> Set[str]:
        """
        马尔可夫毯 = 父节点 ∪ 子节点 ∪ 子节点的父节点

        给定马尔可夫毯，节点与网络中其他节点条件独立。
        """
        blanket = set(self.parents) | set(self.children)
        # 子节点的父节点（共父节点）
        # 注：这里需要在网络层面计算，节点只提供基础集合
        return blanket

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "belief": self.belief.to_dict(),
            "parents": list(self.parents),
            "children": list(self.children),
            "node_type": self.node_type,
        }


# ============================================================
# 信念传播引擎
# ============================================================

class BeliefPropagator:
    """
    信念传播 (Belief Propagation / Sum-Product Algorithm)

    在贝叶斯网络上执行概率推理：
    1. 给定证据节点，计算查询节点的后验边缘概率
    2. 使用消息传递算法（树状结构精确，一般结构近似）
    """

    def __init__(self, max_iterations: int = 10, damping: float = 0.5):
        """
        Args:
            max_iterations: 最大迭代次数
            damping: 阻尼因子（防止震荡）
        """
        self._max_iterations = max_iterations
        self._damping = damping

    def infer(
        self,
        network: 'BayesianNetwork',
        query_nodes: List[str],
        evidence: Dict[str, bool] = None
    ) -> Dict[str, BetaDistribution]:
        """
        给定证据，推断查询节点的后验概率

        使用 Loopy Belief Propagation (LBP):
        - 对于树状网络是精确的
        - 对于含环网络是近似的

        Args:
            network: 贝叶斯网络
            query_nodes: 要查询的节点ID列表
            evidence: 证据 {node_id: is_positive}

        Returns:
            {node_id: BetaDistribution} 后验概率
        """
        evidence = evidence or {}

        # 初始化消息
        # message[(from_node, to_node)] = BetaDistribution
        messages: Dict[Tuple[str, str], BetaDistribution] = {}

        # 迭代消息传递
        for iteration in range(self._max_iterations):
            old_messages = {k: v for k, v in messages.items()}
            max_delta = 0.0

            # 所有非证据节点向其邻居发送消息
            all_nodes = set(network._nodes.keys())
            for node_id in all_nodes:
                if node_id in evidence:
                    continue  # 证据节点固定

                for neighbor_id in network.get_neighbors(node_id):
                    # 收集来自其他邻居的消息（除了目标邻居）
                    incoming = self._collect_incoming(
                        network, node_id, neighbor_id, messages, evidence
                    )

                    # 计算传出消息
                    new_msg = self._compute_message(
                        network, node_id, neighbor_id, incoming, evidence
                    )

                    key = (node_id, neighbor_id)
                    if key in old_messages:
                        # 应用阻尼
                        old = old_messages[key]
                        alpha = self._damping * new_msg.alpha + (1 - self._damping) * old.alpha
                        beta = self._damping * new_msg.beta + (1 - self._damping) * old.beta
                        new_msg = BetaDistribution(alpha=alpha, beta=beta)

                    messages[key] = new_msg

                    if key in old_messages:
                        delta = abs(new_msg.mean - old_messages[key].mean)
                        max_delta = max(max_delta, delta)

            # 收敛检查
            if max_delta < 0.001:
                break

        # 计算查询节点的后验边缘概率
        results = {}
        for node_id in query_nodes:
            posterior = self._compute_marginal(network, node_id, messages, evidence)
            results[node_id] = posterior

        return results

    def _collect_incoming(
        self,
        network: 'BayesianNetwork',
        node_id: str,
        exclude_neighbor: str,
        messages: Dict[Tuple[str, str], BetaDistribution],
        evidence: Dict[str, bool]
    ) -> List[BetaDistribution]:
        """收集来自其他邻居的传入消息"""
        incoming = []
        for neighbor_id in network.get_neighbors(node_id):
            if neighbor_id == exclude_neighbor:
                continue

            if neighbor_id in evidence:
                # 证据节点直接提供信息
                val = evidence[neighbor_id]
                incoming.append(
                    BetaDistribution(alpha=10.0 if val else 1.0, beta=1.0 if val else 10.0)
                )
            else:
                msg = messages.get((neighbor_id, node_id))
                if msg:
                    incoming.append(msg)

        return incoming

    def _compute_message(
        self,
        network: 'BayesianNetwork',
        from_node: str,
        to_node: str,
        incoming: List[BetaDistribution],
        evidence: Dict[str, bool]
    ) -> BetaDistribution:
        """计算从 from_node 到 to_node 的消息"""
        # 自身的先验信念
        node = network._nodes.get(from_node)
        own_belief = node.belief if node else BetaDistribution.uniform()

        # 合并所有传入消息
        combined_alpha = own_belief.alpha
        combined_beta = own_belief.beta

        for msg in incoming:
            combined_alpha += msg.alpha - 1.0  # 减去先验
            combined_beta += msg.beta - 1.0

        # 确保正数
        combined_alpha = max(0.1, combined_alpha)
        combined_beta = max(0.1, combined_beta)

        combined = BetaDistribution(alpha=combined_alpha, beta=combined_beta)

        # 应用因果传递
        edge = network.get_edge(from_node, to_node)
        if edge:
            # P(to=pos | from) = P(to=pos|from=pos)*P(from=pos) + P(to=pos|from=neg)*P(from=neg)
            prob_to = (
                edge.pos_given_pos.mean * combined.mean +
                edge.pos_given_neg.mean * (1 - combined.mean)
            )
            strength = max(combined.effective_sample_size * prob_to, 2.0)
            return BetaDistribution(
                alpha=prob_to * strength,
                beta=(1 - prob_to) * strength
            )

        return combined

    def _compute_marginal(
        self,
        network: 'BayesianNetwork',
        node_id: str,
        messages: Dict[Tuple[str, str], BetaDistribution],
        evidence: Dict[str, bool]
    ) -> BetaDistribution:
        """计算节点的后验边缘概率"""
        if node_id in evidence:
            val = evidence[node_id]
            return BetaDistribution(
                alpha=100.0 if val else 1.0,
                beta=1.0 if val else 100.0
            )

        node = network._nodes.get(node_id)
        own_belief = node.belief if node else BetaDistribution.uniform()

        combined_alpha = own_belief.alpha
        combined_beta = own_belief.beta

        for neighbor_id in network.get_neighbors(node_id):
            if neighbor_id in evidence:
                val = evidence[neighbor_id]
                combined_alpha += (10.0 if val else 1.0) - 1.0
                combined_beta += (1.0 if val else 10.0) - 1.0
            else:
                msg = messages.get((neighbor_id, node_id))
                if msg:
                    combined_alpha += msg.alpha - 1.0
                    combined_beta += msg.beta - 1.0

        combined_alpha = max(0.1, combined_alpha)
        combined_beta = max(0.1, combined_beta)
        return BetaDistribution(alpha=combined_alpha, beta=combined_beta)


# ============================================================
# 贝叶斯网络
# ============================================================

class BayesianNetwork:
    """
    贝叶斯网络 - 对外唯一接口

    功能：
    1. DAG 结构管理（添加/删除节点和边）
    2. 条件概率表学习（从观测数据更新）
    3. 因果强度计算
    4. 信念传播推理
    5. 最可能解释 (MPE)
    6. 敏感性分析
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._nodes: Dict[str, NetworkNode] = {}
        self._edges: Dict[Tuple[str, str], ProbabilisticEdge] = {}
        self._propagator = BeliefPropagator()

        # 与 BayesianEngine 联动
        self._bayesian_engine: Optional[BayesianEngine] = None

    def set_bayesian_engine(self, engine: BayesianEngine):
        """关联贝叶斯引擎（用于自动同步信念状态）"""
        self._bayesian_engine = engine

    # ---- 结构管理 ----

    def add_node(
        self,
        node_id: str,
        label: str = "",
        node_type: str = "memory",
        prior_belief: float = 0.5,
        metadata: Dict = None
    ) -> NetworkNode:
        """
        添加节点

        Args:
            node_id: 节点唯一标识
            label: 标签
            node_type: 类型
            prior_belief: 先验信念
            metadata: 元信息

        Returns:
            NetworkNode
        """
        if node_id in self._nodes:
            return self._nodes[node_id]

        belief = BetaDistribution.weak_informative(prior_belief, strength=2.0)
        node = NetworkNode(
            node_id=node_id,
            label=label,
            belief=belief,
            node_type=node_type,
            metadata=metadata or {}
        )
        self._nodes[node_id] = node
        return node

    def add_edge(
        self,
        parent_id: str,
        child_id: str,
        edge_type: str = "causal",
        initial_strength: float = 0.5
    ) -> ProbabilisticEdge:
        """
        添加有向边 parent → child

        自动创建不存在的节点。

        Args:
            parent_id: 父节点ID
            child_id: 子节点ID
            edge_type: 边类型
            initial_strength: 初始因果强度 (0-1)

        Returns:
            ProbabilisticEdge

        Raises:
            ValueError: 如果添加此边会导致环路
        """
        # 确保节点存在
        self.add_node(parent_id)
        self.add_node(child_id)

        # 环路检测
        if self._would_create_cycle(parent_id, child_id):
            raise ValueError(f"添加边 {parent_id}→{child_id} 会导致环路")

        key = (parent_id, child_id)
        if key in self._edges:
            return self._edges[key]

        edge = ProbabilisticEdge(
            parent_id=parent_id,
            child_id=child_id,
            edge_type=edge_type,
            pos_given_pos=BetaDistribution.weak_informative(initial_strength, strength=2.0),
            pos_given_neg=BetaDistribution.weak_informative(0.3, strength=2.0),
        )
        self._edges[key] = edge

        # 更新节点的邻接关系
        self._nodes[parent_id].children.add(child_id)
        self._nodes[child_id].parents.add(parent_id)

        return edge

    def remove_node(self, node_id: str):
        """移除节点及其所有关联边"""
        if node_id not in self._nodes:
            return

        # 移除所有关联边
        edges_to_remove = [
            (p, c) for (p, c) in self._edges
            if p == node_id or c == node_id
        ]
        for key in edges_to_remove:
            self._remove_edge_internal(*key)

        del self._nodes[node_id]

    def remove_edge(self, parent_id: str, child_id: str):
        """移除边"""
        self._remove_edge_internal(parent_id, child_id)

    def _remove_edge_internal(self, parent_id: str, child_id: str):
        key = (parent_id, child_id)
        if key in self._edges:
            del self._edges[key]
            if parent_id in self._nodes:
                self._nodes[parent_id].children.discard(child_id)
            if child_id in self._nodes:
                self._nodes[child_id].parents.discard(parent_id)

    def _would_create_cycle(self, new_parent: str, new_child: str) -> bool:
        """
        检测添加 new_parent→new_child 是否会引入环路

        使用 BFS 从 new_child 出发，检查是否能到达 new_parent
        """
        if new_parent not in self._nodes or new_child not in self._nodes:
            return False

        visited = set()
        queue = deque([new_child])

        while queue:
            current = queue.popleft()
            if current == new_parent:
                return True
            if current in visited:
                continue
            visited.add(current)

            node = self._nodes.get(current)
            if node:
                for child in node.children:
                    if child not in visited:
                        queue.append(child)

        return False

    def get_node(self, node_id: str) -> Optional[NetworkNode]:
        return self._nodes.get(node_id)

    def get_edge(self, parent_id: str, child_id: str) -> Optional[ProbabilisticEdge]:
        return self._edges.get((parent_id, child_id))

    def get_neighbors(self, node_id: str) -> Set[str]:
        """获取节点的所有邻居"""
        node = self._nodes.get(node_id)
        if not node:
            return set()
        return node.parents | node.children

    def get_parents(self, node_id: str) -> Set[str]:
        node = self._nodes.get(node_id)
        return node.parents if node else set()

    def get_children(self, node_id: str) -> Set[str]:
        node = self._nodes.get(node_id)
        return node.children if node else set()

    # ---- 证据更新 ----

    def observe(
        self,
        parent_id: str,
        child_id: str,
        parent_state: bool,
        child_state: bool,
        weight: float = 1.0
    ):
        """
        观测父-子状态对，更新条件概率表

        Args:
            parent_id: 父节点ID
            child_id: 子节点ID
            parent_state: 父节点的状态 (True=positive, False=negative)
            child_state: 子节点的状态
            weight: 证据权重
        """
        edge = self.get_edge(parent_id, child_id)
        if not edge:
            # 自动创建边
            edge = self.add_edge(parent_id, child_id)

        edge.update(parent_state, child_state, weight)

        # 同时更新节点的自身信念
        for node_id, state in [(parent_id, parent_state), (child_id, child_state)]:
            node = self._nodes.get(node_id)
            if node:
                if state:
                    node.belief.alpha += weight * 0.5
                else:
                    node.belief.beta += weight * 0.5

    def batch_observe(
        self,
        observations: List[Dict]
    ):
        """批量观测"""
        for obs in observations:
            self.observe(
                parent_id=obs["parent_id"],
                child_id=obs["child_id"],
                parent_state=obs.get("parent_state", True),
                child_state=obs.get("child_state", True),
                weight=obs.get("weight", 1.0)
            )

    # ---- 推理查询 ----

    def query_causal_strength(self, parent_id: str, child_id: str) -> Optional[Dict]:
        """查询因果强度"""
        edge = self.get_edge(parent_id, child_id)
        if not edge:
            return None
        return {
            "parent": parent_id,
            "child": child_id,
            "p_child_given_parent_pos": edge.pos_given_pos.mean,
            "p_child_given_parent_neg": edge.pos_given_neg.mean,
            "causal_strength": edge.causal_strength,
            "relative_risk": edge.relative_risk,
            "evidence_count": edge.evidence_count,
            "edge_type": edge.edge_type
        }

    def infer_posterior(
        self,
        query_nodes: List[str],
        evidence: Dict[str, bool] = None
    ) -> Dict[str, BetaDistribution]:
        """
        给定证据，推断查询节点的后验概率

        使用信念传播算法

        Args:
            query_nodes: 要查询的节点
            evidence: 证据字典

        Returns:
            {node_id: BetaDistribution}
        """
        return self._propagator.infer(self, query_nodes, evidence)

    def most_probable_explanation(
        self,
        evidence: Dict[str, bool] = None,
        candidate_nodes: List[str] = None
    ) -> Dict[str, Any]:
        """
        最可能解释 (MPE)

        给定证据，找最可能的节点状态组合

        Args:
            evidence: 已知证据
            candidate_nodes: 候选节点（None=所有非证据节点）

        Returns:
            {
                "states": {node_id: bool},
                "probability": float
            }
        """
        evidence = evidence or {}

        if candidate_nodes is None:
            candidate_nodes = [
                nid for nid in self._nodes
                if nid not in evidence
            ]

        # 计算每个候选节点的后验概率
        posteriors = self._propagator.infer(self, candidate_nodes, evidence)

        # 贪心赋值：每个节点取最大后验
        states = {}
        total_log_prob = 0.0
        for node_id, posterior in posteriors.items():
            if posterior.mean >= 0.5:
                states[node_id] = True
                total_log_prob += math.log(max(posterior.mean, 1e-10))
            else:
                states[node_id] = False
                total_log_prob += math.log(max(1 - posterior.mean, 1e-10))

        return {
            "states": states,
            "log_probability": total_log_prob,
            "posteriors": {nid: p.to_dict() for nid, p in posteriors.items()}
        }

    def sensitivity_analysis(
        self,
        query_node: str,
        evidence_node: str,
        steps: int = 11
    ) -> List[Dict]:
        """
        敏感性分析

        分析证据节点不同状态对查询节点后验概率的影响

        Args:
            query_node: 查询节点
            evidence_node: 证据节点（要变化的节点）
            steps: 采样步数

        Returns:
            [{evidence_prob, query_posterior}, ...]
        """
        results = []

        for i in range(steps):
            p = i / (steps - 1)  # 0.0 → 1.0

            # 设置证据节点的"软证据"（通过固定 Beta 参数模拟）
            # 用强证据模拟：alpha=p*100, beta=(1-p)*100
            temp_evidence = {
                evidence_node: p >= 0.5  # 硬分类用于 BP
            }

            posteriors = self._propagator.infer(self, [query_node], temp_evidence)
            query_post = posteriors.get(query_node)

            results.append({
                "evidence_prob": p,
                "query_posterior_mean": query_post.mean if query_post else None,
                "query_posterior_std": query_post.std if query_post else None,
            })

        return results

    def find_most_influential_parents(
        self,
        node_id: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        找出对目标节点影响最大的父节点

        基于因果强度的绝对值排序

        Args:
            node_id: 目标节点
            top_k: 返回数量

        Returns:
            [{parent_id, causal_strength, relative_risk}, ...]
        """
        parents = self.get_parents(node_id)
        strengths = []

        for parent_id in parents:
            strength = self.query_causal_strength(parent_id, node_id)
            if strength:
                strengths.append(strength)

        strengths.sort(key=lambda x: abs(x["causal_strength"]), reverse=True)
        return strengths[:top_k]

    # ---- 因果链分析 ----

    def trace_causal_chain(
        self,
        cause_node: str,
        effect_node: str,
        max_depth: int = 5
    ) -> List[Dict]:
        """
        追踪因果链（BFS）

        从 cause_node 出发，找到通往 effect_node 的所有路径

        Returns:
            [{path: [node_ids], chain_strength: float}, ...]
        """
        chains = []

        def backtrack(current: str, path: List[str], cumulative_strength: float):
            if len(path) > max_depth:
                return
            if current == effect_node:
                chains.append({
                    "path": list(path),
                    "chain_strength": cumulative_strength
                })
                return

            node = self._nodes.get(current)
            if not node:
                return

            for child in node.children:
                if child not in path:
                    edge = self.get_edge(current, child)
                    edge_strength = abs(edge.causal_strength) if edge else 0.1
                    backtrack(
                        child,
                        path + [child],
                        cumulative_strength * max(edge_strength, 0.01)
                    )

        backtrack(cause_node, [cause_node], 1.0)

        # 按链强度排序
        chains.sort(key=lambda x: x["chain_strength"], reverse=True)
        return chains

    # ---- 统计与分析 ----

    def get_statistics(self) -> Dict:
        """获取网络统计信息"""
        total_nodes = len(self._nodes)
        total_edges = len(self._edges)

        edge_types = defaultdict(int)
        causal_strengths = []
        for edge in self._edges.values():
            edge_types[edge.edge_type] += 1
            causal_strengths.append(edge.causal_strength)

        return {
            "name": self.name,
            "node_count": total_nodes,
            "edge_count": total_edges,
            "edge_type_distribution": dict(edge_types),
            "mean_causal_strength": (
                sum(causal_strengths) / len(causal_strengths)
                if causal_strengths else 0.0
            ),
            "max_causal_strength": max(causal_strengths) if causal_strengths else 0.0,
            "is_dag": not self._has_cycle(),
        }

    def _has_cycle(self) -> bool:
        """检测网络中是否存在环路（使用 DFS）"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self._nodes}

        def dfs(node_id: str) -> bool:
            color[node_id] = GRAY
            node = self._nodes[node_id]
            for child in node.children:
                if color[child] == GRAY:
                    return True
                if color[child] == WHITE and dfs(child):
                    return True
            color[node_id] = BLACK
            return False

        for nid in self._nodes:
            if color[nid] == WHITE:
                if dfs(nid):
                    return True
        return False

    # ---- 持久化 ----

    def to_dict(self) -> Dict:
        """序列化"""
        return {
            "name": self.name,
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "edges": {f"{p}→{c}": e.to_dict() for (p, c), e in self._edges.items()},
            "statistics": self.get_statistics()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> 'BayesianNetwork':
        net = cls(name=d.get("name", "default"))
        # 恢复节点
        for nid, nd in d.get("nodes", {}).items():
            net.add_node(
                node_id=nid,
                label=nd.get("label", ""),
                node_type=nd.get("node_type", "memory")
            )
        # 恢复边
        for key, ed in d.get("edges", {}).items():
            parent, child = key.split("→")
            edge = net.add_edge(
                parent_id=parent,
                child_id=child,
                edge_type=ed.get("edge_type", "causal")
            )
            edge.pos_given_pos = BetaDistribution.from_dict(ed["pos_given_pos"])
            edge.pos_given_neg = BetaDistribution.from_dict(ed["pos_given_neg"])
            edge.evidence_count = ed.get("evidence_count", 0)
        return net

    @classmethod
    def from_json(cls, json_str: str) -> 'BayesianNetwork':
        return cls.from_dict(json.loads(json_str))

    def reset(self):
        """重置网络"""
        self._nodes.clear()
        self._edges.clear()
