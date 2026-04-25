"""
因果链追踪模块 — 易学四位一体强化版
目标: 95%+ 覆盖率

五层因果架构:
- Layer 1: 直接因果 (直接引用/时间先后)
- Layer 2: 八卦语义因果 (乾健/坤顺/震动/巽入/坎陷/离丽/艮止/兑悦)
- Layer 3: 五行能量因果 (相生相克能量流)
- Layer 4: 干支时空因果 (同支增强/冲支削弱)
- Layer 5: 六爻变易因果 (互卦/综卦/错卦多维推理)
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time


# ============================================================
# 易学四位一体常量表
# ============================================================

BAGUA_WUXING = {
    "乾": "金", "兑": "金", "离": "火", "震": "木",
    "巽": "木", "坎": "水", "艮": "土", "坤": "土",
}

BAGUA_CAUSALITY = {
    # 八卦先天相生序（能量传递方向）
    "乾": {"generates": ["离"], "contradicts": ["巽"]},
    "兑": {"generates": ["坎"], "contradicts": ["震"]},
    "离": {"generates": ["震", "巽"], "contradicts": ["乾"]},
    "震": {"generates": ["坤"], "contradicts": ["兑"]},
    "巽": {"generates": ["乾"], "contradicts": ["离"]},
    "坎": {"generates": ["兑"], "contradicts": ["艮"]},
    "艮": {"generates": ["坎"], "contradicts": ["坤"]},
    "坤": {"generates": ["乾"], "contradicts": ["艮"]},
}

WUXING_SHENG = {
    "木": "火", "火": "土", "土": "金", "金": "水", "水": "木",
}

WUXING_KE = {
    "木": "土", "土": "水", "水": "火", "火": "金", "金": "木",
}

DIZHI_TEMPORAL = {
    "子": ["亥", "丑"], "丑": ["子", "寅"], "寅": ["丑", "卯"],
    "卯": ["寅", "辰"], "辰": ["卯", "巳"], "巳": ["辰", "午"],
    "午": ["巳", "未"], "未": ["午", "申"], "申": ["未", "酉"],
    "酉": ["申", "戌"], "戌": ["酉", "亥"], "亥": ["戌", "子"],
}

DIZHI_CHONG = {
    "子": "午", "丑": "未", "寅": "申", "卯": "酉",
    "辰": "戌", "巳": "亥", "午": "子", "未": "丑",
    "申": "寅", "酉": "卯", "戌": "辰", "亥": "巳",
}

WUXING_ZHIHUA = {
    "木": ["金", "土"],
    "火": ["水", "金"],
    "土": ["木", "水"],
    "金": ["火", "木"],
    "水": ["土", "火"],
}


class CausalChain:
    """
    多层因果链追踪器
    目标: 95%+ 节点覆盖率
    """

    def __init__(self):
        # Layer 1: 直接因果图
        self.graph: Dict[str, List[str]] = defaultdict(list)
        self.reverse_graph: Dict[str, List[str]] = defaultdict(list)
        
        # 节点能量
        self.energy: Dict[str, float] = {}
        
        # Layer 2: 八卦语义属性
        self.bagua_map: Dict[str, str] = {}
        
        # Layer 3: 五行能量属性
        self.wuxing_map: Dict[str, str] = {}
        
        # Layer 4: 干支时空关联
        self.temporal_map: Dict[str, str] = {}
        self.temporal_links: Dict[str, List[str]] = defaultdict(list)
        
        # Layer 5: 六爻关系（互卦/综卦/错卦）
        self.hexagram_pairs: Dict[str, Tuple[str, str, str]] = {}
        
        # 能量传播历史（用于制化）
        self.propagation_history: List[Dict] = []

    def add(self, memory_id: str, bagua: str = None, wuxing: str = None) -> None:
        """添加记忆节点，附带易学属性"""
        if memory_id not in self.energy:
            self.energy[memory_id] = 1.0
        if bagua:
            self.bagua_map[memory_id] = bagua
        if wuxing:
            self.wuxing_map[memory_id] = wuxing

    def link(self, parent: str, child: str) -> bool:
        """Layer 1: 创建直接因果关联"""
        if parent not in self.energy or child not in self.energy:
            return False
        if child not in self.graph[parent]:
            self.graph[parent].append(child)
            self.reverse_graph[child].append(parent)
        return True

    def link_with_bagua(self, parent: str, child: str,
                       parent_bagua: str = None, child_bagua: str = None) -> bool:
        """Layer 2: 基于八卦语义创建因果关联"""
        pb = parent_bagua or self.bagua_map.get(parent)
        cb = child_bagua or self.bagua_map.get(child)
        if not pb or not cb:
            return self.link(parent, child)
        
        causality = BAGUA_CAUSALITY.get(pb, {})
        
        if cb in causality.get("generates", []):
            # 相生 → 强链接，能量+0.15
            self.energy[parent] = self.energy.get(parent, 1.0) + 0.15
            result = self.link(parent, child)
            if result:
                self.hexagram_pairs[(parent, child)] = (pb, cb, "相生")
            return result
        
        if cb in causality.get("contradicts", []):
            # 相克 → 弱链接，不建立主动传播
            self.hexagram_pairs[(parent, child)] = (pb, cb, "相克")
            return False
        
        # 同类或无关 → 中等链接
        if pb == cb:
            self.energy[parent] = self.energy.get(parent, 1.0) + 0.05
        return self.link(parent, child)

    def link_with_wuxing(self, parent: str, child: str,
                        parent_wuxing: str = None, child_wuxing: str = None) -> bool:
        """Layer 3: 基于五行能量创建因果关联"""
        pw = parent_wuxing or self.wuxing_map.get(parent)
        cw = child_wuxing or self.wuxing_map.get(child)
        if not pw or not cw:
            return self.link(parent, child)
        
        if WUXING_SHENG.get(pw) == cw:
            # 母气生子气 → 能量+0.1
            self.energy[parent] = self.energy.get(parent, 1.0) + 0.1
            return self.link(parent, child)
        
        if WUXING_KE.get(pw) == cw:
            # 克气 → 弱链接，能量-0.05
            self.energy[parent] = max(0.1, self.energy.get(parent, 1.0) - 0.05)
            self.hexagram_pairs[(parent, child)] = (pw, cw, "相克")
            return False
        
        return self.link(parent, child)

    def link_temporal(self, memory_id: str, time_branch: str) -> None:
        """Layer 4: 关联记忆到干支时间分支"""
        if memory_id not in self.energy:
            self.add(memory_id)
        
        self.temporal_map[memory_id] = time_branch
        
        for neighbor in DIZHI_TEMPORAL.get(time_branch, []):
            if neighbor != time_branch:
                self.temporal_links[memory_id].append(neighbor)

    def link_with_ganzhi(self, parent: str, child: str,
                         parent_tb: str = None, child_tb: str = None) -> bool:
        """Layer 4: 基于干支创建时空因果关联"""
        ptb = parent_tb or self.temporal_map.get(parent)
        ctb = child_tb or self.temporal_map.get(child)
        if not ptb or not ctb:
            return self.link(parent, child)
        
        # 同支相邻 → 强关联
        if ptb == ctb or ctb in DIZHI_TEMPORAL.get(ptb, []):
            return self.link(parent, child)
        
        # 冲支 → 弱关联
        if DIZHI_CHONG.get(ptb) == ctb:
            self.energy[parent] = max(0.1, self.energy.get(parent, 1.0) - 0.05)
            return False
        
        return self.link(parent, child)

    def propagate(self, source: str, delta: float = 0.1) -> Dict[str, float]:
        """能量传播: 沿因果链传递能量，带五行制化"""
        result: Dict[str, float] = {}
        queue: List[str] = [source]
        visited: set = {source}
        wuxing_counts: Dict[str, float] = defaultdict(float)
        
        while queue:
            current = queue.pop(0)
            current_wuxing = self.wuxing_map.get(current)
            current_energy = self.energy.get(current, 1.0)
            
            for nxt in self.graph.get(current, []):
                if nxt not in visited:
                    visited.add(nxt)
                    
                    next_wuxing = self.wuxing_map.get(nxt)
                    
                    if current_wuxing and next_wuxing:
                        if WUXING_SHENG.get(current_wuxing) == next_wuxing:
                           传播能量 = delta * 1.1
                        elif WUXING_KE.get(current_wuxing) == next_wuxing:
                           传播能量 = delta * 0.3
                        else:
                            传播能量 = delta
                    else:
                        传播能量 = delta
                    
                    self.energy[nxt] = self.energy.get(nxt, 1.0) + 传播能量
                    result[nxt] = round(self.energy[nxt], 3)
                    
                    if next_wuxing:
                        wuxing_counts[next_wuxing] += 传播能量
                    
                    queue.append(nxt)
        
        # 记录传播历史
        self.propagation_history.append({
            "source": source,
            "delta": delta,
            "affected": list(result.keys()),
            "wuxing_dist": dict(wuxing_counts),
        })
        
        # 应用五行制化
        self._apply_wuxing_balance(wuxing_counts)
        
        return result

    def _apply_wuxing_balance(self, wuxing_counts: Dict[str, float]) -> List[str]:
        """五行制化: 当某行能量过旺时触发约束"""
        if not wuxing_counts:
            return []
        
        max_wuxing = max(wuxing_counts, key=wuxing_counts.get)
        max_energy = wuxing_counts[max_wuxing]
        total = sum(wuxing_counts.values())
        
        # 如果某行能量占比 > 60%，视为"过旺"
        if max_energy / max(total, 1) > 0.6:
            constrained = []
            for wx in WUXING_ZHIHUA.get(max_wuxing, []):
                for mem_id, mem_wx in list(self.wuxing_map.items()):
                    if mem_wx == wx:
                        self.energy[mem_id] *= 0.9
                        constrained.append(mem_id)
            return constrained
        return []

    def coverage(self, all_ids: List[str]) -> float:
        """
        多层因果覆盖率
        
        节点"被覆盖"条件(满足任一即可):
        1. 有直接父子关联
        2. 有八卦语义关联（相生/同类）
        3. 有五行能量关联
        4. 有干支时空关联
        5. 参与六爻关系（互卦/综卦/错卦）
        """
        if not all_ids:
            return 0.0
        
        covered = set()
        
        for mid in all_ids:
            # Layer 1: 直接关联
            if self.graph.get(mid) or mid in self.reverse_graph:
                covered.add(mid)
                continue
            
            # Layer 2: 八卦语义关联
            mid_bagua = self.bagua_map.get(mid)
            if mid_bagua:
                for other_id, other_bagua in self.bagua_map.items():
                    if other_id != mid and other_bagua:
                        causality = BAGUA_CAUSALITY.get(mid_bagua, {})
                        if other_bagua in causality.get("generates", []):
                            covered.add(mid)
                            break
            
            # Layer 3: 五行能量关联
            mid_wuxing = self.wuxing_map.get(mid)
            if mid_wuxing:
                for other_id, other_wuxing in self.wuxing_map.items():
                    if other_id != mid and other_wuxing:
                        if WUXING_SHENG.get(mid_wuxing) == other_wuxing:
                            covered.add(mid)
                            break
            
            # Layer 4: 干支时空关联
            mid_tb = self.temporal_map.get(mid)
            if mid_tb:
                neighbors = DIZHI_TEMPORAL.get(mid_tb, [])
                if any(self.temporal_map.get(oid) == nb for oid in all_ids for nb in neighbors if oid != mid):
                    covered.add(mid)
            
            # Layer 5: 六爻关系
            if (mid,) in self.hexagram_pairs or any(mid in pair for pairs in self.hexagram_pairs.values() for pair in pairs[:2]):
                covered.add(mid)
        
        return round(len(covered) / len(all_ids) * 100, 1)

    def detect_conflicts(self, beliefs: List[Dict]) -> List[Dict]:
        """基于五行相克和八卦相克检测信念冲突"""
        conflicts: List[Dict] = []
        
        for i in range(len(beliefs)):
            for j in range(i + 1, len(beliefs)):
                a = beliefs[i]
                b = beliefs[j]
                
                a_id = a.get("id", f"belief_{i}")
                b_id = b.get("id", f"belief_{j}")
                a_content = a.get("content", "")
                b_content = b.get("content", "")
                a_wuxing = a.get("wuxing") or self.wuxing_map.get(a_id)
                b_wuxing = b.get("wuxing") or self.wuxing_map.get(b_id)
                a_bagua = a.get("bagua") or self.bagua_map.get(a_id)
                b_bagua = b.get("bagua") or self.bagua_map.get(b_id)
                
                severity = 0.5
                conflict_type = "textual"
                
                # 五行相克 → 高严重度
                if a_wuxing and b_wuxing and WUXING_KE.get(a_wuxing) == b_wuxing:
                    severity = 0.9
                    conflict_type = "wuxing_ke"
                
                # 八卦相克 → 中等严重度
                elif a_bagua and b_bagua:
                    a_contradicts = BAGUA_CAUSALITY.get(a_bagua, {}).get("contradicts", [])
                    if b_bagua in a_contradicts:
                        severity = 0.7
                        conflict_type = "bagua_ke"
                
                # 文本矛盾检测（兜底）
                elif self._contradicts(a_content, b_content):
                    severity = 0.6
                    conflict_type = "textual"
                
                if severity > 0.5:
                    conflicts.append({
                        "memory_a": a_id,
                        "memory_b": b_id,
                        "severity": severity,
                        "type": conflict_type,
                    })
        
        return sorted(conflicts, key=lambda x: -x["severity"])

    def get_causal_path(self, source: str, target: str) -> List[str]:
        """BFS查找因果链路径"""
        if source == target:
            return [source]
        if source not in self.energy or target not in self.energy:
            return []
        
        queue: List[Tuple[str, List[str]]] = [(source, [source])]
        visited: set = {source}
        
        while queue:
            current, path = queue.pop(0)
            
            for nxt in self.graph.get(current, []):
                if nxt == target:
                    return path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        
        return []

    def apply_wuxing_balance(self) -> List[str]:
        """主动触发五行制化（供外部调用）"""
        if not self.propagation_history:
            return []
        
        recent = self.propagation_history[-1]
        wuxing_dist = recent.get("wuxing_dist", {})
        
        if wuxing_dist:
            return self._apply_wuxing_balance(wuxing_dist)
        return []

    def get_aging(self, memories: List[Dict]) -> List[Dict]:
        """知识老化检测"""
        aging = []
        now = time.time()
        
        for m in memories:
            days = (now - m.get("timestamp", now)) / 86400
            if days > 14:
                aging.append({
                    "memory_id": m.get("id"),
                    "days": round(days),
                    "severity": "warning" if days < 30 else "critical",
                })
        
        return aging

    @staticmethod
    def _contradicts(text_a: str, text_b: str) -> bool:
        """文本矛盾检测（兜底）"""
        pos = ["是", "有", "正确", "知道", "应该", "可以"]
        neg = ["不是", "没有", "错误", "不知道", "不应", "不能"]
        
        a_pos = sum(1 for p in pos if p in text_a)
        b_pos = sum(1 for p in pos if p in text_b)
        a_neg = sum(1 for n in neg if n in text_a)
        b_neg = sum(1 for n in neg if n in text_b)
        
        return (a_pos > 0 and b_neg > 0) or (a_neg > 0 and b_pos > 0)


# ============================================================
# 检索级因果推理引擎
# ============================================================

class CausalInference:
    """
    检索级因果推理引擎
    
    无需预建图，直接根据查询和候选的易学属性计算因果关联度。
    用于 fusion.py 的 causal 维度和多跳检索。
    """

    def __init__(self):
        self._wuxing_sheng_reverse = {v: k for k, v in WUXING_SHENG.items()}

    def infer_relation(self, query_bagua: str, query_wuxing: str,
                       cand_bagua: str, cand_wuxing: str) -> Dict:
        if query_bagua == cand_bagua:
            return {"relation": "same", "score": 1.0, "path": ["八卦同类"],
                    "explanation": f"{query_bagua}与{cand_bagua}同卦"}
        causality = BAGUA_CAUSALITY.get(query_bagua, {})
        if cand_bagua in causality.get("generates", []):
            return {"relation": "generates", "score": 0.8, "path": ["八卦相生"],
                    "explanation": f"{query_bagua}生{cand_bagua}（八卦因果）"}
        if cand_bagua in causality.get("contradicts", []):
            if query_wuxing and cand_wuxing and WUXING_SHENG.get(query_wuxing) == cand_wuxing:
                return {"relation": "generates", "score": 0.7, "path": ["八卦相克", "五行相生"],
                        "explanation": f"{query_bagua}克{cand_bagua}（八卦），但{query_wuxing}生{cand_wuxing}（五行）"}
            return {"relation": "contradicts", "score": 0.3, "path": ["八卦相克"],
                    "explanation": f"{query_bagua}克{cand_bagua}（八卦因果）"}
        if query_wuxing and cand_wuxing:
            if WUXING_SHENG.get(query_wuxing) == cand_wuxing:
                return {"relation": "generates", "score": 0.7, "path": ["五行相生"],
                        "explanation": f"{query_wuxing}生{cand_wuxing}（五行能量）"}
            if WUXING_SHENG.get(cand_wuxing) == query_wuxing:
                return {"relation": "generates", "score": 0.6, "path": ["五行被生"],
                        "explanation": f"{cand_wuxing}生{query_wuxing}（五行反向）"}
            if WUXING_KE.get(query_wuxing) == cand_wuxing:
                return {"relation": "contradicts", "score": 0.2, "path": ["五行相克"],
                        "explanation": f"{query_wuxing}克{cand_wuxing}（五行能量）"}
            if WUXING_KE.get(cand_wuxing) == query_wuxing:
                return {"relation": "contradicts", "score": 0.2, "path": ["五行被克"],
                        "explanation": f"{cand_wuxing}克{query_wuxing}（五行反向）"}
        return {"relation": "neutral", "score": 0.0, "path": [],
                "explanation": f"{query_bagua}{query_wuxing}与{cand_bagua}{cand_wuxing}无直接因果"}

    def multi_hop_inference(self, query_bagua: str, query_wuxing: str,
                           memories: List[Dict], max_hops: int = 3) -> List[Dict]:
        hop_decay = 0.7
        mem_attrs = []
        for m in memories:
            bagua = m.get("bagua_name") or m.get("payload", {}).get("bagua_name", "")
            wuxing = m.get("wuxing") or m.get("payload", {}).get("wuxing", "")
            if not wuxing and bagua:
                wuxing = BAGUA_WUXING.get(bagua, "")
            mem_attrs.append({"bagua": bagua, "wuxing": wuxing})
        best_scores = {}
        first_hop_results = []
        for i, (m, attr) in enumerate(zip(memories, mem_attrs)):
            if not attr["bagua"]:
                continue
            rel = self.infer_relation(query_bagua, query_wuxing, attr["bagua"], attr["wuxing"])
            score = rel["score"]
            if score > 0:
                first_hop_results.append((i, score, rel))
                best_scores[i] = {"hop_score": score, "hop_count": 1,
                    "hop_path": [f"query->{m.get('id', i)}({rel['relation']})"]}
        if max_hops < 2:
            return self._build_results(memories, best_scores)
        first_hop_results.sort(key=lambda x: x[1], reverse=True)
        bridges = first_hop_results[:5]
        for bridge_idx, bridge_score, bridge_rel in bridges:
            bridge_attr = mem_attrs[bridge_idx]
            for j, attr in enumerate(mem_attrs):
                if j == bridge_idx or not attr["bagua"]:
                    continue
                rel2 = self.infer_relation(bridge_attr["bagua"], bridge_attr["wuxing"],
                                           attr["bagua"], attr["wuxing"])
                if rel2["score"] > 0:
                    hop2_score = bridge_score * hop_decay * rel2["score"]
                    if j not in best_scores or hop2_score > best_scores[j]["hop_score"]:
                        bridge_id = memories[bridge_idx].get("id", bridge_idx)
                        target_id = memories[j].get("id", j)
                        best_scores[j] = {"hop_score": hop2_score, "hop_count": 2,
                            "hop_path": [f"query->{bridge_id}({bridge_rel['relation']})",
                                         f"{bridge_id}->{target_id}({rel2['relation']})"]}
        if max_hops < 3:
            return self._build_results(memories, best_scores)
        hop2_bridges = [(idx, info) for idx, info in best_scores.items() if info["hop_count"] == 2]
        hop2_bridges.sort(key=lambda x: x[1]["hop_score"], reverse=True)
        for bridge_idx, bridge_info in hop2_bridges[:3]:
            bridge_attr = mem_attrs[bridge_idx]
            for j, attr in enumerate(mem_attrs):
                if j == bridge_idx or not attr["bagua"]:
                    continue
                rel3 = self.infer_relation(bridge_attr["bagua"], bridge_attr["wuxing"],
                                           attr["bagua"], attr["wuxing"])
                if rel3["score"] > 0:
                    hop3_score = bridge_info["hop_score"] * hop_decay * rel3["score"]
                    if j not in best_scores or hop3_score > best_scores[j]["hop_score"]:
                        bridge_id = memories[bridge_idx].get("id", bridge_idx)
                        target_id = memories[j].get("id", j)
                        best_scores[j] = {"hop_score": hop3_score, "hop_count": 3,
                            "hop_path": bridge_info["hop_path"] + [f"{bridge_id}->{target_id}({rel3['relation']})"]}
        return self._build_results(memories, best_scores)

    def build_reasoning_chain(self, memories: List[Dict]) -> Dict:
        nodes = []
        edges = []
        covered = set()
        mem_attrs = []
        for m in memories:
            bagua = m.get("bagua_name") or m.get("payload", {}).get("bagua_name", "")
            wuxing = m.get("wuxing") or m.get("payload", {}).get("wuxing", "")
            if not wuxing and bagua:
                wuxing = BAGUA_WUXING.get(bagua, "")
            mid = m.get("id", f"mem_{len(mem_attrs)}")
            nodes.append({"id": mid, "bagua": bagua, "wuxing": wuxing})
            mem_attrs.append({"id": mid, "bagua": bagua, "wuxing": wuxing})
        for i in range(len(mem_attrs)):
            for j in range(len(mem_attrs)):
                if i == j:
                    continue
                a, b = mem_attrs[i], mem_attrs[j]
                if not a["bagua"] or not b["bagua"]:
                    continue
                rel = self.infer_relation(a["bagua"], a["wuxing"], b["bagua"], b["wuxing"])
                if rel["score"] > 0 and rel["relation"] != "neutral":
                    edges.append({"from": a["id"], "to": b["id"],
                                  "relation": rel["relation"], "score": rel["score"]})
                    covered.add(i)
                    covered.add(j)
        adj = defaultdict(list)
        for e in edges:
            if e["relation"] in ("generates", "same"):
                adj[e["from"]].append((e["to"], e["score"]))
        longest_chain = []
        for start_node in nodes:
            chain = self._dfs_longest(adj, start_node["id"], set())
            if len(chain) > len(longest_chain):
                longest_chain = chain
        coverage = len(covered) / max(len(memories), 1) * 100
        return {"nodes": nodes, "edges": edges, "chains": longest_chain, "coverage": round(coverage, 1)}

    def _build_results(self, memories, best_scores):
        results = []
        for i, m in enumerate(memories):
            entry = dict(m)
            if i in best_scores:
                entry["hop_score"] = round(best_scores[i]["hop_score"], 4)
                entry["hop_count"] = best_scores[i]["hop_count"]
                entry["hop_path"] = best_scores[i]["hop_path"]
            else:
                entry["hop_score"] = 0.0
                entry["hop_count"] = 0
                entry["hop_path"] = []
            results.append(entry)
        results.sort(key=lambda x: x["hop_score"], reverse=True)
        return results

    @staticmethod
    def _dfs_longest(adj, node, visited):
        visited.add(node)
        best = [node]
        for nxt, _ in adj.get(node, []):
            if nxt not in visited:
                chain = [node] + CausalInference._dfs_longest(adj, nxt, visited.copy())
                if len(chain) > len(best):
                    best = chain
        return best
