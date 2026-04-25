"""
全息检索接口 — 5维连续融合得分系统

对外暴露：MultiViewRetriever
内部实现：封装在su_core._sys中

5维融合：
1. semantic（0.40）：语义向量相似度
2. bagua_soft（0.15）：八卦软匹配（概率分布相似度）
3. wuxing_energy（0.15）：五行能量匹配（含生克关系）
4. holographic（0.15）：全息结构匹配（本/互/综/错卦）
5. causal（0.15）：因果链关联度
"""

from typing import List, Dict, Any, Tuple
from .encoders import EncoderCore, EncodingInfo, _cosine_similarity_dict
from ._c2 import Wuxing, WUXING_SHENG, WUXING_KE, wuxing_from_bagua
from .causal import BAGUA_CAUSALITY


class MultiViewRetriever:
    """
    全息检索器 - 5维连续融合

    提供五维融合检索能力：
    1. 语义向量相似度（核心维度）
    2. 八卦软匹配（概率分布相似度）
    3. 五行能量匹配（含生克关系）
    4. 全息结构匹配（本/互/综/错卦）
    5. 因果链关联度

    对外隐藏：具体算法、权重配置
    """

    def __init__(self):
        self.hexagram_system = EncoderCore()
        # 5维融合权重配置
        self._weights = {
            "semantic": 0.40,
            "bagua_soft": 0.15,
            "wuxing_energy": 0.15,
            "holographic": 0.15,
            "causal": 0.15,
        }

    def retrieve(
        self,
        query_content: str,
        query_hexagram: EncodingInfo,
        candidates: List[Dict[str, Any]],
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        全息检索主入口

        Args:
            query_content: 查询文本
            query_hexagram: 查询对应的卦象（EncodingInfo，带 bagua_probs/wuxing_scores）
            candidates: 候选记忆列表
            top_k: 返回数量

        Returns:
            排序后的记忆列表，附带全息得分和融合详情
        """
        if not candidates:
            return []

        # 提取查询的八卦名（用于因果维度）
        query_bagua_name = self._get_bagua_name(query_hexagram)

        for i, cand in enumerate(candidates):
            hex_idx = cand.get("hexagram_index", 0)

            # ---- 维度1: semantic（0.40）----
            semantic_score = cand.get("vector_score", 0.5)

            # ---- 维度2: bagua_soft（0.15）----
            bagua_soft_score = self._compute_bagua_soft(query_hexagram, cand, hex_idx)

            # ---- 维度3: wuxing_energy（0.15）----
            wuxing_score = self._compute_wuxing_energy(query_hexagram, cand, hex_idx)

            # ---- 维度4: holographic（0.15）----
            holo_score = self._compute_holographic(query_hexagram, cand, hex_idx)

            # ---- 维度5: causal（0.15）----
            causal_score = self._compute_causal(query_bagua_name, cand)

            # 综合得分 = 5维加权和
            total_score = (
                self._weights["semantic"] * semantic_score +
                self._weights["bagua_soft"] * bagua_soft_score +
                self._weights["wuxing_energy"] * wuxing_score +
                self._weights["holographic"] * holo_score +
                self._weights["causal"] * causal_score
            )

            cand["holographic_score"] = round(total_score, 4)
            cand["fusion_detail"] = {
                "semantic": round(semantic_score, 4),
                "bagua_soft": round(bagua_soft_score, 4),
                "wuxing_energy": round(wuxing_score, 4),
                "holographic": round(holo_score, 4),
                "causal": round(causal_score, 4),
            }

        # 按综合得分排序
        candidates.sort(key=lambda x: x.get("holographic_score", 0), reverse=True)

        return candidates[:top_k]

    def _get_bagua_name(self, info: EncodingInfo) -> str:
        """从 EncodingInfo 提取主八卦名"""
        if info.bagua_probs:
            return max(info.bagua_probs, key=info.bagua_probs.get)
        # fallback: 从卦名推断上卦
        from .encoders import BAGUA_NAMES, HEXAGRAM_TRIGRAMS_BELOW
        below_idx = HEXAGRAM_TRIGRAMS_BELOW[info.index]
        if 0 <= below_idx < len(BAGUA_NAMES):
            return BAGUA_NAMES[below_idx]
        return "坤"

    def _compute_bagua_soft(self, query_info: EncodingInfo, cand: Dict, hex_idx: int) -> float:
        """计算八卦软匹配得分"""
        q_probs = query_info.bagua_probs
        c_probs = cand.get("bagua_probs")

        if q_probs and c_probs:
            return max(0.0, _cosine_similarity_dict(q_probs, c_probs))

        # fallback: 0/1 本卦匹配
        return 1.0 if hex_idx == query_info.index else 0.0

    def _compute_wuxing_energy(self, query_info: EncodingInfo, cand: Dict, hex_idx: int) -> float:
        """计算五行能量匹配得分"""
        q_scores = query_info.wuxing_scores
        c_scores = cand.get("wuxing_scores")

        base_score = 0.0
        if q_scores and c_scores:
            base_score = max(0.0, _cosine_similarity_dict(q_scores, c_scores))
        else:
            # fallback: 同五行 → 1.0
            cand_info = EncodingInfo.from_index(hex_idx)
            base_score = 1.0 if cand_info.wuxing == query_info.wuxing else 0.0

        # 生克加成
        q_wuxing = self._dominant_wuxing(query_info)
        c_wuxing = self._dominant_wuxing_from_cand(cand, hex_idx)

        if q_wuxing and c_wuxing:
            if WUXING_SHENG.get(q_wuxing) == c_wuxing or WUXING_SHENG.get(c_wuxing) == q_wuxing:
                base_score = min(1.0, base_score + 0.2)
            elif WUXING_KE.get(q_wuxing) == c_wuxing or WUXING_KE.get(c_wuxing) == q_wuxing:
                base_score = max(0.0, base_score - 0.1)

        return base_score

    def _dominant_wuxing(self, info: EncodingInfo):
        """从 EncodingInfo 获取主五行"""
        if info.wuxing_scores:
            dominant_name = max(info.wuxing_scores, key=info.wuxing_scores.get)
            return self._name_to_wuxing(dominant_name)
        return self._name_to_wuxing(info.wuxing)

    def _dominant_wuxing_from_cand(self, cand: Dict, hex_idx: int):
        """从候选获取主五行"""
        c_scores = cand.get("wuxing_scores")
        if c_scores:
            dominant_name = max(c_scores, key=c_scores.get)
            return self._name_to_wuxing(dominant_name)
        payload = cand.get("payload", {})
        wx_name = payload.get("wuxing") or EncodingInfo.from_index(hex_idx).wuxing
        return self._name_to_wuxing(wx_name)

    @staticmethod
    def _name_to_wuxing(name: str):
        """五行名称 → Wuxing 枚举"""
        mapping = {"金": Wuxing.JIN, "木": Wuxing.MU, "水": Wuxing.SHUI, "火": Wuxing.HUO, "土": Wuxing.TU}
        return mapping.get(name)

    def _compute_holographic(self, query_info: EncodingInfo, cand: Dict, hex_idx: int) -> float:
        """计算全息结构匹配得分"""
        # 构建候选 EncodingInfo（如果有 bagua_probs）
        candidate_infos = {}
        c_probs = cand.get("bagua_probs")
        c_wscores = cand.get("wuxing_scores")
        if c_probs or c_wscores:
            ci = EncodingInfo.from_index(hex_idx)
            ci.bagua_probs = c_probs
            ci.wuxing_scores = c_wscores
            candidate_infos[hex_idx] = ci

        scores = self.hexagram_system.retrieve_holographic(
            query_index=query_info.index,
            candidate_indices=[hex_idx],
            top_k=1,
            query_info=query_info,
            candidate_infos=candidate_infos if candidate_infos else None,
        )

        if scores:
            return scores[0][1]
        return 0.0

    def _compute_causal(self, query_bagua_name: str, cand: Dict) -> float:
        """计算因果链关联度"""
        payload = cand.get("payload", {})
        cand_hexagram_name = payload.get("hexagram_name", "")

        # 从卦名推断八卦（取下卦）
        cand_bagua_name = self._hexagram_to_bagua(cand_hexagram_name, cand)

        if not query_bagua_name or not cand_bagua_name:
            return 0.0

        # 同卦 → 1.0
        if query_bagua_name == cand_bagua_name:
            return 1.0

        # 查 BAGUA_CAUSALITY
        causality = BAGUA_CAUSALITY.get(query_bagua_name, {})
        if cand_bagua_name in causality.get("generates", []):
            return 0.8
        if cand_bagua_name in causality.get("contradicts", []):
            return 0.3

        return 0.0

    def _hexagram_to_bagua(self, hexagram_name: str, cand: Dict) -> str:
        """从64卦名推断所属八卦（下卦）"""
        # 先尝试从候选的 bagua_probs 取 argmax
        c_probs = cand.get("bagua_probs")
        if c_probs:
            return max(c_probs, key=c_probs.get)

        # 尝试从 payload 获取 wuxing 再映射到八卦
        payload = cand.get("payload", {})
        wuxing_name = payload.get("wuxing", "")
        wuxing_to_default_bagua = {
            "金": "乾", "木": "震", "水": "坎", "火": "离", "土": "坤",
        }
        if wuxing_name in wuxing_to_default_bagua:
            return wuxing_to_default_bagua[wuxing_name]

        # fallback: 从 hexagram_index
        hex_idx = cand.get("hexagram_index", 0)
        from .encoders import BAGUA_NAMES, HEXAGRAM_TRIGRAMS_BELOW
        below_idx = HEXAGRAM_TRIGRAMS_BELOW[hex_idx]
        if 0 <= below_idx < len(BAGUA_NAMES):
            return BAGUA_NAMES[below_idx]
        return "坤"
