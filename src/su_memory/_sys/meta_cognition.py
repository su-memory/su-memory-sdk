"""
元认知模块：76.2% → 80%+
"""
import time
from typing import Dict, List

class MetaCognition:
    def discover_gaps(self, types: Dict, domains: List, memories: List) -> List[Dict]:
        gaps = []
        total = sum(types.values())
        # 1. 领域覆盖空洞
        if types.get("fact", 0) / max(total, 1) < 0.3:
            gaps.append({"type": "domain", "severity": 0.7})
        # 2. 时间空洞
        if memories:
            oldest = min(m.get("timestamp", time.time()) for m in memories)
            if time.time() - oldest > 86400 * 30:
                gaps.append({"type": "temporal", "severity": 0.6})
        # 3. 因果空洞
        isolated = sum(1 for m in memories if not m.get("causal") and not m.get("causal_children"))
        if isolated / max(len(memories), 1) > 0.5:
            gaps.append({"type": "causal", "severity": 0.5})
        return gaps

    def detect_conflicts(self, beliefs: Dict) -> List[Dict]:
        conflicts = []
        ids = list(beliefs.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_content = beliefs[ids[i]].get("content", "")
                b_content = beliefs[ids[j]].get("content", "")
                if self._contradicts(a_content, b_content):
                    conflicts.append({"memory_a": ids[i], "memory_b": ids[j], "severity": 0.8})
        return sorted(conflicts, key=lambda x: -x["severity"])

    def _contradicts(self, text_a: str, text_b: str) -> bool:
        pos = ["是", "有", "正确", "知道"]
        neg = ["不是", "没有", "错误", "不知道"]
        a_pos = sum(1 for p in pos if p in text_a)
        b_pos = sum(1 for p in pos if p in text_b)
        a_neg = sum(1 for n in neg if n in text_a)
        b_neg = sum(1 for n in neg if n in text_b)
        return (a_pos and b_neg) or (a_neg and b_pos) > 0

    def get_aging(self, memories: List[Dict]) -> List[Dict]:
        aging = []
        now = time.time()
        for m in memories:
            days = (now - m.get("timestamp", now)) / 86400
            if 14 < days < 30:
                aging.append({"id": m["id"], "days": round(days), "severity": "warning"})
            elif days >= 30:
                aging.append({"id": m["id"], "days": round(days), "severity": "critical"})
        return aging
