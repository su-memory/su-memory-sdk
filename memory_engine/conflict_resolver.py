"""
ConflictResolver - 冲突消解引擎

Phase 0: 简单冲突检测（基于关键词）
Phase 1: 五行相克增强（基于周易框架）
"""

import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ConflictResolver:
    """
    记忆冲突检测与消解
    
    Phase 0: 基于规则/关键词的冲突检测
    Phase 1: 基于五行生克的因果冲突检测
    """
    
    def __init__(self):
        # 冲突关键词对（一方出现则可能与另一方冲突）
        self.conflict_pairs = [
            (["有", "没有", "不存在"], ["无", "不是", "否认"]),
            (["喜欢", "爱"], ["讨厌", "恨", "不喜欢"]),
            (["是", "确实"], ["不是", "否认", "否定"]),
            (["正确", "对"], ["错误", "错"]),
            (["可以", "允许"], ["不能", "禁止", "不可以"]),
            (["知道", "了解"], ["不知道", "不清楚"]),
        ]
        
        # 冲突关系定义（Phase 1会扩展为五行相克）
        self.conflict_relations = [
            ("喜欢", "讨厌"),
            ("有", "没有"),
            ("正确", "错误"),
            ("知道", "不知道"),
            ("可以", "不能"),
        ]
    
    def detect_conflicts(self, new_content: str, existing_memories: List[Dict]) -> List[Dict]:
        """
        检测新记忆与现有记忆的冲突
        
        Returns:
            冲突列表，每个冲突包含 (new_memory_id, existing_memory_id, conflict_type)
        """
        conflicts = []
        
        for existing in existing_memories:
            conflict = self._check_pair_conflict(new_content, existing["content"])
            if conflict:
                conflicts.append({
                    "new_id": new_content,
                    "existing_id": existing["id"],
                    "conflict_type": conflict,
                    "existing_content": existing["content"]
                })
        
        return conflicts
    
    def _check_pair_conflict(self, content1: str, content2: str) -> str:
        """检查两条记忆是否冲突"""
        content1_lower = content1.lower()
        content2_lower = content2.lower()
        
        for pos_words, neg_words in self.conflict_pairs:
            pos1 = any(w in content1_lower for w in pos_words)
            pos2 = any(w in content2_lower for w in pos_words)
            neg1 = any(w in content1_lower for w in neg_words)
            neg2 = any(w in content2_lower for w in neg_words)
            
            # 一正一负则冲突
            if (pos1 and neg2) or (pos2 and neg1):
                return "contradiction"
            
            # 同一关键词的肯定和否定冲突
            for word_set in [pos_words + neg_words]:
                has_pos1 = any(w in content1_lower for w in word_set if w in pos_words)
                has_neg2 = any(w in content2_lower for w in word_set if w in neg_words)
                has_pos2 = any(w in content2_lower for w in word_set if w in pos_words)
                has_neg1 = any(w in content1_lower for w in word_set if w in neg_words)
                
                if (has_pos1 and has_neg2) or (has_pos2 and has_neg1):
                    return "contradiction"
        
        return None
    
    def resolve(self, conflicts: List[Dict]) -> List[str]:
        """
        解决冲突，决定保留哪些记忆
        
        Returns:
            需要标记为无效的记忆ID列表
        """
        invalid_ids = []
        
        for conflict in conflicts:
            # Phase 0策略：默认保留新的，标记旧的为冲突
            # Phase 1会考虑：置信度、五行相克、卦气状态
            invalid_ids.append(conflict["existing_id"])
        
        return invalid_ids
    
    async def auto_resolve(self, tenant_id: str, user_id: str, memory_id: str, content: str) -> Dict[str, Any]:
        """
        自动冲突检测与消解
        
        Returns:
            {"has_conflicts": bool, "resolved_memories": [...], "action_taken": str}
        """
        # 获取用户所有记忆
        from memory_engine.manager import MemoryManager
        manager = MemoryManager()
        
        all_memories = await manager.query_memory(tenant_id, user_id, "", limit=100)
        
        # 检测冲突
        conflicts = self.detect_conflicts(content, all_memories)
        
        if not conflicts:
            return {"has_conflicts": False, "resolved_memories": [], "action_taken": "none"}
        
        # 解决冲突
        invalid_ids = self.resolve(conflicts)
        
        return {
            "has_conflicts": True,
            "conflict_count": len(conflicts),
            "resolved_memories": invalid_ids,
            "action_taken": "marked_conflict"
        }
