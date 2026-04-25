"""
ForgettingEngine - 遗忘引擎

Phase 0: 基于访问频率+时间的简单遗忘
Phase 1: 基于卦气状态（旺相休囚死）的主动遗忘
"""

import time
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ForgettingEngine:
    """
    遗忘引擎 - 控制记忆的归档和删除
    
    Phase 0策略：
    1. 低优先级 + 久未访问 → 归档
    2. 超过30天未访问 → 自动归档
    3. 超过90天未访问 → 删除（可选）
    
    Phase 1策略：
    1. 卦气状态为"死" → 主动归档
    2. 卦气状态为"囚"超过7天 → 归档
    3. 能量强度低于阈值 → 归档
    """
    
    # 遗忘阈值配置
    ARCHIVE_AFTER_DAYS = 30      # 30天未访问 → 归档
    DELETE_AFTER_DAYS = 90       # 90天未访问 → 删除（可选）
    MIN_PRIORITY_TO_KEEP = 3     # 优先级低于此 → 考虑遗忘
    
    def __init__(self):
        self.archive_count = 0
        self.delete_count = 0
    
    def should_archive(self, memory: Dict[str, Any]) -> bool:
        """
        判断记忆是否应该归档
        
        归档条件（满足任一）：
        1. 30天以上未访问 + 低优先级
        2. 优先级极低（<3）+ 60天以上未访问
        """
        last_access = memory.get("last_access", memory.get("timestamp", 0))
        priority = memory.get("priority", 5)
        status = memory.get("status", "active")
        
        if status != "active":
            return False
        
        now = time.time()
        days_since_access = (now - last_access) / (24 * 3600)
        
        # 条件1: 30天以上未访问 + 低优先级
        if days_since_access > 30 and priority < self.MIN_PRIORITY_TO_KEEP:
            return True
        
        # 条件2: 60天以上未访问 + 极低优先级
        if days_since_access > 60 and priority < 2:
            return True
        
        return False
    
    def should_delete(self, memory: Dict[str, Any]) -> bool:
        """
        判断记忆是否应该删除
        
        删除条件：
        1. 90天以上未访问 + 已归档
        2. 用户显式删除
        """
        last_access = memory.get("last_access", memory.get("timestamp", 0))
        status = memory.get("status", "active")
        
        now = time.time()
        days_since_access = (now - last_access) / (24 * 3600)
        
        # 90天以上未访问 + 已归档
        if days_since_access > self.DELETE_AFTER_DAYS and status == "archived":
            return True
        
        return False
    
    async def process_forgetting(
        self,
        tenant_id: str,
        user_id: str,
        all_memories: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        处理遗忘流程
        
        Returns:
            {
                "archived": [memory_ids],
                "deleted": [memory_ids],
                "kept": [memory_ids]
            }
        """
        archived = []
        deleted = []
        kept = []
        
        for memory in all_memories:
            if memory.get("status") == "archived":
                # 已归档的检查是否删除
                if self.should_delete(memory):
                    deleted.append(memory["id"])
                    self.delete_count += 1
                else:
                    kept.append(memory["id"])
            elif memory.get("status") == "active":
                # 活跃的检查是否归档
                if self.should_archive(memory):
                    archived.append(memory["id"])
                    self.archive_count += 1
                    logger.info(f"Memory {memory['id']} marked for archival")
                else:
                    kept.append(memory["id"])
        
        return {
            "archived": archived,
            "deleted": deleted,
            "kept": kept,
            "archive_count": self.archive_count,
            "delete_count": self.delete_count
        }
    
    def get_forgetting_candidates(
        self,
        memories: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        获取最应该被遗忘的记忆候选
        
        Returns:
            按遗忘优先级排序的记忆列表（最需要遗忘的在前）
        """
        candidates = []
        
        for memory in memories:
            if memory.get("status") != "active":
                continue
            
            last_access = memory.get("last_access", memory.get("timestamp", 0))
            priority = memory.get("priority", 5)
            
            now = time.time()
            days_since_access = (now - last_access) / (24 * 3600)
            
            # 遗忘优先级得分（越高越应该遗忘）
            forgetting_score = (
                days_since_access / 30 * 0.4 +  # 时间因素
                (10 - priority) / 10 * 0.6       # 优先级因素（低优先级优先遗忘）
            )
            
            candidates.append({
                "memory_id": memory["id"],
                "forgetting_score": round(forgetting_score, 3),
                "days_since_access": round(days_since_access, 1),
                "priority": priority,
                "content_preview": memory["content"][:50] + "..."
            })
        
        # 按遗忘得分排序
        candidates.sort(key=lambda x: x["forgetting_score"], reverse=True)
        
        return candidates[:top_k]
