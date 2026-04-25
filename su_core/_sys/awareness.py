"""
元认知系统

Phase 1: 主动发现认知空洞、信念冲突、知识老化

对外暴露：MetaCognition
内部实现：封装在su_core._internal中
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass
class CognitiveGap:
    """认知空洞"""
    gap_id: str
    gap_type: str              # "domain" | "temporal" | "causal"
    description: str           # 空洞描述
    severity: float            # 严重程度 0-1
    suggestions: List[str]     # 建议行动
    discovered_at: float       # 发现时间


@dataclass
class KnowledgeAging:
    """知识老化"""
    memory_id: str
    days_since_update: int
    current_stage: str         # 信念阶段
    severity: str               # "normal" | "warning" | "critical"
    suggestion: str


class MetaCognition:
    """
    元认知系统 - 对外唯一接口
    
    功能：
    1. 发现认知空洞（用户知识的盲区）
    2. 检测信念冲突
    3. 预警知识老化
    4. 提供主动建议
    
    对外隐藏：发现算法、阈值配置
    """
    
    # 认知空洞检测阈值
    DOMAIN_COVERAGE_THRESHOLD = 0.7   # 领域覆盖率低于70% → 空洞
    TEMPORAL_GAP_DAYS = 90            # 某类记忆超过90天未更新 → 时间空洞
    CONFLICT_SEVERITY_THRESHOLD = 0.8  # 冲突置信度都>0.8 → 严重冲突
    
    # 知识老化阈值
    AGING_WARNING_DAYS = 30
    AGING_CRITICAL_DAYS = 60
    
    def __init__(self):
        self._gaps: List[CognitiveGap] = []
        self._last_scan = 0
        self._scan_interval = 3600  # 每小时最多扫描一次
    
    def discover_gaps(
        self,
        memory_types: Dict[str, int],
        user_domains: List[str],
        memory_list: List[Dict]
    ) -> List[CognitiveGap]:
        """
        发现认知空洞
        
        Args:
            memory_types: 各类型记忆的数量 {"fact": 10, "preference": 5, ...}
            user_domains: 用户关注的领域列表
            memory_list: 记忆列表 [{"id", "type", "timestamp", "stage", ...}]
        
        Returns:
            发现的空洞列表
        """
        gaps = []
        now = time.time()
        
        # 1. 领域覆盖空洞
        domain_gaps = self._detect_domain_gaps(memory_types, user_domains)
        gaps.extend(domain_gaps)
        
        # 2. 时序空洞
        temporal_gaps = self._detect_temporal_gaps(memory_list)
        gaps.extend(temporal_gaps)
        
        # 3. 因果空洞
        causal_gaps = self._detect_causal_gaps(memory_list)
        gaps.extend(causal_gaps)
        
        # 更新存储
        self._gaps = gaps
        self._last_scan = now
        
        return gaps
    
    def _detect_domain_gaps(
        self,
        memory_types: Dict[str, int],
        user_domains: List[str]
    ) -> List[CognitiveGap]:
        """检测领域覆盖空洞"""
        gaps = []
        
        total_memories = sum(memory_types.values())
        if total_memories == 0:
            return gaps
        
        # 检查各类记忆的分布
        type_ratios = {k: v / total_memories for k, v in memory_types.items()}
        
        # 事实类记忆过少
        if type_ratios.get("fact", 0) < 0.3:
            gaps.append(CognitiveGap(
                gap_id=f"domain_fact_{int(time.time())}",
                gap_type="domain",
                description="事实类记忆偏少，可能影响判断准确性",
                severity=0.7,
                suggestions=["补充更多基础事实信息", "建立知识库"],
                discovered_at=time.time()
            ))
        
        # 偏好类记忆过少
        if type_ratios.get("preference", 0) < 0.1:
            gaps.append(CognitiveGap(
                gap_id=f"domain_pref_{int(time.time())}",
                gap_type="domain",
                description="用户偏好信息不足，可能影响个性化服务",
                severity=0.6,
                suggestions=["收集用户偏好", "记录用户选择"],
                discovered_at=time.time()
            ))
        
        # 事件类记忆过多（可能有噪音）
        if type_ratios.get("event", 0) > 0.5:
            gaps.append(CognitiveGap(
                gap_id=f"domain_event_{int(time.time())}",
                gap_type="domain",
                description="事件记忆占比过高，可能需要整理",
                severity=0.5,
                suggestions=["对事件进行归纳总结", "提取关键规律"],
                discovered_at=time.time()
            ))
        
        return gaps
    
    def _detect_temporal_gaps(self, memory_list: List[Dict]) -> List[CognitiveGap]:
        """检测时序空洞"""
        gaps = []
        now = time.time()
        
        # 按类型分组，检查最后更新时间
        type_last_update: Dict[str, float] = defaultdict(lambda: 0)
        type_counts: Dict[str, int] = defaultdict(int)
        
        for mem in memory_list:
            mem_type = mem.get("type", "fact")
            timestamp = mem.get("timestamp", 0)
            
            type_counts[mem_type] += 1
            if timestamp > type_last_update[mem_type]:
                type_last_update[mem_type] = timestamp
        
        # 检查各类型的时序空洞
        for mem_type, last_update in type_last_update.items():
            days_elapsed = (now - last_update) / (24 * 3600)
            
            if days_elapsed > self.AGING_CRITICAL_DAYS and type_counts[mem_type] > 3:
                gaps.append(CognitiveGap(
                    gap_id=f"temporal_{mem_type}_{int(time.time())}",
                    gap_type="temporal",
                    description=f"{mem_type}类记忆已{days_elapsed:.0f}天未更新，可能已过时",
                    severity=0.8,
                    suggestions=[f"更新{ mem_type}类信息", "检查最新动态"],
                    discovered_at=now
                ))
        
        return gaps
    
    def _detect_causal_gaps(self, memory_list: List[Dict]) -> List[CognitiveGap]:
        """检测因果空洞（孤立的记忆节点）"""
        gaps = []
        
        # 找出没有任何关联的记忆（孤立节点）
        isolated_count = 0
        for mem in memory_list:
            # 没有causal_parents也没有causal_children
            if not mem.get("causal_parents") and not mem.get("causal_children"):
                isolated_count += 1
        
        total = len(memory_list)
        if total > 10 and isolated_count / total > 0.8:
            gaps.append(CognitiveGap(
                gap_id=f"causal_isolated_{int(time.time())}",
                gap_type="causal",
                description="大量记忆缺乏关联，建议建立记忆间的因果联系",
                severity=0.6,
                suggestions=["主动建立记忆关联", "分析记忆间的因果关系"],
                discovered_at=time.time()
            ))
        
        return gaps
    
    def detect_conflicts(
        self,
        beliefs: Dict[str, Dict]
    ) -> List[Dict[str, Any]]:
        """
        检测严重信念冲突
        
        Returns:
            [{"memory_a": id, "memory_b": id, "severity": 0.9, "description": "..."}]
        """
        conflicts = []
        
        memory_ids = list(beliefs.keys())
        
        for i, id_a in enumerate(memory_ids):
            for id_b in memory_ids[i+1:]:
                state_a = beliefs[id_a]
                state_b = beliefs[id_b]
                
                # 两者置信度都高但阶段对立
                if (state_a["confidence"] > 0.7 and 
                    state_b["confidence"] > 0.7 and
                    state_a["stage"] in ["强化", "确认"] and
                    state_b["stage"] in ["强化", "确认"]):
                    
                    # 检查内容是否矛盾（简化判断）
                    content_a = state_a.get("content", "")[:100].lower()
                    content_b = state_b.get("content", "")[:100].lower()
                    
                    if self._is_contradictory(content_a, content_b):
                        conflicts.append({
                            "memory_a": id_a,
                            "memory_b": id_b,
                            "severity": (state_a["confidence"] + state_b["confidence"]) / 2,
                            "description": f"两个高置信度信念存在内容矛盾",
                            "stage_a": state_a["stage"],
                            "stage_b": state_b["stage"]
                        })
        
        # 按严重程度排序
        conflicts.sort(key=lambda x: x["severity"], reverse=True)
        return conflicts
    
    def _is_contradictory(self, content_a: str, content_b: str) -> bool:
        """简单判断内容是否矛盾"""
        # 肯定的词
        positive = ["是", "有", "正确", "可以", "知道", "喜欢", "能"]
        # 否定的词
        negative = ["不是", "没有", "错误", "不能", "不知道", "讨厌", "否"]
        
        pos_a = any(w in content_a for w in positive)
        neg_a = any(w in content_a for w in negative)
        pos_b = any(w in content_b for w in positive)
        neg_b = any(w in content_b for w in negative)
        
        # 一正一负且有共同关键词
        if (pos_a and neg_b) or (pos_b and neg_a):
            return True
        
        return False
    
    def get_aging_warnings(self, memory_list: List[Dict]) -> List[KnowledgeAging]:
        """获取知识老化预警"""
        warnings = []
        now = time.time()
        
        for mem in memory_list:
            days_elapsed = (now - mem.get("timestamp", now)) / (24 * 3600)
            
            if days_elapsed > self.AGING_CRITICAL_DAYS:
                severity = "critical"
            elif days_elapsed > self.AGING_WARNING_DAYS:
                severity = "warning"
            else:
                continue
            
            warnings.append(KnowledgeAging(
                memory_id=mem["id"],
                days_since_update=int(days_elapsed),
                current_stage=mem.get("stage", "未知"),
                severity=severity,
                suggestion="建议更新或验证此记忆的准确性"
            ))
        
        return warnings
    
    def get_suggestions(self) -> List[str]:
        """获取主动建议（基于当前空洞）"""
        suggestions = []
        
        for gap in self._gaps:
            suggestions.extend(gap.suggestions)
        
        # 去重，保持顺序
        seen = set()
        unique = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        
        return unique[:5]  # 最多返回5条
