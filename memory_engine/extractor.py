"""
MemoryExtractor - 记忆信息抽取（Phase 2增强版）

Phase 0: 基础抽取（实体识别、类型分类、简单压缩）
Phase 1: su_core核心能力集成
Phase 2: 完整融合
"""

import re
import hashlib
import numpy as np
import logging
from typing import Dict, Any, Optional

# sentence-transformers导入（带fallback）
ST_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None

from su_core import (
    SemanticEncoder, SuCompressor,
    TemporalSystem, BeliefTracker, MetaCognition
)

logger = logging.getLogger(__name__)

# 向量模型（全局单例）
_embedding_model = None


def get_embedding_model():
    """加载sentence-transformers模型（带fallback"""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    if ST_AVAILABLE:
        try:
            model_name = "all-MiniLM-L6-v2"
            _embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
            return _embedding_model
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer: {e}")
    return None


class MemoryExtractor:
    """
    记忆信息抽取器（Phase 2增强版）

    功能：
    1. 实体识别（NER）
    2. 类型分类（fact/preference/event/belief）
    3. su_core语义压缩（高压缩率）
    4. 向量化（SentenceTransformer + fallback hash）
    5. 64卦编码（SemanticEncoder）
    6. 信念状态初始化（BeliefTracker）
    """

    def __init__(self):
        self.model = None  # 延迟加载
        self.semantic_encoder = SemanticEncoder()
        self.compressor = SuCompressor()
        self.temporal_system = TemporalSystem()
        self.belief_tracker = BeliefTracker()
        self.meta_cognition = MetaCognition()

    async def extract(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """异步版本的extract"""
        return self.extract_sync(content, metadata)

    def extract_sync(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """同步版本的extract"""
        metadata = metadata or {}

        entities = self._extract_entities(content)
        memory_type = self._classify_type(content, metadata)
        compressed_result = self.compressor.compress(content, mode="semantic")
        compressed = compressed_result["compressed"]
        encoding_info = self.semantic_encoder.encode(content, memory_type)

        temporal_info = self.temporal_system.get_current_ganzhi()
        dynamic_priority = self.temporal_system.calculate_priority(
            base_priority=metadata.get("priority", 5),
            ganzhi_info=temporal_info,
            memory_wuxing=encoding_info.wuxing
        )

        base_priority = self._assess_priority(content, memory_type, metadata)
        final_priority = int((base_priority + dynamic_priority.final_priority * 10) / 2)

        return {
            "entities": entities,
            "type": memory_type,
            "compressed": compressed,
            "compression_ratio": compressed_result.get("ratio", 1.0),
            "compression_entities": compressed_result.get("entities", []),
            "priority": final_priority,
            "encoding_info": {
                "hexagram_name": encoding_info.name,
                "hexagram_index": encoding_info.index,
                "wuxing": encoding_info.wuxing,
                "direction": encoding_info.direction,
                "hu_gua": encoding_info.hu_gua,
                "zong_gua": encoding_info.zong_gua,
                "cuo_gua": encoding_info.cuo_gua,
                # Task 15新增：八卦概率分布和五行得分
                "bagua_probs": getattr(encoding_info, "bagua_probs", None),
                "wuxing_scores": getattr(encoding_info, "wuxing_scores", None),
            },
            "dynamic_priority": {
                "base": round(dynamic_priority.base_priority, 3),
                "season_boost": round(dynamic_priority.season_boost, 3),
                "final": round(dynamic_priority.final_priority, 3),
            },
        }

    def _hash_embedding(self, text: str) -> list:
        """Fallback: 基于MD5的确定性伪向量（用于无PyTorch环境"""
        hash_bytes = hashlib.md5(text.encode()).digest(32)
        vec = np.frombuffer(hash_bytes, dtype=np.float32)
        # 归一化到[-1, 1]
        vec = vec / (np.linalg.norm(vec) + 1e-8) * 2 - 1
        return vec.tolist()

    def encode(self, text: str) -> list:
        """
        文本向量化

        优先使用sentence-transformers，fallback用MD5伪向量
        """
        model = get_embedding_model()
        if model is not None:
            try:
                vec = model.encode(text)
                return vec.tolist()
            except Exception as e:
                logger.warning(f"sentence-transformers encode failed: {e}, using hash fallback")
        return self._hash_embedding(text)

    async def init_belief(self, memory_id: str) -> Dict[str, Any]:
        """初始化记忆的信念状态"""
        state = self.belief_tracker.initialize(memory_id)
        return {
            "stage": state.stage,
            "confidence": state.confidence,
            "created_at": state.created_at,
        }

    async def reinforce_belief(self, memory_id: str) -> Dict[str, Any]:
        """强化记忆信念"""
        state = self.belief_tracker.reinforce(memory_id)
        return {
            "stage": state.stage,
            "confidence": state.confidence,
            "reinforce_count": state.reinforce_count,
        }

    async def shake_belief(self, memory_id: str) -> Dict[str, Any]:
        """动摇记忆信念"""
        state = self.belief_tracker.shake(memory_id)
        return {
            "stage": state.stage,
            "confidence": state.confidence,
            "shake_count": state.shake_count,
        }

    def _extract_entities(self, content: str) -> list:
        """简单实体识别"""
        entities = []

        # 数字
        numbers = re.findall(r'\d+', content)
        entities.extend([{"type": "number", "value": n} for n in numbers[:5]])

        # 时间表达式
        for pattern in [
            r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?',
            r'今天|明天|后天|昨天|前天',
            r'上周|这周|下周',
            r'去年|今年|明年',
        ]:
            matches = re.findall(pattern, content)
            entities.extend([{"type": "time", "value": m} for m in matches[:3]])

        # 关键关系词
        for word in ['我', '你', '他', '孩子', '父母', '医生', '老师', '老板']:
            if word in content:
                entities.append({"type": "relation", "value": word})

        return entities[:10]

    def _classify_type(self, content: str, metadata: Dict = None) -> str:
        """记忆类型分类"""
        if metadata and "type" in metadata:
            return metadata["type"]

        lower = content.lower()
        if any(w in lower for w in ['喜欢', '想要', '偏好', '讨厌']):
            return "preference"
        if any(w in lower for w in ['发生', '做了', '去了', '买了']):
            return "event"
        if any(w in lower for w in ['认为', '相信', '应该', '大概']):
            return "belief"
        return "fact"

    def _assess_priority(self, content: str, memory_type: str, metadata: Dict = None) -> int:
        """评估记忆优先级 0-10"""
        base = {"preference": 7, "belief": 6, "fact": 5, "event": 4}.get(memory_type, 5)
        if metadata and "priority" in metadata:
            return min(10, max(0, metadata["priority"]))
        if any(w in content for w in ['紧急', '必须', '重要']):
            base = min(10, base + 2)
        if re.search(r'\d+', content):
            base += 1
        return base
