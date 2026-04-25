"""
MemoryManager - 记忆管理器核心（Phase 2增强版）

协调记忆的增删改查、压缩、检索、冲突消解全流程
Phase 2: 完整集成su_core核心能力
"""

import uuid
import time
import logging
from datetime import date
from typing import List, Dict, Any, Optional

from storage.vector_db import VectorDB
from storage.relational_db import RelationalDB
from .extractor import MemoryExtractor
from .retriever import MemoryRetriever
from .conflict_resolver import ConflictResolver
from .forgetting import ForgettingEngine

from su_core import BeliefTracker, MetaCognition

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    记忆管理器 - Phase 2增强版
    
    核心职责：
    1. 记忆写入（Retain）→ 完整提取+编码+压缩+信念初始化
    2. 记忆检索（Recall）→ 多视图融合检索
    3. 冲突检测与消解（Reflect）
    4. 遗忘触发
    5. 信念追踪（BeliefTracker集成）
    6. 元认知（MetaCognition集成）
    """
    
    def __init__(self):
        self.vector_db = VectorDB()
        self.relational_db = RelationalDB()
        self.extractor = MemoryExtractor()  # Phase 2增强版
        self.retriever = MemoryRetriever(self.vector_db)
        self.conflict_resolver = ConflictResolver()
        self.forgetting_engine = ForgettingEngine()
        # Phase 2新增：su_core核心能力
        self.belief_tracker = BeliefTracker()
        self.meta_cognition = MetaCognition()
        
        logger.info("MemoryManager Phase 2 initialized with su_core integration")
    
    async def create_tenant(self, name: str, plan: str) -> Dict[str, Any]:
        """创建租户"""
        tenant_id = str(uuid.uuid4())
        api_key = f"sk_{uuid.uuid4().hex}"
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        await self.relational_db.create_tenant(
            tenant_id=tenant_id,
            name=name,
            plan=plan,
            api_key=api_key
        )
        
        await self.vector_db.create_collection(tenant_id)
        
        logger.info(f"Created tenant: {name} ({tenant_id})")
        
        return {
            "tenant_id": tenant_id,
            "name": name,
            "api_key": api_key,
            "created_at": created_at
        }
    
    async def add_memory(
        self,
        tenant_id: str,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        写入一条记忆（Phase 2增强）
        
        Phase 2流程：
        1. su_core完整提取（编码+压缩+动态优先级）
        2. 生成向量
        3. 存入向量库 + 关系库（含卦象索引）
        4. 初始化信念状态
        5. 触发元认知检查（可选）
        """
        memory_id = str(uuid.uuid4())
        timestamp = int(time.time())
        metadata = metadata or {}
        
        # Phase 2: su_core完整提取
        extracted = await self.extractor.extract(content, metadata)
        
        # 生成向量
        embedding = await self.extractor.encode(content)
        
        # Phase 2: 准备完整payload（含卦象信息）
        payload = {
            "user_id": user_id,
            "content": content,
            "content_compressed": extracted.get("compressed", content),
            "memory_type": extracted.get("type", "fact"),
            "timestamp": timestamp,
            "metadata": metadata,
            # Phase 2新增：卦象编码信息
            "hexagram_index": extracted.get("encoding_info", {}).get("hexagram_index", 0),
            "hexagram_name": extracted.get("encoding_info", {}).get("hexagram_name", ""),
            "wuxing": extracted.get("encoding_info", {}).get("wuxing", "土"),
            "direction": extracted.get("encoding_info", {}).get("direction", ""),
            # 动态优先级
            "dynamic_priority": extracted.get("dynamic_priority", {}).get("final", 0.5),
            # 全息视角
            "hu_gua": extracted.get("encoding_info", {}).get("hu_gua", 0),
            "zong_gua": extracted.get("encoding_info", {}).get("zong_gua", 0),
            "cuo_gua": extracted.get("encoding_info", {}).get("cuo_gua", 0),
        }
        
        # Task 15新增：干支时空标注
        temporal_info = self.extractor.temporal_system.get_current_ganzhi()
        payload["ganzhi"] = temporal_info.ganzhi
        payload["ganzhi_wuxing"] = temporal_info.wuxing
        payload["season"] = temporal_info.season
        payload["jiazi_position"] = self.extractor.temporal_system.get_jiazi_position(date.today())
        
        # 使用完整月令旺相计算动态优先级
        today = date.today()
        monthly_states = self.extractor.temporal_system.get_monthly_wuxing_state(today.month, today.day)
        payload["wuxing_state"] = monthly_states.get(payload["wuxing"], "休")
        
        # Task 15新增：存储八卦概率分布和五行得分（供融合检索使用）
        encoding_info = extracted.get("encoding_info", {})
        if "bagua_probs" in encoding_info and encoding_info["bagua_probs"] is not None:
            payload["bagua_probs"] = encoding_info["bagua_probs"]
        if "wuxing_scores" in encoding_info and encoding_info["wuxing_scores"] is not None:
            payload["wuxing_scores"] = encoding_info["wuxing_scores"]
        
        # 存入向量库
        await self.vector_db.insert(
            collection=tenant_id,
            id=memory_id,
            vector=embedding,
            payload=payload
        )
        
        # 存入关系库
        await self.relational_db.insert_memory(
            memory_id=memory_id,
            tenant_id=tenant_id,
            user_id=user_id,
            content=content,
            compressed_content=extracted.get("compressed", content),
            memory_type=extracted.get("type", "fact"),
            priority=extracted.get("priority", 5),
            timestamp=timestamp,
            hexagram_index=payload["hexagram_index"],
            wuxing=payload["wuxing"],
            dynamic_priority=payload["dynamic_priority"]
        )
        
        # Phase 2: 初始化信念状态
        await self.extractor.init_belief(memory_id)
        
        logger.info(f"Memory stored Phase 2: {memory_id} ({payload['hexagram_name']}卦, {payload['wuxing']}, 优先级{extracted['priority']})")
        
        return memory_id
    
    async def query_memory(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        limit: int = 8,
        use_holographic: bool = True,
        use_multi_hop: bool = False,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        检索记忆（Phase 2增强 + Task 15多跳检索）
        
        Phase 2流程：
        1. 生成查询向量
        2. 多视图融合检索（use_holographic=True）
        3. 时序重排
        4. 返回top-k（含全息得分）
        
        Task 15新增：
        - use_multi_hop=True 时，调用 retrieve_multi_hop() 替代标准 retrieve()
        """
        # 生成查询向量
        query_vector = await self.extractor.encode(query)
        
        # Task 15: 多跳检索
        if use_multi_hop:
            results = await self.retriever.retrieve_multi_hop(
                collection=tenant_id,
                query_vector=query_vector,
                user_id=user_id,
                query_text=query,
                limit=limit,
                max_hops=max_hops,
                use_holographic=use_holographic
            )
            logger.debug(f"Query returned {len(results)} memories (multi_hop, max_hops={max_hops})")
            return results
        
        # 多视图检索
        results = await self.retriever.retrieve(
            collection=tenant_id,
            query_vector=query_vector,
            user_id=user_id,
            query_text=query,
            limit=limit,
            use_holographic=use_holographic
        )
        
        logger.debug(f"Query returned {len(results)} memories (holographic={use_holographic})")
        
        return results
    
    async def delete_memory(
        self,
        tenant_id: str,
        user_id: str,
        memory_id: str
    ):
        """删除记忆"""
        await self.vector_db.delete(collection=tenant_id, id=memory_id)
        await self.relational_db.delete_memory(memory_id)
        
        logger.info(f"Memory deleted: {memory_id}")
    
    async def reinforce_memory(self, memory_id: str) -> Dict[str, Any]:
        """强化记忆信念"""
        result = await self.extractor.reinforce_belief(memory_id)
        return result
    
    async def shake_memory(self, memory_id: str) -> Dict[str, Any]:
        """动摇记忆信念"""
        result = await self.extractor.shake_belief(memory_id)
        return result
    
    async def get_stats(self, tenant_id: str, user_id: str) -> Dict[str, Any]:
        """获取记忆统计"""
        stats = await self.relational_db.get_user_stats(tenant_id, user_id)
        return {
            "user_id": user_id,
            "total_memories": stats.get("total", 0),
            "active_memories": stats.get("active", 0),
            "archived_memories": stats.get("archived", 0),
            "storage_bytes": stats.get("bytes", 0)
        }
    
    async def get_belief_stats(self) -> Dict[str, Any]:
        """获取信念统计（Phase 2新增）"""
        distribution = self.belief_tracker.get_stage_distribution()
        return {
            "stage_distribution": distribution,
            "total_beliefs": sum(distribution.values())
        }
    
    async def get_cognitive_gaps(self, memory_types: Dict[str, int], user_domains: List[str]) -> List[Dict]:
        """获取认知空洞（Phase 2新增）"""
        all_memories = await self.query_memory(
            tenant_id="system",
            user_id="system",
            query="",
            limit=100
        )
        return self.meta_cognition.discover_gaps(
            memory_types=memory_types,
            user_domains=user_domains,
            memory_list=[{"id": m["id"], "type": m["memory_type"], "timestamp": m["timestamp"]} for m in all_memories]
        )
