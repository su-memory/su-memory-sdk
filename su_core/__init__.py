"""
su_core 核心模块 - 对外接口

Phase 1: 易理核心层完整接口

本模块通过明文接口提供功能，不暴露内部实现细节。
对外只能看到：接口类、参数、返回值
内部算法完全黑盒封装在 _sys 目录中。
"""

# ============================================================
# Phase 1 对外接口（从_sys导入并重新导出）
# ============================================================

# 语义编码器（文本→高维向量编码）
from ._sys.encoders import SemanticEncoder, EncoderCore, EncodingInfo

# 八卦语义推断
from ._sys._c1 import infer_bagua_soft, infer_bagua_from_content, get_bagua_relations

# 多视图检索融合
from ._sys.fusion import MultiViewRetriever

# 压缩编解码
from ._sys.codec import SuCompressor

# 时序系统（动态优先级调度）
from ._sys.chrono import TemporalSystem, TemporalInfo, DynamicPriority

# 状态追踪（信念生命周期）
from ._sys.states import BeliefTracker, BeliefState, BeliefStage

# 认知感知（主动发现知识空洞）
from ._sys.awareness import MetaCognition, CognitiveGap, KnowledgeAging

# 因果推理引擎
from ._sys.causal import CausalInference

# 周易推理引擎
from ._sys.yijing import YiJingInference


# ============================================================
# 公开接口清单（供外部调用）
# ============================================================

__all__ = [
    # 语义编码器
    "SemanticEncoder",      # 文本→向量编码
    "EncoderCore",          # 编码核心系统
    "EncodingInfo",         # 编码信息
    
    # 八卦语义推断
    "infer_bagua_soft",     # 八卦概率分布推断
    "infer_bagua_from_content",  # 八卦归属推断
    "get_bagua_relations",  # 八卦关系判断
    
    # 多视图检索
    "MultiViewRetriever",   # 多视图检索器
    
    # 压缩编解码
    "SuCompressor",         # 压缩引擎
    
    # 时序系统
    "TemporalSystem",       # 时序系统
    "TemporalInfo",         # 时序信息
    "DynamicPriority",      # 动态优先级
    
    # 状态追踪
    "BeliefTracker",        # 信念追踪器
    "BeliefState",          # 信念状态
    "BeliefStage",          # 信念阶段
    
    # 认知感知
    "MetaCognition",        # 元认知系统
    "CognitiveGap",         # 认知空洞
    "KnowledgeAging",       # 知识老化
    
    # 推理引擎
    "CausalInference",      # 因果推理引擎
    "YiJingInference",      # 周易推理引擎
]


# ============================================================
# 版本信息
# ============================================================

__version__ = "1.0.0"
__phase__ = "Phase 1"
