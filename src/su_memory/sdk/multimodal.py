"""
多模态嵌入模块 - MultimodalEmbedding

功能：
- 图像向量编码（基于 CLIP）
- 音频向量编码（基于 Whisper/Speech）
- 多模态融合检索

架构：
- MultimodalEmbeddingManager: 多模态管理器
- ImageEncoder: 图像编码器（CLIP）
- AudioEncoder: 音频编码器（可选）
- MultimodalFusion: 多模态融合

使用方式：
    from su_memory.sdk.multimodal import MultimodalEmbeddingManager
    
    manager = MultimodalEmbeddingManager(
        enable_image=True,
        enable_audio=False
    )
    
    # 添加多模态记忆
    manager.add_memory(
        memory_id="mem_001",
        text="一只可爱的猫咪",
        image_path="/path/to/cat.jpg"
    )
    
    # 多模态检索
    results = manager.search("猫", top_k=5, mode="multimodal")
"""

import os
import time
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


# ============================================================
# 数据结构
# ============================================================

@dataclass
class MultimodalMemory:
    """多模态记忆节点"""
    memory_id: str
    content: str  # 文本内容
    text_vector: Optional[List[float]] = None  # 文本向量
    image_vector: Optional[List[float]] = None  # 图像向量
    image_path: Optional[str] = None  # 图像路径
    audio_vector: Optional[List[float]] = None  # 音频向量
    audio_path: Optional[str] = None  # 音频路径
    timestamp: int = 0  # 时间戳
    energy_type: str = "土"  # Energy System类型
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalSearchResult:
    """多模态检索结果"""
    memory_id: str
    content: str
    score: float
    text_score: float = 0.0
    image_score: float = 0.0
    audio_score: float = 0.0
    source: str = "text"  # "text", "image", "audio", "multimodal"


# ============================================================
# CLIP 图像编码器（模拟实现）
# ============================================================

class ImageEncoder:
    """
    图像编码器（基于 CLIP）
    
    注意：实际使用时需要安装 CLIP 模型
    当前实现为模拟版本，支持图像路径存储和占位符向量
    
    安装方式：
        pip install git+https://github.com/openai/CLIP.git
        # 或使用预训练权重
    """
    
    # 默认向量维度（CLIP ViT-B/32）
    DEFAULT_DIMS = 512
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cpu",
        cache_dir: str = None
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/clip")
        
        self._model = None
        self._preprocess = None
        self._available = False
        
        # 尝试加载 CLIP
        self._try_load_clip()
    
    def _try_load_clip(self):
        """尝试加载 CLIP 模型"""
        try:
            import clip
            import torch
            
            # 加载模型
            model, preprocess = clip.load(self.model_name, device=self.device)
            self._model = model
            self._preprocess = preprocess
            self._available = True
            print(f"[ImageEncoder] CLIP 模型已加载: {self.model_name}")
        except ImportError:
            print("[ImageEncoder] CLIP 未安装，使用模拟模式")
            print("[ImageEncoder] 安装方式: pip install git+https://github.com/openai/CLIP.git")
            self._available = False
        except Exception as e:
            print(f"[ImageEncoder] CLIP 加载失败: {e}")
            self._available = False
    
    @property
    def available(self) -> bool:
        """检查模型是否可用"""
        return self._available
    
    @property
    def dims(self) -> int:
        """返回向量维度"""
        return self.DEFAULT_DIMS
    
    def encode_image(self, image_path: str) -> Optional[List[float]]:
        """
        编码图像为向量
        
        Args:
            image_path: 图像文件路径
        
        Returns:
            图像向量（512维）
        """
        if not self._available:
            # 模拟模式：返回基于路径的伪向量
            return self._simulate_encode(image_path)
        
        try:
            import clip
            import torch
            from PIL import Image
            
            # 加载图像
            image = self._preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            
            # 编码
            with torch.no_grad():
                image_features = self._model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().tolist()[0]
        except Exception as e:
            print(f"[ImageEncoder] 图像编码失败: {e}")
            return None
    
    def encode_images_batch(self, image_paths: List[str]) -> List[Optional[List[float]]]:
        """批量编码图像"""
        return [self.encode_image(p) for p in image_paths]
    
    def _simulate_encode(self, image_path: str) -> List[float]:
        """
        模拟图像编码
        
        生成基于文件路径的伪向量
        实际使用时会被真实 CLIP 向量替换
        """
        # 使用文件路径的哈希作为种子
        import hashlib
        hash_val = int(hashlib.md5(image_path.encode()).hexdigest(), 16) % (10**8)
        
        np.random.seed(hash_val)
        vector = np.random.randn(self.DEFAULT_DIMS).tolist()
        
        # 归一化
        norm = sum(x*x for x in vector) ** 0.5
        vector = [x / norm for x in vector]
        
        return vector
    
    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot / (norm1 * norm2 + 1e-8)


# ============================================================
# 音频编码器（可选）
# ============================================================

class AudioEncoder:
    """
    音频编码器（基于 Whisper/Speech）
    
    当前为模拟实现
    """
    
    DEFAULT_DIMS = 512
    
    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._available = False
        
        self._try_load_whisper()
    
    def _try_load_whisper(self):
        """尝试加载 Whisper 模型"""
        try:
            import whisper
            self._model = whisper.load_model(self.model_name, device=self.device)
            self._available = True
            print(f"[AudioEncoder] Whisper 模型已加载: {self.model_name}")
        except ImportError:
            print("[AudioEncoder] Whisper 未安装，使用模拟模式")
            self._available = False
        except Exception as e:
            print(f"[AudioEncoder] Whisper 加载失败: {e}")
            self._available = False
    
    @property
    def available(self) -> bool:
        return self._available
    
    @property
    def dims(self) -> int:
        return self.DEFAULT_DIMS
    
    def encode_audio(self, audio_path: str) -> Optional[List[float]]:
        """编码音频为向量"""
        if not self._available:
            return self._simulate_encode(audio_path)
        
        try:
            import whisper
            # 转录为文本
            result = self._model.transcribe(audio_path)
            # TODO: 使用文本生成音频向量
            return self._simulate_encode(audio_path)
        except Exception as e:
            print(f"[AudioEncoder] 音频编码失败: {e}")
            return None
    
    def _simulate_encode(self, audio_path: str) -> List[float]:
        """模拟音频编码"""
        import hashlib
        hash_val = int(hashlib.md5(audio_path.encode()).hexdigest(), 16) % (10**8)
        
        np.random.seed(hash_val + 1)  # 与图像不同的种子
        vector = np.random.randn(self.DEFAULT_DIMS).tolist()
        
        norm = sum(x*x for x in vector) ** 0.5
        vector = [x / norm for x in vector]
        
        return vector


# ============================================================
# 多模态融合管理器
# ============================================================

class MultimodalEmbeddingManager:
    """
    多模态嵌入管理器
    
    支持文本、图像、音频三种模态的嵌入和检索
    
    使用方式：
        manager = MultimodalEmbeddingManager(
            text_embedding_func=encode_func,
            enable_image=True,
            enable_audio=False
        )
        
        manager.add_multimodal_memory(
            memory_id="mem_001",
            content="猫咪图片",
            image_path="/path/to/cat.jpg"
        )
        
        # 多模态检索
        results = manager.search("可爱的猫", mode="multimodal")
    """
    
    def __init__(
        self,
        text_embedding_func: Callable[[str], List[float]] = None,
        enable_image: bool = False,
        enable_audio: bool = False,
        image_weight: float = 0.4,
        audio_weight: float = 0.3,
        text_weight: float = 0.3
    ):
        self.text_embedding_func = text_embedding_func
        self.enable_image = enable_image
        self.enable_audio = enable_audio
        
        # 权重配置
        self.image_weight = image_weight
        self.audio_weight = audio_weight
        self.text_weight = text_weight
        
        # 编码器
        self._image_encoder = ImageEncoder() if enable_image else None
        self._audio_encoder = AudioEncoder() if enable_audio else None
        
        # 存储
        self._memories: Dict[str, MultimodalMemory] = {}
        self._text_vectors: Dict[str, np.ndarray] = {}
        self._image_vectors: Dict[str, np.ndarray] = {}
        self._audio_vectors: Dict[str, np.ndarray] = {}
        
        print(f"[MultimodalEmbeddingManager] 初始化完成")
        print(f"  - 文本嵌入: {'启用' if text_embedding_func else '禁用'}")
        print(f"  - 图像编码: {'启用' if self._image_encoder else '禁用'}")
        print(f"  - 音频编码: {'启用' if self._audio_encoder else '禁用'}")
        print(f"  - 权重配置: text={text_weight}, image={image_weight}, audio={audio_weight}")
    
    @property
    def n_memories(self) -> int:
        """记忆数量"""
        return len(self._memories)
    
    def add_multimodal_memory(
        self,
        memory_id: str,
        content: str,
        text_vector: List[float] = None,
        image_path: str = None,
        image_vector: List[float] = None,
        audio_path: str = None,
        audio_vector: List[float] = None,
        timestamp: int = None,
        energy_type: str = "土",
        metadata: Dict = None
    ) -> bool:
        """
        添加多模态记忆
        
        Args:
            memory_id: 记忆ID
            content: 文本内容
            text_vector: 文本向量（可选，自动编码）
            image_path: 图像路径（可选）
            image_vector: 图像向量（可选，自动从 image_path 编码）
            audio_path: 音频路径（可选）
            audio_vector: 音频向量（可选）
            timestamp: 时间戳
            energy_type: Energy System类型
            metadata: 元数据
        
        Returns:
            是否添加成功
        """
        ts = timestamp or int(time.time())
        
        # 编码文本向量
        if text_vector is None and self.text_embedding_func:
            text_vector = self.text_embedding_func(content)
        
        # 编码图像向量
        if image_vector is None and image_path and self._image_encoder:
            image_vector = self._image_encoder.encode_image(image_path)
        
        # 编码音频向量
        if audio_vector is None and audio_path and self._audio_encoder:
            audio_vector = self._audio_encoder.encode_audio(audio_path)
        
        # 创建记忆节点
        memory = MultimodalMemory(
            memory_id=memory_id,
            content=content,
            text_vector=text_vector,
            image_vector=image_vector,
            image_path=image_path,
            audio_vector=audio_vector,
            audio_path=audio_path,
            timestamp=ts,
            energy_type=energy_type,
            metadata=metadata or {}
        )
        
        self._memories[memory_id] = memory
        
        # 存储向量
        if text_vector:
            self._text_vectors[memory_id] = np.array(text_vector, dtype=np.float32)
        if image_vector:
            self._image_vectors[memory_id] = np.array(image_vector, dtype=np.float32)
        if audio_vector:
            self._audio_vectors[memory_id] = np.array(audio_vector, dtype=np.float32)
        
        return True
    
    def search(
        self,
        query: str,
        query_image: str = None,
        query_audio: str = None,
        top_k: int = 5,
        mode: str = "text"  # "text", "image", "audio", "multimodal"
    ) -> List[MultimodalSearchResult]:
        """
        多模态检索
        
        Args:
            query: 文本查询
            query_image: 图像查询路径
            query_audio: 音频查询路径
            top_k: 返回数量
            mode: 检索模式
                - "text": 仅文本检索
                - "image": 仅图像检索
                - "audio": 仅音频检索
                - "multimodal": 多模态融合检索
        
        Returns:
            检索结果列表
        """
        if not self._memories:
            return []
        
        results = []
        
        # 编码查询向量
        query_vec = None
        if self.text_embedding_func:
            query_vec = self.text_embedding_func(query)
        
        query_image_vec = None
        if query_image and self._image_encoder:
            query_image_vec = self._image_encoder.encode_image(query_image)
        
        query_audio_vec = None
        if query_audio and self._audio_encoder:
            query_audio_vec = self._audio_encoder.encode_audio(query_audio)
        
        # 计算相似度
        for memory_id, memory in self._memories.items():
            text_score = 0.0
            image_score = 0.0
            audio_score = 0.0
            
            # 文本相似度
            if query_vec and memory.text_vector:
                text_score = self._cosine_similarity(query_vec, memory.text_vector)
            
            # 图像相似度
            if query_image_vec and memory.image_vector:
                image_score = self._cosine_similarity(query_image_vec, memory.image_vector)
            
            # 音频相似度
            if query_audio_vec and memory.audio_vector:
                audio_score = self._cosine_similarity(query_audio_vec, memory.audio_vector)
            
            # 计算综合得分
            final_score = 0.0
            source = "text"
            
            if mode == "text":
                final_score = text_score
            elif mode == "image":
                final_score = image_score
                source = "image"
            elif mode == "audio":
                final_score = audio_score
                source = "audio"
            elif mode == "multimodal":
                # 多模态融合
                w1, w2, w3 = self.text_weight, self.image_weight, self.audio_weight
                final_score = w1 * text_score + w2 * image_score + w3 * audio_score
                source = "multimodal"
                
                # 如果有任一模态匹配，标记来源
                if image_score > text_score and image_score > audio_score:
                    source = "image"
                elif audio_score > text_score and audio_score > image_score:
                    source = "audio"
            
            results.append(MultimodalSearchResult(
                memory_id=memory_id,
                content=memory.content,
                score=final_score,
                text_score=text_score,
                image_score=image_score,
                audio_score=audio_score,
                source=source
            ))
        
        # 排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        return dot / (norm1 * norm2 + 1e-8)
    
    def get_memory(self, memory_id: str) -> Optional[MultimodalMemory]:
        """获取记忆"""
        return self._memories.get(memory_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "n_memories": len(self._memories),
            "text_vectors": len(self._text_vectors),
            "image_vectors": len(self._image_vectors),
            "audio_vectors": len(self._audio_vectors),
            "image_encoder_available": self._image_encoder.available if self._image_encoder else False,
            "audio_encoder_available": self._audio_encoder.available if self._audio_encoder else False,
            "weights": {
                "text": self.text_weight,
                "image": self.image_weight,
                "audio": self.audio_weight
            }
        }


# ============================================================
# 工厂函数
# ============================================================

def create_multimodal_manager(
    text_embedding_func: Callable[[str], List[float]] = None,
    enable_image: bool = False,
    enable_audio: bool = False,
    image_weight: float = 0.4,
    audio_weight: float = 0.3,
    text_weight: float = 0.3
) -> MultimodalEmbeddingManager:
    """
    创建多模态嵌入管理器
    
    Args:
        text_embedding_func: 文本嵌入函数
        enable_image: 是否启用图像编码
        enable_audio: 是否启用音频编码
        image_weight: 图像权重
        audio_weight: 音频权重
        text_weight: 文本权重
    
    Returns:
        MultimodalEmbeddingManager 实例
    """
    return MultimodalEmbeddingManager(
        text_embedding_func=text_embedding_func,
        enable_image=enable_image,
        enable_audio=enable_audio,
        image_weight=image_weight,
        audio_weight=audio_weight,
        text_weight=text_weight
    )