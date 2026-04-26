"""
Embedding Plugin (嵌入插件)

v1.7.0 W25-W26 官方插件示例

基于简单hash的文本向量化插件，用于演示插件接口使用。

Features:
- 轻量级hash向量化实现
- 支持批量处理
- 基于N-gram的特征提取

【Pre-Phase Numeric】- Uses prior ordering for numerical calculations
【Post-Phase Symbolic】- Uses post ordering for symbolic applications
"""

from typing import Dict, List, Optional, Any
import hashlib
import re
from collections import Counter

from .._sys._plugin_interface import (
    PluginInterface,
    PluginType,
    create_plugin_metadata,
)


# =============================================================================
# Hash Embedding Vectorizer
# =============================================================================

class HashVectorizer:
    """
    基于Hash的轻量级向量化器。
    
    将文本转换为固定维度的数值向量。
    使用n-gram特征和hash编码。
    
    Example:
        >>> vectorizer = HashVectorizer(dimension=128, ngram_range=(1, 3))
        >>> vector = vectorizer.transform("Hello world")
        >>> print(f"Vector shape: {len(vector)}")
    """
    
    def __init__(
        self,
        dimension: int = 128,
        ngram_range: tuple = (1, 3),
        hash_seed: int = 42,
    ):
        """
        初始化向量化器。
        
        Args:
            dimension: 向量维度
            ngram_range: N-gram范围 (min_n, max_n)
            hash_seed: Hash随机种子
        """
        self._dimension = dimension
        self._min_n, self._max_n = ngram_range
        self._hash_seed = hash_seed
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        # 简单分词：按空格和标点分割，转小写
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def _generate_ngrams(self, tokens: List[str]) -> List[str]:
        """生成N-grams"""
        ngrams = []
        for n in range(self._min_n, self._max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = "_".join(tokens[i:i+n])
                ngrams.append(ngram)
        return ngrams
    
    def _hash_token(self, token: str) -> int:
        """Hash token到整数"""
        hash_obj = hashlib.md5(
            f"{self._hash_seed}_{token}".encode('utf-8')
        )
        return int(hash_obj.hexdigest(), 16)
    
    def transform(self, text: str) -> List[float]:
        """
        将文本转换为向量。
        
        Args:
            text: 输入文本
        
        Returns:
            归一化向量
        """
        tokens = self._tokenize(text)
        ngrams = self._generate_ngrams(tokens)
        
        # 计算词频
        frequencies = Counter(ngrams)
        
        # 初始化向量
        vector = [0.0] * self._dimension
        
        # 填充向量
        for ngram, freq in frequencies.items():
            idx = self._hash_token(ngram) % self._dimension
            vector[idx] += freq
        
        # L2归一化
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
    def transform_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量转换文本。
        
        Args:
            texts: 文本列表
        
        Returns:
            向量列表
        """
        return [self.transform(text) for text in texts]


# =============================================================================
# Text Embedding Plugin
# =============================================================================

class TextEmbeddingPlugin(PluginInterface):
    """
    文本嵌入插件。
    
    提供文本到向量的转换功能。
    基于HashVectorizer实现轻量级向量化。
    
    Example:
        >>> plugin = TextEmbeddingPlugin()
        >>> plugin.initialize({"dimension": 256})
        >>> 
        >>> result = plugin.execute({
        ...     "text": "Hello world",
        ...     "operation": "embed"
        ... })
        >>> vector = result["vector"]
    """
    
    def __init__(self):
        """初始化插件"""
        self._initialized = False
        self._vectorizer: Optional[HashVectorizer] = None
        self._config: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        return "text_embedding_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "基于Hash的轻量级文本嵌入插件，提供文本向量化功能"
    
    @property
    def author(self) -> str:
        return "su-memory-sdk"
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.EMBEDDING
    
    @property
    def dependencies(self) -> List[str]:
        return []
    
    @property
    def config_schema(self) -> Dict[str, Any]:
        return {
            "required": [],
            "properties": {
                "dimension": {
                    "type": "integer",
                    "default": 128,
                    "description": "向量维度"
                },
                "ngram_range": {
                    "type": "array",
                    "default": [1, 3],
                    "description": "N-gram范围"
                },
                "hash_seed": {
                    "type": "integer",
                    "default": 42,
                    "description": "Hash随机种子"
                }
            }
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        初始化插件。
        
        Args:
            config: 配置字典
        
        Returns:
            True表示初始化成功
        """
        try:
            self._config = config
            
            dimension = config.get("dimension", 128)
            ngram_range = tuple(config.get("ngram_range", [1, 3]))
            hash_seed = config.get("hash_seed", 42)
            
            self._vectorizer = HashVectorizer(
                dimension=dimension,
                ngram_range=ngram_range,
                hash_seed=hash_seed,
            )
            
            self._initialized = True
            return True
            
        except Exception as e:
            self._initialized = False
            return False
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """
        执行嵌入操作。
        
        Args:
            context: 执行上下文
                - text: str, 输入文本
                - operation: str, 操作类型 ("embed", "batch_embed")
                - texts: List[str], 批量文本（用于batch_embed）
        
        Returns:
            执行结果字典
                - vector: List[float], 单个向量（embed）
                - vectors: List[List[float]], 向量列表（batch_embed）
                - dimension: int, 向量维度
        """
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")
        
        operation = context.get("operation", "embed")
        
        if operation == "embed":
            text = context.get("text", "")
            if not text:
                raise ValueError("Missing 'text' in context")
            
            vector = self._vectorizer.transform(text)
            
            return {
                "vector": vector,
                "dimension": len(vector),
                "text": text,
                "success": True,
            }
        
        elif operation == "batch_embed":
            texts = context.get("texts", [])
            if not texts:
                raise ValueError("Missing 'texts' in context")
            
            vectors = self._vectorizer.transform_batch(texts)
            
            return {
                "vectors": vectors,
                "count": len(vectors),
                "dimension": len(vectors[0]) if vectors else 0,
                "success": True,
            }
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def cleanup(self) -> None:
        """清理资源"""
        self._vectorizer = None
        self._initialized = False
        self._config = {}


# =============================================================================
# Plugin Factory
# =============================================================================

def create_text_embedding_plugin() -> TextEmbeddingPlugin:
    """创建文本嵌入插件实例"""
    return TextEmbeddingPlugin()


# =============================================================================
# Test Suite
# =============================================================================

def test_embedding_plugin():
    """测试嵌入插件"""
    print("=" * 60)
    print("Testing Text Embedding Plugin")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    def test(name: str, condition: bool):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}")
            failed += 1
    
    # Test 1: 插件创建
    print("\n[Test 1] Plugin Creation")
    print("-" * 40)
    
    plugin = TextEmbeddingPlugin()
    test("插件创建", plugin is not None)
    test("插件名称", plugin.name == "text_embedding_plugin")
    test("插件版本", plugin.version == "1.0.0")
    test("插件类型", plugin.plugin_type == PluginType.EMBEDDING)
    
    # Test 2: 初始化
    print("\n[Test 2] Initialization")
    print("-" * 40)
    
    config = {"dimension": 64, "ngram_range": [1, 2]}
    success = plugin.initialize(config)
    test("初始化成功", success)
    
    # Test 3: 单个文本嵌入
    print("\n[Test 3] Single Embedding")
    print("-" * 40)
    
    result = plugin.execute({
        "operation": "embed",
        "text": "Hello world",
    })
    
    test("执行成功", result.get("success"))
    test("向量存在", "vector" in result)
    test("向量维度", result.get("dimension") == 64)
    
    # Test 4: 批量嵌入
    print("\n[Test 4] Batch Embedding")
    print("-" * 40)
    
    result = plugin.execute({
        "operation": "batch_embed",
        "texts": ["Hello", "World", "Test"],
    })
    
    test("批量执行成功", result.get("success"))
    test("向量数量", result.get("count") == 3)
    
    # Test 5: 向量相似性
    print("\n[Test 5] Vector Similarity")
    print("-" * 40)
    
    result1 = plugin.execute({"operation": "embed", "text": "hello world"})
    result2 = plugin.execute({"operation": "embed", "text": "hello world"})
    result3 = plugin.execute({"operation": "embed", "text": "goodbye universe"})
    
    v1 = result1["vector"]
    v2 = result2["vector"]
    v3 = result3["vector"]
    
    # 计算余弦相似度
    def cosine_sim(a, b):
        return sum(x * y for x, y in zip(a, b))
    
    sim_same = cosine_sim(v1, v2)
    sim_diff = cosine_sim(v1, v3)
    
    test("相同文本相似度高", sim_same > 0.99)
    test("不同文本相似度低", sim_diff < sim_same)
    
    # Test 6: 清理
    print("\n[Test 6] Cleanup")
    print("-" * 40)
    
    plugin.cleanup()
    test("清理完成", True)
    
    # Test 7: 配置验证
    print("\n[Test 7] Config Validation")
    print("-" * 40)
    
    metadata = plugin.get_metadata()
    test("元数据存在", metadata is not None)
    test("配置模式", "dimension" in metadata.config_schema.get("properties", {}))
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = test_embedding_plugin()
    exit(0 if success else 1)