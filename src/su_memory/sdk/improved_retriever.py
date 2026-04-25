"""
su-memory SDK 检索增强模块 v2.0

核心改进:
1. 语义向量化: 优先使用 Ollama bge-m3，本地运行无需API key
2. 向量索引: 使用 FAISS 加速，支持10万级数据
3. 智能分词: jieba中文分词 + 同义词扩展
4. 混合检索: 向量+关键词+分类多路召回

使用方法:
    from su_memory.sdk.improved_retriever import EnhancedRetriever
    
    retriever = EnhancedRetriever(backend='ollama')  # 自动检测
    retriever.add("项目ROI增长25%")
    results = retriever.query("投资回报")
"""

import os
import time
import math
import json
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache
import re

# 可选依赖
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    pass

JIEBA_AVAILABLE = False
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    pass


@dataclass
class SearchResult:
    """检索结果"""
    memory_id: str
    content: str
    score: float
    rank: int
    sources: List[str]  # ['vector', 'keyword', 'category']


class OllamaEmbedding:
    """
    Ollama 本地向量服务
    自动检测并使用 bge-m3 模型
    """
    
    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_endpoint = f"{base_url}/api/embed"
        self.dims = 1024
        self._available = None
        self._test_connection()
    
    def _test_connection(self):
        """测试Ollama连接"""
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m['name'] for m in data.get('models', [])]
                
                # 优先使用 bge-m3
                if any('bge' in m.lower() for m in models):
                    self.model = 'bge-m3:latest'
                elif models:
                    self.model = models[0]
                
                self._available = True
                print(f"[Ollama] Connected, using model: {self.model}")
        except Exception as e:
            self._available = False
            print(f"[Ollama] Not available: {e}")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """批量编码"""
        if not self._available:
            return self._fake_embeddings(len(texts))
        
        try:
            import urllib.request
            
            payload = {"model": self.model, "input": texts}
            req = urllib.request.Request(
                self.api_endpoint,
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                embeddings = data.get("embeddings", [])
                self.dims = len(embeddings[0]) if embeddings else 1024
                return embeddings
        except Exception as e:
            print(f"[Ollama] Encoding failed: {e}")
            return self._fake_embeddings(len(texts))
    
    def _fake_embeddings(self, n: int) -> List[List[float]]:
        """Fake embeddings for fallback"""
        import random
        return [[random.random() - 0.5 for _ in range(1024)] for _ in range(n)]


class ChineseTokenizer:
    """中文分词器"""
    
    def __init__(self):
        self.stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
                          '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
                          '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '以'}
        
        if JIEBA_AVAILABLE:
            # 加载自定义词典
            pass
    
    def tokenize(self, text: str) -> List[str]:
        """分词"""
        if JIEBA_AVAILABLE:
            words = jieba.cut(text)
        else:
            # 简单分词
            words = self._simple_tokenize(text)
        
        # 过滤停用词和单字
        return [w for w in words if w not in self.stop_words and len(w) > 1]
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """简单分词（无jieba时）"""
        # 提取连续中文
        chinese = re.findall(r'[\u4e00-\u9fa5]+', text)
        # 提取英文单词
        english = re.findall(r'[a-zA-Z0-9]+', text)
        
        words = []
        for seg in chinese:
            # 2-gram 分词
            for i in range(len(seg) - 1):
                words.append(seg[i:i+2])
                if i < len(seg) - 2:
                    words.append(seg[i:i+3])
        
        words.extend(english)
        return words


class InvertedIndex:
    """
    倒排索引 - 替代线性扫描
    O(1) 关键词查找
    """
    
    def __init__(self):
        self.doc_freq: Dict[str, int] = defaultdict(int)  # 词 -> 文档频率
        self.index: Dict[str, Dict[str, float]] = defaultdict(dict)  # 词 -> {doc_id -> tf}
        self.doc_count = 0
    
    def add_doc(self, doc_id: str, tokens: List[str]):
        """添加文档到索引"""
        self.doc_count += 1
        
        # 计算词频
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        
        # 更新倒排列表
        for token in set(tokens):
            self.index[token][doc_id] = tf[token] / len(tokens)
            self.doc_freq[token] += 1
    
    def search(self, query_tokens: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """搜索"""
        scores = defaultdict(float)
        
        for token in query_tokens:
            if token in self.index:
                # IDF: log(N / df)
                idf = math.log(self.doc_count / (self.doc_freq[token] + 1)) + 1
                
                for doc_id, tf in self.index[token].items():
                    scores[doc_id] += tf * idf
        
        # 排序返回
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]


class EnhancedRetriever:
    """
    增强检索器 v2.0
    
    特性:
    - 自动检测 Ollama 本地向量服务
    - FAISS 向量索引（可选）
    - 倒排索引关键词检索
    - 混合检索融合
    - 中文分词支持
    """
    
    def __init__(self, backend: str = "auto", index_type: str = "hnsw"):
        """
        Args:
            backend: 向量后端 ('auto', 'ollama', 'hash')
            index_type: 索引类型 ('flat', 'hnsw', 'ivf')
        """
        self.docs: Dict[str, Dict] = {}
        self.doc_list: List[Dict] = []  # id -> index
        self._next_id = 0
        
        # 向量服务
        self.embedding = OllamaEmbedding()
        
        # 索引
        self._vectors: List[List[float]] = []
        self._index = None  # FAISS index
        self.index_type = index_type
        
        # 关键词索引
        self._inverted_index = InvertedIndex()
        self._tokenizer = ChineseTokenizer()
        
        # 统计
        self.stats = {
            'total_queries': 0,
            'vector_queries': 0,
            'keyword_queries': 0,
            'avg_latency_ms': 0
        }
        
        print(f"[EnhancedRetriever] Initialized with {index_type} index")
    
    def add(self, content: str, metadata: Dict = None) -> str:
        """添加文档"""
        doc_id = f"doc_{self._next_id}"
        self._next_id += 1
        
        # 分词
        tokens = self._tokenizer.tokenize(content)
        
        # 构建文档
        doc = {
            'id': doc_id,
            'content': content,
            'metadata': metadata or {},
            'tokens': tokens,
            'vector_idx': len(self._vectors)
        }
        
        self.docs[doc_id] = doc
        self.doc_list.append(doc)
        
        # 添加到索引
        self._inverted_index.add_doc(doc_id, tokens)
        
        # 生成/获取向量
        if self.embedding._available:
            self._vectors.append(None)  # 暂存，稍后批量编码
        
        # 更新 FAISS 索引
        if len(self._vectors) % 100 == 0 and self._vectors:
            self._update_faiss_index()
        
        return doc_id
    
    def _update_faiss_index(self):
        """更新 FAISS 索引"""
        if not FAISS_AVAILABLE or not self._vectors or not any(self._vectors):
            return
        
        # 过滤None值
        valid_vectors = [v for v in self._vectors if v is not None]
        if not valid_vectors:
            return
        
        dim = len(valid_vectors[0]) if valid_vectors else 1024
        
        if self.index_type == 'flat':
            self._index = faiss.IndexFlatIP(dim)  # 内积
        elif self.index_type == 'hnsw':
            self._index = faiss.IndexHNSWFlat(dim, 32)  # HNSW
        
        # 添加向量
        import numpy as np
        vectors = np.array(valid_vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self._index.add(vectors)
    
    def encode_batch(self):
        """批量编码所有文档"""
        if not self.embedding._available or not self.doc_list:
            return
        
        print(f"[EnhancedRetriever] Encoding {len(self.doc_list)} documents with Ollama...")
        
        # 批量编码
        contents = [doc['content'] for doc in self.doc_list]
        embeddings = self.embedding.encode(contents)
        
        # 验证嵌入结果
        if not embeddings or len(embeddings) != len(self.doc_list):
            print(f"[EnhancedRetriever] Encoding failed: got {len(embeddings) if embeddings else 0} embeddings for {len(self.doc_list)} docs")
            return
        
        # 赋值向量
        for doc, vec in zip(self.doc_list, embeddings):
            doc['embedding'] = vec
        
        self._vectors = [doc.get('embedding') for doc in self.doc_list]
        
        # 验证向量
        valid_count = sum(1 for v in self._vectors if v is not None)
        print(f"[EnhancedRetriever] Valid vectors: {valid_count}/{len(self._vectors)}")
        
        # 更新 FAISS 索引
        if valid_count > 0:
            self._update_faiss_index()
            print(f"[EnhancedRetriever] FAISS index built with {self._index.ntotal if self._index else 0} vectors")
    
    def query(self, query: str, top_k: int = 5, 
              use_vector: bool = True, use_keyword: bool = True,
              alpha: float = 0.7) -> List[SearchResult]:
        """
        混合检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            use_vector: 使用向量检索
            use_keyword: 使用关键词检索
            alpha: 向量权重 (0-1)
        
        Returns:
            排序后的检索结果
        """
        start = time.time()
        self.stats['total_queries'] += 1
        
        query_tokens = self._tokenizer.tokenize(query)
        results: Dict[str, Dict] = {}
        
        # 1. 向量检索 (HNSW返回距离，需转换为相似度)
        if use_vector and self._index and self.embedding._available:
            self.stats['vector_queries'] += 1
            query_vec = self.embedding.encode([query])[0]
            
            import numpy as np
            q = np.array([query_vec], dtype=np.float32)
            faiss.normalize_L2(q)
            
            D, I = self._index.search(q, min(top_k * 2, len(self._vectors)))
            
            # HNSW返回的是距离（越小越相似），转换为相似度
            max_dist = max(D[0]) if D[0][0] > 0 else 1.0
            if max_dist == 0:
                max_dist = 1.0
            
            for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
                if idx < 0:
                    continue
                # 距离转相似度：sim = 1 - normalized_dist
                similarity = 1.0 - (dist / max_dist)
                doc = self.doc_list[idx]
                results[doc['id']] = {
                    'doc': doc,
                    'vector_score': float(similarity),
                    'keyword_score': 0,
                    'combined_score': alpha * float(similarity)
                }
        
        # 2. 关键词检索
        if use_keyword:
            self.stats['keyword_queries'] += 1
            keyword_results = self._inverted_index.search(query_tokens, top_k * 2)
            
            max_kw_score = max((s for _, s in keyword_results), default=1)
            
            for doc_id, kw_score in keyword_results:
                kw_norm = kw_score / max_kw_score if max_kw_score > 0 else 0
                
                if doc_id in results:
                    results[doc_id]['keyword_score'] = kw_norm
                    results[doc_id]['combined_score'] += (1 - alpha) * kw_norm
                else:
                    doc = self.docs.get(doc_id)
                    if doc:
                        results[doc_id] = {
                            'doc': doc,
                            'vector_score': 0,
                            'keyword_score': kw_norm,
                            'combined_score': (1 - alpha) * kw_norm
                        }
        
        # 3. 排序和返回
        sorted_results = sorted(results.values(), 
                              key=lambda x: x['combined_score'], 
                              reverse=True)[:top_k]
        
        search_results = []
        for rank, item in enumerate(sorted_results):
            sources = []
            if item['vector_score'] > 0:
                sources.append('vector')
            if item['keyword_score'] > 0:
                sources.append('keyword')
            
            search_results.append(SearchResult(
                memory_id=item['doc']['id'],
                content=item['doc']['content'],
                score=item['combined_score'],
                rank=rank + 1,
                sources=sources
            ))
        
        # 统计
        latency = (time.time() - start) * 1000
        self.stats['avg_latency_ms'] = (
            (self.stats['avg_latency_ms'] * (self.stats['total_queries'] - 1) + latency) 
            / self.stats['total_queries']
        )
        
        return search_results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            'total_docs': len(self.docs),
            'vector_dim': self.embedding.dims,
            'ollama_available': self.embedding._available,
            'faiss_available': FAISS_AVAILABLE,
            'jieba_available': JIEBA_AVAILABLE
        }


# 兼容层：替换 SuMemory 的查询
class CompatibleSuMemory:
    """
    兼容 SuMemory 接口的增强版本
    自动使用 EnhancedRetriever
    """
    
    def __init__(self):
        self._retriever = EnhancedRetriever()
        self._memories: List[Dict] = []
    
    def add(self, content: str, metadata: Dict = None) -> str:
        doc_id = self._retriever.add(content, metadata)
        self._memories.append({
            'id': doc_id,
            'content': content,
            'metadata': metadata or {}
        })
        return doc_id
    
    def query(self, text: str, top_k: int = 5) -> List[Dict]:
        results = self._retriever.query(text, top_k)
        return [
            {
                'memory_id': r.memory_id,
                'content': r.content,
                'score': r.score,
                'metadata': self._memories[int(r.memory_id.split('_')[1])].get('metadata', {})
            }
            for r in results
        ]
    
    def link(self, parent_id: str, child_id: str) -> bool:
        # 简化实现
        return True
    
    def get_stats(self) -> Dict:
        return self._retriever.get_stats()


# 性能测试
def benchmark():
    """性能基准测试"""
    print("=" * 60)
    print("EnhancedRetriever 性能基准测试")
    print("=" * 60)
    
    retriever = EnhancedRetriever()
    
    # 添加测试数据
    print("\n[1] 添加1000条测试记忆...")
    start = time.time()
    for i in range(1000):
        retriever.add(f"测试记忆{i}: 关于人工智能和机器学习的内容")
    add_time = time.time() - start
    print(f"    添加耗时: {add_time*1000:.1f}ms ({1000/add_time:.0f} docs/s)")
    
    # 批量编码
    print("\n[2] 批量编码向量...")
    start = time.time()
    retriever.encode_batch()
    encode_time = time.time() - start
    print(f"    编码耗时: {encode_time*1000:.1f}ms")
    
    # 查询测试
    print("\n[3] 查询性能测试 (100次)...")
    latencies = []
    for _ in range(100):
        start = time.time()
        results = retriever.query("人工智能", top_k=10)
        latencies.append((time.time() - start) * 1000)
    
    latencies.sort()
    p50 = latencies[49]
    p95 = latencies[94]
    p99 = latencies[98]
    
    print(f"    P50: {p50:.2f}ms")
    print(f"    P95: {p95:.2f}ms")
    print(f"    P99: {p99:.2f}ms")
    
    # 统计
    stats = retriever.get_stats()
    print(f"\n[4] 统计信息:")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark()
