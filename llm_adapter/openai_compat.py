"""
LLMAdapter - 统一LLM接口

兼容所有OpenAI兼容的大模型：
- Ollama
- vLLM
- 通义千问
- Qwen私有化
- 任何OpenAI兼容接口
"""

import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import httpx

logger = logging.getLogger(__name__)


class LLMAdapter:
    """
    统一的大模型适配器
    
    支持所有OpenAI兼容接口的模型
    """
    
    def __init__(self):
        # 从环境变量读取配置
        self.provider = os.getenv("LLM_PROVIDER", "ollama")
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.api_key = os.getenv("LLM_API_KEY", "ollama")
        self.default_model = os.getenv("LLM_MODEL", "qwen2.5:7b")
        
        # 创建OpenAI兼容客户端
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=httpx.Client(timeout=120.0)
        )
        
        logger.info(f"LLMAdapter initialized: provider={self.provider}, model={self.default_model}")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """
        发起对话请求
        
        Args:
            messages: [{"role": "user", "content": "..."}]
            model: 模型名称（可选，默认用配置的模型）
            temperature: 温度参数
            max_tokens: 最大生成长度
        
        Returns:
            OpenAI格式的响应
        """
        model = model or self.default_model
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return {
                "id": response.id,
                "model": response.model,
                "choices": [
                    {
                        "index": c.index,
                        "message": {
                            "role": c.message.role,
                            "content": c.message.content
                        },
                        "finish_reason": c.finish_reason
                    }
                    for c in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            raise
    
    async def embed(self, texts: List[str], model: str = None) -> List[List[float]]:
        """
        文本嵌入
        
        备注：embedding通常走专门的embedding模型，不走这个接口
        """
        # 这个方法主要给需要统一接口的场景用
        # 实际embedding在extractor.py中通过sentence-transformers处理
        raise NotImplementedError("Use sentence-transformers for embeddings")
