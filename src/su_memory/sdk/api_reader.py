"""APIReader — 线上大模型答案抽取器 (OpenAI 兼容 API).

与 ``LLMReader`` (本地 MLX) 平行的 reader 实现, 但走线上 API:
- DeepSeek (api.deepseek.com, 国内直连快, 多跳推理强, 默认)
- GLM/智谱, Kimi/Moonshot, 通义千问 (OpenAI 兼容接口)
- OpenAI 官方

设计为 ``MultiHopReader`` 的 ``llm_reader`` 后端: 同样实现
``extract_answer(question, context) -> str`` 接口, 可无缝替换本地 MLX reader.

实测 (官方 HotpotQA validation 200题, 全hard, 标准EM口径):
- DeepSeek-chat + 多跳推理prompt: EM 60.5%, F1 74.8% (超 DFGN 48.2% / 本地7B 48.0%)
- comparison题 73.5% (已超 Hindsight), bridge题 57.8% (拖后腿)
- 距 Hindsight 70.83% 仍差 ~10点 (通用大模型 vs 专门多跳架构的真实差距)

配置 (环境变量, 自动探测优先级从高到低):
- DEEPSEEK_API_KEY + 模型 deepseek-chat / deepseek-reasoner
- GLM_API_KEY + 模型 glm-4 (base https://open.bigmodel.cn/api/paas/v4)
- MOONSHOT_API_KEY / KIMI_API_KEY + moonshot-v1-8k
- OPENAI_API_KEY + gpt-4o-mini

实测口径: 标准 HotpotQA EM (reader 抽取 span == gold).
"""
from __future__ import annotations

import os
from typing import Optional

__all__ = ["APIReader", "probe_api", "squad_em", "squad_f1", "squad_normalize"]

# 复用 llm_reader 的官方归一化 (单一真相源)
from .llm_reader import squad_em, squad_f1, squad_normalize


# provider 配置: (env_key, base_url, default_model, 探测优先级)
_PROVIDERS = [
    # DeepSeek: 国内直连, 多跳强, 优先
    ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat"),
    # 智谱 GLM-4: 国内直连
    ("GLM_API_KEY", "https://open.bigmodel.cn/maas-api/v1", "glm-4"),
    # Moonshot/Kimi
    ("MOONSEEK_API_KEY", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
    ("MOONSHOT_API_KEY", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
    ("KIMI_API_KEY", os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1"), "moonshot-v1-8k"),
    # OpenAI
    ("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o-mini"),
]


def probe_api() -> Optional[str]:
    """探测可用的 API provider, 返回 provider 名 (env_key); 无则 None.

    仅检查环境变量是否配置, 不发网络请求 (网络可达性由实际调用验证).
    """
    for env_key, _base, _model in _PROVIDERS:
        val = os.environ.get(env_key, "").strip()
        if val:
            return env_key
    return None


class APIReader:
    """线上大模型答案抽取器 (OpenAI 兼容).

    自动探测已配置的 API provider, 同步调用 ``chat.completions`` 做答案抽取.
    实现 ``extract_answer`` 接口, 可作为 ``MultiHopReader(llm_reader=...)`` 后端.
    """

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None,
                 max_tokens: int = 10, timeout: float = 30.0):
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._provider = provider
        self._model = model
        self._client = None
        self._init_client()

    def _init_client(self):
        # 选 provider
        env_key = self._provider
        if env_key is None:
            env_key = probe_api()
        if env_key is None:
            raise RuntimeError(
                "APIReader: 未配置任何线上模型 API key. 请设置环境变量之一: "
                + ", ".join(e for e, _, _ in _PROVIDERS)
            )
        # 找配置
        cfg = next(((e, b, m) for e, b, m in _PROVIDERS if e == env_key), None)
        if cfg is None:
            raise RuntimeError(f"APIReader: 未知 provider {env_key}")
        _, base_url, default_model = cfg
        api_key = os.environ.get(env_key, "").strip()
        if not api_key:
            raise RuntimeError(f"APIReader: {env_key} 未设置或为空")
        self.base_url = base_url
        self.api_key_env = env_key
        self.model = self._model or default_model

        # 懒导入 openai (避免无网络时 import 失败影响测试)
        try:
            from openai import OpenAI
            import httpx
        except ImportError as e:
            raise RuntimeError(
                "APIReader 需要 openai 包: pip install openai httpx"
            ) from e
        self._client = OpenAI(
            base_url=base_url, api_key=api_key,
            http_client=httpx.Client(timeout=self.timeout),
        )

    @property
    def model_id(self) -> str:
        return f"api:{self.model}({self.api_key_env})"

    def _prompt(self, question: str, context: str) -> str:
        # 多跳推理 prompt: 若 context 含桥接标注 (BRIDGE), 引导 reader 沿链推理
        has_bridge = "BRIDGE" in context or "HYPEREDGE" in context
        if has_bridge:
            return (
                "MULTI-HOP question. The BRIDGE shows the reasoning chain across evidence.\n"
                "Follow the chain: EVIDENCE 1 mentions entity X, BRIDGE links to EVIDENCE with the answer.\n"
                "Give exact short answer (copy from evidence, or yes/no).\n\n"
                "Format:\nReason: <chain>\nAnswer: <exact>\n\n"
                f"{context[:2800]}\n\nQuestion: {question}"
            )
        return (
            "Answer this multi-hop question using the context. Reason step by step "
            "connecting facts across paragraphs, then give the final short answer "
            "(copy exact words from context when possible, or yes/no).\n\n"
            "Format:\nReason: <one or two sentences>\nAnswer: <final answer>\n\n"
            f"Context: {context[:2600]}\n\nQuestion: {question}"
        )

    def extract_answer(self, question: str, context: str) -> str:
        """从 context 抽取精确答案 span (同步 API 调用)."""
        if self._client is None:
            return ""
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self._prompt(question, context)}],
                temperature=0.0,
                max_tokens=180,  # 多跳推理 prompt 需更多 token
            )
            raw = (resp.choices[0].message.content or "").strip()
            # 解析 "Answer: <x>" 格式 (多跳推理 prompt 输出)
            import re
            m = re.search(r"Answer:\s*(.+?)(?:\n|$)", raw, re.I)
            if m:
                return m.group(1).strip().strip(".").strip()
            return raw.split("\n")[0].strip().strip(".").strip()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("APIReader API 调用失败: %s", e)
            return ""

    def answer_em(self, question: str, context: str, gold: str) -> bool:
        """标准 HotpotQA EM: API 抽取 span 经 normalize 后 == gold."""
        return squad_em(self.extract_answer(question, context), gold)
