"""
LLM 答案提取重排序器 — LLMReranker

用于 LongMemEval 等检索评测场景：在向量检索结果基础上，
调 LLM (Ollama 本地 / DeepSeek 云端) 从时序上下文中直接提取答案，
然后与 gold_answer 做语义匹配。

用法:
    from su_memory.sdk._llm_reranker import LLMReranker, create_reranker
    reranker = create_reranker("qwen3.6:27b", provider="ollama")
    reranker = create_reranker("deepseek-chat", provider="deepseek")
    llm_answer = reranker.answer_from_context("用户问题", ["chunk1", "chunk2", ...])
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

# 默认模型
DEFAULT_MODEL = "gemma4"
DEFAULT_TIMEOUT = 30  # 秒
DEFAULT_NUM_PREDICT = 500  # token 数（gemma4 需要较大值以容纳内部推理）


class LLMReranker:
    """
    LLM 答案提取重排序器。

    读取检索到的 chunk 上下文，用 LLM 直接回答问题，
    然后由调用方与 gold_answer 做语义匹配。

    v3.5.9: 支持多 provider — ollama (本地) / deepseek (云端)
    v4.4.0: 新增 minimax / glm 云端 provider
    """

    # v4.4.0: 默认模型映射
    _DEFAULT_MODELS: dict[str, str] = {
        "deepseek": "deepseek-chat",
        "minimax": "abab6.5s-chat",
        "glm": "glm-4-flash",
        "openai": "gpt-4o-mini",
        "ollama": DEFAULT_MODEL,
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        provider: str = "ollama",
        ollama_url: str = "http://localhost:11434",
        timeout: int = DEFAULT_TIMEOUT,
        num_predict: int = DEFAULT_NUM_PREDICT,
        max_chunks: int = 70,          # v3.6.1: 50→70，配合召回窗口扩大
        chunk_char_limit: int = 700,
        max_chunks_temporal: int = 100,   # v3.6.1: 80→100，时序推理需更大上下文
        max_chunks_multisession: int = 80,  # v3.6.1: 60→80，多会话需更多chunk
    ):
        self.model = model
        self.provider = provider
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        self.num_predict = num_predict
        # v3.5.9: DeepSeek API 配置
        self._deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self._deepseek_base_url = os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
        ).rstrip("/")
        # v4.1: OpenAI API 配置
        self._openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self._openai_base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        # v4.4.0: MiniMax API 配置
        self._minimax_api_key = os.environ.get("MINIMAX_API_KEY", "")
        self._minimax_base_url = os.environ.get(
            "MINIMAX_BASE_URL", "https://api.minimax.chat/v1"
        ).rstrip("/")
        # v4.4.0: GLM (智谱) API 配置
        self._glm_api_key = os.environ.get("GLM_API_KEY", os.environ.get("ZHIPU_API_KEY", ""))
        self._glm_base_url = os.environ.get(
            "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
        ).rstrip("/")
        # v3.5.9: 可配置上下文窗口参数
        self.max_chunks = max_chunks
        self.chunk_char_limit = chunk_char_limit
        self.max_chunks_temporal = max_chunks_temporal
        self.max_chunks_multisession = max_chunks_multisession
        # v4.4.0: auto 模式—自动检测可用 provider
        if provider == "auto":
            self._resolve_auto_provider()
        # v4.4.0: 未指定 model 或仍为通用默认值时，使用 provider 专属默认模型
        if not self.model or self.model == DEFAULT_MODEL:
            self.model = self._DEFAULT_MODELS.get(self.provider, DEFAULT_MODEL)

    def answer_from_context(
        self,
        question: str,
        context_chunks: list[str],
        question_type: str = "unknown",
    ) -> str:
        """
        从检索上下文用 LLM 提取答案。

        Args:
            question: 用户问题
            context_chunks: 按时间序排列的检索结果 content 列表
            question_type: 问题类型（single-session / multi-session /
                          temporal-reasoning / knowledge-update）

        Returns:
            LLM 生成的答案文本（失败时返回空字符串）
        """
        if not context_chunks:
            return ""

        # v3.5.9: 根据问题类型动态调整上下文大小
        if question_type == "temporal-reasoning":
            max_c = self.max_chunks_temporal
        elif question_type == "multi-session":
            max_c = self.max_chunks_multisession
        else:
            max_c = self.max_chunks
        char_limit = self.chunk_char_limit

        # v3.5.9: 句子边界感知截断 + chunk 位置标签
        truncated = []
        total_chunks = len(context_chunks)
        for i, c in enumerate(context_chunks[:max_c]):
            # 位置标签
            pos_label = f"[Chunk {i + 1}/{min(total_chunks, max_c)}]"
            if len(c) > char_limit:
                # 在 char_limit 附近查找最近的句子边界
                boundary_chars = ['. ', '\n', '? ', '! ']
                best_cut = char_limit
                for bch in boundary_chars:
                    pos = c.rfind(bch, int(char_limit * 0.8), char_limit + 100)
                    if pos > 0:
                        best_cut = pos + len(bch)
                        break
                truncated.append(pos_label + " " + c[:best_cut] + "…")
            else:
                truncated.append(pos_label + " " + c)

        context = "\n---\n".join(truncated)

        # v3.6.0: 增强 prompt hints，根据维度给出更具体的指令
        # Phase 1: V4-Pro 针对性优化 — 更严格的输出格式约束 + few-shot
        # v4.0: 事实感知提示 — [FACT] 标签的结构化事实优先级高于原始对话
        type_hints = {
            "temporal-reasoning": (
                "Chunks are labelled with position numbers [Chunk N/M] and session/time labels. "
                "[FACT] items are pre-extracted structured facts with resolved references — TRUST THEM over raw chunks. "
                "Answer based on WHEN information was mentioned. "
                "For questions about 'earliest' or 'first', look at chronologically earliest chunks. "
                "For questions about 'latest' or 'most recent', look at chronologically latest chunks."
            ),
            "multi-session": (
                "Information may appear across multiple labelled chunks from different sessions. "
                "[FACT] items are pre-extracted structured facts — they summarize key information across sessions. "
                "Compare and synthesize details mentioned at different times. "
                "For counting questions, list all matching items before answering."
            ),
            "knowledge-update": (
                "CRITICAL: [FACT] items with type:experience are the most reliable for current values. "
                "Chunks are ordered with most recent FIRST. "
                "[LATEST] marks the newest information. [N-1], [N-2] mark progressively older information. "
                "If the same entity (person, number, location) appears in multiple chunks with different values, "
                "OUTPUT ONLY the [LATEST] value. DO NOT mention the update, change, or old value. "
                "Simply state the most recent fact as if it were the only one."
            ),
            "single-session-preference": (
                "[FACT] items with pref:preference or pref:anti_preference signals are the most reliable. "
                "Look for explicit user preferences (likes, dislikes, favorites). "
                "If a preference is stated, output it directly."
            ),
            "single-session": "",
        }
        hint = type_hints.get(question_type, "")

        # v3.6.0: 输出格式约束
        output_constraint = (
            "IMPORTANT: Output ONLY the exact fact, value, or short phrase that answers the question. "
            "Do NOT write a sentence, explanation, or commentary. "
            "Do NOT say 'the answer is', 'according to', or 'based on'. "
            "Just output the answer directly. Maximum 20 words."
        )

        # v3.6.0: Few-shot 示例（针对 V4-Pro 倾向生成冗长回答的问题）
        few_shot = ""
        if question_type == "knowledge-update":
            few_shot = """
Examples of correct answers:
Q: What city does Alice live in?
A: Boston

Q: How many children does Bob have?
A: 3

Q: Where does Carol work?
A: Google
---
"""
        elif question_type in ("temporal-reasoning", "multi-session"):
            few_shot = """
Examples of correct answers:
Q: What was discussed in the first session?
A: Project kickoff meeting

Q: How did the plan change over time?
A: From waterfall to agile
---
"""

        prompt = f"""Based on the conversation snippets below, answer the question.
{hint + ' ' if hint else ''}{output_constraint}

{few_shot}
Context:
{context}

Question: {question}
Answer:"""

        try:
            if self.provider == "deepseek":
                raw_answer = self._call_deepseek(prompt)
            elif self.provider == "openai":
                raw_answer = self._call_openai(prompt)
            elif self.provider == "minimax":
                raw_answer = self._call_minimax(prompt)
            elif self.provider == "glm":
                raw_answer = self._call_glm(prompt)
            else:
                raw_answer = self._call_ollama(prompt, question)
            # v3.6.0: 后处理清洗（去除常见冗余前缀/引号/截断）
            return self._postprocess_answer(raw_answer) if raw_answer else ""
        except requests.Timeout:
            logger.warning(
                "[LLMReranker] timeout (%.0fs) for question: %s",
                self.timeout,
                question[:80],
            )
        except requests.ConnectionError:
            logger.warning(
                "[LLMReranker] connection refused at %s",
                self.ollama_url if self.provider == "ollama" else (
                    self._minimax_base_url if self.provider == "minimax"
                    else self._glm_base_url if self.provider == "glm"
                    else self._deepseek_base_url
                ),
            )
        except Exception:
            logger.exception("[LLMReranker] unexpected error")

        return ""

    def _call_ollama(self, prompt: str, question: str) -> str:
        """Call Ollama /api/generate endpoint."""
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": self.num_predict,
                },
            },
            timeout=self.timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            answer = (data.get("response") or "").strip()
            if not answer and data.get("thinking"):
                answer = _extract_from_thinking(data["thinking"])
            if answer:
                logger.debug(
                    "[LLMReranker] answer extracted: %s → %s",
                    question[:50], answer[:100],
                )
            return answer
        else:
            logger.warning("[LLMReranker] Ollama error %d: %s", resp.status_code, resp.text[:200])
        return ""

    def _call_deepseek(self, prompt: str) -> str:
        """Call DeepSeek /v1/chat/completions (OpenAI-compatible) endpoint."""
        if not self._deepseek_api_key:
            logger.warning("[LLMReranker] DeepSeek API key not set")
            return ""
        resp = requests.post(
            f"{self._deepseek_base_url}/chat/completions",
            json={
                "model": self.model or "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": self.num_predict,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._deepseek_api_key}",
            },
            timeout=self.timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                answer = (choices[0].get("message", {}).get("content", "") or "").strip()
                if answer:
                    logger.debug("[LLMReranker] DeepSeek answer: %s", answer[:100])
                return answer
        else:
            logger.warning("[LLMReranker] DeepSeek error %d: %s", resp.status_code, resp.text[:200])
        return ""

    def _call_openai(self, prompt: str) -> str:
        """v4.1: Call OpenAI /v1/chat/completions endpoint."""
        if not self._openai_api_key:
            logger.warning("[LLMReranker] OpenAI API key not set")
            return ""
        resp = requests.post(
            f"{self._openai_base_url}/chat/completions",
            json={
                "model": self.model or "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": self.num_predict,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._openai_api_key}",
            },
            timeout=self.timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                answer = (choices[0].get("message", {}).get("content", "") or "").strip()
                if answer:
                    logger.debug("[LLMReranker] OpenAI answer: %s", answer[:100])
                return answer
        else:
            logger.warning("[LLMReranker] OpenAI error %d: %s", resp.status_code, resp.text[:200])
        return ""

    def _call_minimax(self, prompt: str) -> str:
        """v4.4.0: Call MiniMax /text/chatcompletion_v2 endpoint."""
        if not self._minimax_api_key:
            logger.warning("[LLMReranker] MiniMax API key not set")
            return ""
        resp = requests.post(
            f"{self._minimax_base_url}/text/chatcompletion_v2",
            json={
                "model": self.model or "abab6.5s-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.01,
                "tokens_to_generate": self.num_predict,
                "stream": False,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._minimax_api_key}",
            },
            timeout=self.timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            # MiniMax v2 API returns OpenAI-compatible: choices[0].message.content
            choices = data.get("choices", [])
            if choices:
                reply = (choices[0].get("message", {}).get("content", "") or "").strip()
                if reply:
                    logger.debug("[LLMReranker] MiniMax answer: %s", reply[:100])
                return reply
            return ""
        else:
            logger.warning("[LLMReranker] MiniMax error %d: %s", resp.status_code, resp.text[:200])
        return ""

    def _call_glm(self, prompt: str) -> str:
        """v4.4.0: Call GLM (Zhipu) /chat/completions (OpenAI-compatible) endpoint."""
        if not self._glm_api_key:
            logger.warning("[LLMReranker] GLM/Zhipu API key not set")
            return ""
        resp = requests.post(
            f"{self._glm_base_url}/chat/completions",
            json={
                "model": self.model or "glm-4-flash",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.01,
                "max_tokens": self.num_predict,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._glm_api_key}",
            },
            timeout=self.timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                answer = (choices[0].get("message", {}).get("content", "") or "").strip()
                if answer:
                    logger.debug("[LLMReranker] GLM answer: %s", answer[:100])
                return answer
        else:
            logger.warning("[LLMReranker] GLM error %d: %s", resp.status_code, resp.text[:200])
        return ""

    def _resolve_auto_provider(self) -> None:
        """v4.4.0: auto 模式下按优先级自动检测可用 provider。"""
        # 优先级: deepseek > minimax > glm > openai > ollama
        if os.environ.get("DEEPSEEK_API_KEY"):
            self.provider = "deepseek"
            self.model = self.model or "deepseek-chat"
            logger.info("[LLMReranker] auto → deepseek")
        elif os.environ.get("MINIMAX_API_KEY"):
            self.provider = "minimax"
            self.model = self.model or "abab6.5s-chat"
            logger.info("[LLMReranker] auto → minimax")
        elif os.environ.get("GLM_API_KEY") or os.environ.get("ZHIPU_API_KEY"):
            self.provider = "glm"
            self.model = self.model or "glm-4-flash"
            logger.info("[LLMReranker] auto → glm")
        elif os.environ.get("OPENAI_API_KEY"):
            self.provider = "openai"
            self.model = self.model or "gpt-4o-mini"
            logger.info("[LLMReranker] auto → openai")
        elif check_ollama_available(self.ollama_url):
            self.provider = "ollama"
            logger.info("[LLMReranker] auto → ollama")

    def answer_batch(
        self,
        questions: list[str],
        all_contexts: list[list[str]],
        question_types: list[str] | None = None,
        verbose: bool = False,
    ) -> list[str]:
        """
        批量提取答案（顺序执行，每个问题一次 LLM 调用）。

        Args:
            questions: 问题列表
            all_contexts: 每个问题的上下文 chunk 列表
            question_types: 可选的问题类型列表
            verbose: 是否打印进度

        Returns:
            答案列表（与输入顺序对应）
        """
        n = len(questions)
        results: list[str] = []
        for i, (q, ctx) in enumerate(zip(questions, all_contexts)):
            qtype = (
                question_types[i]
                if question_types and i < len(question_types)
                else "unknown"
            )
            if verbose:
                logger.info("  [LLM] %d/%d %s…", i + 1, n, q[:60])
            ans = self.answer_from_context(q, ctx, qtype)
            results.append(ans)
        return results

    def _postprocess_answer(self, raw: str) -> str:
        """v3.6.0: LLM 答案后处理清洗。

        清除 V4-Pro 等模型常见的冗余输出模式，提取纯净答案。

        处理步骤:
        1. 提取引号内内容
        2. 去除常见前缀 ("The answer is", "Answer:" 等)
        3. 去除换行后的解释性语句
        4. 去除末尾句号/标点冗余
        """
        if not raw or not raw.strip():
            return ""

        text = raw.strip()

        # Step 1: 提取引号内内容（如果整个回答是引号包裹的）
        quote_match = re.match(r'^["\'](.+)["\']$', text)
        if quote_match:
            text = quote_match.group(1).strip()

        # Step 2: 去除常见前缀 (v3.6.1: 修复 (?i) 位置 — Python re 要求 flag 在表达式最前端)
        prefix_patterns = [
            r'(?i)^the\s+answer\s+is\s*[:\s-]*\s*',
            r'(?i)^answer\s*[:\s-]+\s*',
            r'(?i)^according\s+to\s+the\s+\w+\s*[,:]\s*',
            r'(?i)^based\s+on\s+the\s+\w+\s*[,:]\s*',
            r'(?i)^i\s+think\s+',
            r'(?i)^it\s+is\s+',
            r'(?i)^it\s+was\s+',
            r'(?i)^this\s+is\s+',
            r'(?i)^that\s+is\s+',
        ]
        for pat in prefix_patterns:
            text = re.sub(pat, '', text, count=1)

        # Step 3: 去除换行后的解释（取首行/首句）
        # 如果第一行是完整句子，去掉后续的冗余解释
        first_line = text.split('\n')[0].strip()
        if len(first_line) > 0 and len(first_line) >= len(text) * 0.3:
            text = first_line

        # 如果包含 "This means", "In other words" 等解释模式，截断
        explanation_patterns = [
            r'(?i)\s+this\s+means\b.*$',
            r'(?i)\s+in\s+other\s+words\b.*$',
            r'(?i)\s+that\s+is\s+to\s+say\b.*$',
            r'(?i)\s+therefore\b.*$',
            r'(?i)\s+however\b.*$',
            r'(?i)\s+note\s+that\b.*$',
        ]
        for pat in explanation_patterns:
            text = re.sub(pat, '', text)

        # Step 4: 去除末尾多余标点/空格
        text = text.rstrip('. \t\n')

        # Step 5: 如果清理后变空，回退到原始
        if not text.strip():
            return raw.strip()

        return text.strip()


def create_reranker(model: str = DEFAULT_MODEL, provider: str = "ollama", **kwargs: Any) -> LLMReranker:
    """工厂函数：创建 LLMReranker 实例。v3.5.9: 支持 provider 参数"""
    return LLMReranker(model=model, provider=provider, **kwargs)


# v4.3.1: 别名 — benchmark 脚本导入使用 create_llm_reranker
create_llm_reranker = create_reranker


def _extract_from_thinking(thinking: str) -> str:
    """从思考模型 (qwen3.5) 的 thinking 字段提取最终答案。"""
    # 尝试匹配 <answer> 标签
    import re
    m = re.search(r'<answer>(.*?)</answer>', thinking, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 尝试匹配 "Answer:" 后的内容
    m = re.search(r'Answer:\s*(.+?)(?:\n|$)', thinking, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 回退: 取最后一行非空内容
    lines = [l.strip() for l in thinking.split('\n') if l.strip()]
    if lines:
        return lines[-1]
    return ""


# 便捷检查
def check_ollama_available(url: str = "http://localhost:11434") -> bool:
    """检查 Ollama 服务是否可用。"""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def check_deepseek_available() -> bool:
    """检查 DeepSeek API 是否可用。"""
    return bool(os.environ.get("DEEPSEEK_API_KEY", ""))


def check_openai_available() -> bool:
    """v4.1: 检查 OpenAI API 是否可用。"""
    return bool(os.environ.get("OPENAI_API_KEY", ""))


def check_minimax_available() -> bool:
    """v4.4.0: 检查 MiniMax API 是否可用。"""
    return bool(os.environ.get("MINIMAX_API_KEY", ""))


def check_glm_available() -> bool:
    """v4.4.0: 检查 GLM (智谱) API 是否可用。"""
    return bool(os.environ.get("GLM_API_KEY", os.environ.get("ZHIPU_API_KEY", "")))
