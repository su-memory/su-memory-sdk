"""
su-memory v4.0 — Structured Event Extractor (结构化事件提取引擎)

从对话文本中提取自包含的叙事事实(narrative fact)，替代原始chunk检索。

核心能力:
- 共指消解: "她" → 具体人名
- 时间归一化: "上周" → ISO 8601 日期范围
- 叙事式提取: 保留完整推理链，非碎片式
- 词汇别名: 2-4个同义改写，提升关键词召回
- 实体提取: 人名/地名/组织名

参考: Chronos (95.6%), Hindsight (91.4%), Supermemory (95%)
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class TemporalRange:
    """ISO 8601 日期范围"""
    start: str   # ISO 8601 日期字符串，如 "2024-03-01"
    end: str     # ISO 8601 日期字符串，如 "2024-03-31"
    granularity: str = "day"  # day / week / month / year


@dataclass
class ExtractedFact:
    """结构化叙事事实 — 对话中提取的自包含知识单元"""
    fact_id: str                # 唯一ID
    narrative: str              # 自包含叙事文本（消解指代）
    fact_type: str              # world / experience / opinion / observation
    event_date_start: str | None = None   # ISO 8601 事件开始时间
    event_date_end: str | None = None     # ISO 8601 事件结束时间
    mention_date: str | None = None       # ISO 8601 提及时间
    entities: list[str] = field(default_factory=list)   # 提及的实体
    subject: str | None = None            # 事件主体
    verb: str | None = None               # 事件动词
    object: str | None = None             # 事件客体
    lexical_aliases: list[str] = field(default_factory=list)  # 同义改写
    source_chunk_id: str = ""             # 来源chunk ID
    source_session_id: str = ""           # 来源session ID
    confidence: float = 1.0               # 置信度
    preference_signal: str | None = None  # preference / anti_preference / None

    def to_metadata(self) -> dict[str, Any]:
        """转换为 memory.add() 的 metadata 字典"""
        return {
            "is_extracted_fact": True,
            "fact_type": self.fact_type,
            "event_date_start": self.event_date_start or "",
            "event_date_end": self.event_date_end or "",
            "mention_date": self.mention_date or "",
            "entities": json.dumps(self.entities, ensure_ascii=False),
            "lexical_aliases": json.dumps(self.lexical_aliases, ensure_ascii=False),
            "source_session_id": self.source_session_id,
            "source_chunk_id": self.source_chunk_id,
            "subject": self.subject or "",
            "verb": self.verb or "",
            "object": self.object or "",
            "confidence": self.confidence,
            "preference_signal": self.preference_signal or "",
        }

    def to_searchable_text(self) -> str:
        """生成用于向量化和检索的文本（叙事 + 别名 + 实体 + 时间标签）"""
        parts = [self.narrative]
        if self.lexical_aliases:
            parts.append(" | aliases: " + "; ".join(self.lexical_aliases))
        if self.entities:
            parts.append(" | entities: " + ", ".join(self.entities))
        if self.event_date_start:
            time_ref = f" | event_time: {self.event_date_start}"
            if self.event_date_end and self.event_date_end != self.event_date_start:
                time_ref += f" to {self.event_date_end}"
            parts.append(time_ref)
        if self.preference_signal:
            parts.append(f" | preference: {self.preference_signal}")
        return "".join(parts)


# ---------------------------------------------------------------------------
# 提取模式
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT_FULL = """\
You are a precise fact extractor for a memory system. Extract structured narrative facts from the conversation below.

RULES:
1. Each fact must be SELF-CONTAINED — resolve all pronouns to specific names (e.g., "she" → "Alice").
2. Use NARRATIVE style — combine related exchanges into coherent facts that preserve reasoning chains.
3. Extract 2-5 facts per conversation, covering the most important information.
4. For EACH fact, provide:
   - narrative: The self-contained fact text
   - fact_type: One of "world" (objective fact), "experience" (event/action), "opinion" (subjective view), "observation" (summary)
   - entities: List of person names, places, organizations mentioned
   - event_date_start/end: ISO 8601 date range if a time is mentioned (e.g., "2024-03-15"), null if not
   - subject/verb/object: The core event structure (e.g., "Alice"/"moved to"/"Boston")
   - lexical_aliases: 2-4 alternative phrasings using different vocabulary (for keyword search recall)
   - preference_signal: "preference" if this reveals a user preference/like, "anti_preference" if dislike, null otherwise

CONVERSATION:
{conversation}

Session date: {session_date}
Session ID: {session_id}

OUTPUT FORMAT — Return a JSON array of facts:
[
  {{
    "narrative": "Alice moved to Boston in March 2024 for a new job at TechCorp.",
    "fact_type": "experience",
    "entities": ["Alice", "Boston", "TechCorp"],
    "event_date_start": "2024-03-01",
    "event_date_end": "2024-03-31",
    "subject": "Alice",
    "verb": "moved to",
    "object": "Boston",
    "lexical_aliases": ["Alice relocated to Boston", "Alice settled in Boston area", "Alice's new home is Boston"],
    "preference_signal": null
  }}
]

Return ONLY the JSON array, no other text."""

EXTRACTION_PROMPT_LIGHT = """\
Extract key facts from this conversation. Resolve all pronouns to specific names. For each fact, provide:
- narrative: self-contained fact text (resolve pronouns!)
- entities: names mentioned
- event_date_start/end: ISO date if mentioned, null otherwise
- fact_type: "world", "experience", "opinion", or "observation"
- preference_signal: "preference"/"anti_preference"/null

CONVERSATION:
{conversation}

Session date: {session_date}

Return a JSON array of facts:
[{{"narrative": "...", "fact_type": "...", "entities": [...], "event_date_start": null, "event_date_end": null, "preference_signal": null}}]

Return ONLY the JSON array."""


# ---------------------------------------------------------------------------
# 规则回退提取（无LLM时使用）
# ---------------------------------------------------------------------------

# 常见时间表达式正则
_TIME_PATTERNS = [
    # 绝对日期
    (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s*(\d{4})\b', 'absolute'),
    (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s*(\d{4})\b', 'absolute'),
    (r'\b(\d{4})-(\d{2})-(\d{2})\b', 'absolute'),
    # 相对日期
    (r'\b(last|past|previous)\s+(week|month|year|weekend)\b', 'relative'),
    (r'\b(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b', 'relative'),
    (r'\brecently\b', 'relative'),
    (r'\byesterday\b', 'relative'),
    (r'\btomorrow\b', 'relative'),
    # 序列时间
    (r'\b(first|earliest|initial)\b', 'ordinal'),
    (r'\b(latest|most recent|last|current)\b', 'ordinal'),
    (r'\b(before|after|prior to|following)\b', 'ordinal'),
]

# 人名提取正则（英文大写开头连续词）
_PERSON_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
# 单大写词（但排除常见非人名词）
_STOP_WORDS = frozenset({
    'the', 'and', 'but', 'for', 'with', 'that', 'this', 'from', 'have',
    'been', 'was', 'are', 'not', 'can', 'you', 'your', 'all', 'has',
    'had', 'its', 'they', 'them', 'their', 'there', 'then', 'than',
    'when', 'what', 'which', 'who', 'how', 'why', 'where', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
    'some', 'such', 'very', 'just', 'also', 'into', 'over', 'only',
    'about', 'after', 'before', 'between', 'both', 'each', 'every',
    'more', 'most', 'other', 'same', 'still', 'through', 'under',
    'until', 'upon', 'while', 'because', 'although', 'however',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'june', 'july',
    'august', 'september', 'october', 'november', 'december',
})


def _extract_facts_rule_based(
    conversation_text: str,
    session_id: str,
    session_date: str | None = None,
) -> list[ExtractedFact]:
    """规则回退提取 — 无LLM时使用正则和启发式方法"""
    facts: list[ExtractedFact] = []

    # 按对话轮次分割
    lines = conversation_text.strip().split('\n')
    current_speaker = None
    buffer_lines: list[str] = []
    chunk_id = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 检测角色标签 [user] 或 [assistant]
        role_match = re.match(r'^\[(user|assistant)\]\s*(.*)', line)
        if role_match:
            # 先处理之前缓冲的内容
            if buffer_lines:
                text = ' '.join(buffer_lines)
                fact = _make_rule_fact(text, current_speaker, session_id, chunk_id, session_date)
                if fact:
                    facts.append(fact)
                    chunk_id += 1
                buffer_lines = []
            current_speaker = role_match.group(1)
            content = role_match.group(2).strip()
            if content:
                buffer_lines.append(content)
        else:
            buffer_lines.append(line)

    # 处理最后缓冲
    if buffer_lines:
        text = ' '.join(buffer_lines)
        fact = _make_rule_fact(text, current_speaker, session_id, chunk_id, session_date)
        if fact:
            facts.append(fact)

    # 合并短事实（相邻的同一角色轮次）
    if len(facts) > 1:
        merged: list[ExtractedFact] = []
        i = 0
        while i < len(facts):
            if i + 1 < len(facts) and len(facts[i].narrative) < 50 and len(facts[i + 1].narrative) < 80:
                # 合并
                combined = facts[i].narrative + " " + facts[i + 1].narrative
                entities = list(set(facts[i].entities + facts[i + 1].entities))
                merged.append(ExtractedFact(
                    fact_id=facts[i].fact_id,
                    narrative=combined,
                    fact_type=facts[i].fact_type,
                    entities=entities,
                    source_session_id=session_id,
                    source_chunk_id=f"merged_{facts[i].source_chunk_id}",
                    confidence=0.6,
                ))
                i += 2
            else:
                merged.append(facts[i])
                i += 1
        facts = merged

    return facts


def _make_rule_fact(
    text: str,
    speaker: str | None,
    session_id: str,
    chunk_id: int,
    session_date: str | None,
) -> ExtractedFact | None:
    """从单段文本创建规则提取的事实"""
    if len(text.strip()) < 10:
        return None

    entities: list[str] = []
    # 提取人名
    for m in _PERSON_PATTERN.finditer(text):
        name = m.group(1)
        if name.lower() not in _STOP_WORDS:
            entities.append(name)

    # 提取时间表达式
    event_date_start = None
    event_date_end = None
    for pat, ptype in _TIME_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            # 简化：仅标记有无时间表达式，精确解析留给TemporalParser
            event_date_start = session_date  # fallback to session date
            break

    # 检测preference信号
    pref_signal = None
    pref_patterns = [
        (r'\b(?:i\s+)?(?:love|like|enjoy|prefer|favorite|favourite)\b', 'preference'),
        (r'\b(?:i\s+)?(?:hate|dislike|don\'t\s+like|can\'t\s+stand|avoid)\b', 'anti_preference'),
    ]
    for pat, sig in pref_patterns:
        if re.search(pat, text, re.IGNORECASE):
            pref_signal = sig
            break

    fact_type = "experience" if speaker == "user" else "observation"
    if pref_signal:
        fact_type = "opinion"

    return ExtractedFact(
        fact_id=str(uuid.uuid4())[:12],
        narrative=text.strip(),
        fact_type=fact_type,
        event_date_start=event_date_start,
        event_date_end=event_date_end,
        mention_date=session_date,
        entities=entities,
        source_session_id=session_id,
        source_chunk_id=f"rule_{chunk_id}",
        confidence=0.6,
        preference_signal=pref_signal,
    )


# ---------------------------------------------------------------------------
# EventExtractor 主类
# ---------------------------------------------------------------------------

class EventExtractor:
    """结构化事件提取器 — 从对话中提取自包含的叙事事实

    支持三种模式:
    - Full: 完整LLM提取（叙事+时间归一化+实体+别名+preference）
    - Light: 简化LLM提取（叙事+实体+时间）
    - Rule: 无LLM时规则回退（正则+启发式）
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "gemma4",
        ollama_url: str = "http://localhost:11434",
        mode: str = "full",   # full / light / rule
        timeout: int = 60,
        batch_size: int = 25,  # 每批次最大对话轮数
        deepseek_api_key: str = "",
        deepseek_base_url: str = "https://api.deepseek.com/v1",
    ):
        self.provider = provider
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.mode = mode
        self.timeout = timeout
        self.batch_size = batch_size
        self._deepseek_api_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._deepseek_base_url = deepseek_base_url.rstrip("/")
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

    def extract_facts(
        self,
        session_turns: list[dict[str, Any]],
        session_id: str,
        session_date: str | None = None,
    ) -> list[ExtractedFact]:
        """从session对话中提取结构化事实

        Args:
            session_turns: 对话轮次列表，每个包含 role/content
            session_id: Session ID
            session_date: Session日期 (ISO 8601)

        Returns:
            提取的事实列表
        """
        if not session_turns:
            return []

        if self.mode == "rule":
            conversation = self._format_turns(session_turns)
            return _extract_facts_rule_based(conversation, session_id, session_date)

        # LLM模式：按批次提取
        all_facts: list[ExtractedFact] = []
        batches = self._make_batches(session_turns)

        for batch_idx, batch in enumerate(batches):
            conversation = self._format_turns(batch)
            if len(conversation.strip()) < 20:
                continue

            facts = self._extract_batch(
                conversation, session_id, session_date, batch_idx
            )
            all_facts.extend(facts)

        return all_facts

    def _make_batches(
        self, turns: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """将对话分成批次，每批最多batch_size轮，重叠5轮"""
        if len(turns) <= self.batch_size:
            return [turns]

        batches = []
        overlap = 5
        start = 0
        while start < len(turns):
            end = min(start + self.batch_size, len(turns))
            batches.append(turns[start:end])
            if end >= len(turns):
                break
            start = end - overlap
        return batches

    def _format_turns(self, turns: list[dict[str, Any]]) -> str:
        """格式化对话轮次为文本"""
        lines = []
        for turn in turns:
            role = str(turn.get("role", "user")).strip() or "user"
            content = str(turn.get("content", "")).strip()
            if content:
                lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    def _extract_batch(
        self,
        conversation: str,
        session_id: str,
        session_date: str | None,
        batch_idx: int,
    ) -> list[ExtractedFact]:
        """从一批对话中提取事实 — v4.1: 增加重试机制"""
        prompt_template = EXTRACTION_PROMPT_FULL if self.mode == "full" else EXTRACTION_PROMPT_LIGHT
        prompt = prompt_template.format(
            conversation=conversation,
            session_date=session_date or "unknown",
            session_id=session_id,
        )

        try:
            if self.provider == "deepseek":
                raw_response = self._call_deepseek(prompt)
            elif self.provider == "openai":
                raw_response = self._call_openai(prompt)
            elif self.provider == "minimax":
                raw_response = self._call_minimax(prompt)
            elif self.provider == "glm":
                raw_response = self._call_glm(prompt)
            else:
                raw_response = self._call_ollama(prompt)
        except Exception as exc:
            logger.warning("[EventExtractor] LLM call failed: %s, falling back to rules", exc)
            return _extract_facts_rule_based(conversation, session_id, session_date)

        if not raw_response:
            return _extract_facts_rule_based(conversation, session_id, session_date)

        facts = self._parse_facts(raw_response, session_id, session_date, batch_idx)

        # v4.1: 如果解析结果为空，重试一次（追加格式指令）
        if not facts and raw_response:
            logger.info("[EventExtractor] First attempt produced no facts, retrying with format hint")
            retry_prompt = (
                "Your previous response could not be parsed as JSON. "
                "Please try again and return ONLY a valid JSON array. "
                "Example format:\n"
                '[{"narrative": "...", "fact_type": "experience", "entities": [...], '
                '"event_date_start": null, "preference_signal": null}]\n\n'
                f"Original question: Extract facts from this conversation.\n\n"
                f"CONVERSATION:\n{conversation[:2000]}"
            )
            try:
                if self.provider == "deepseek":
                    retry_response = self._call_deepseek(retry_prompt)
                elif self.provider == "openai":
                    retry_response = self._call_openai(retry_prompt)
                elif self.provider == "minimax":
                    retry_response = self._call_minimax(retry_prompt)
                elif self.provider == "glm":
                    retry_response = self._call_glm(retry_prompt)
                else:
                    retry_response = self._call_ollama(retry_prompt)
                if retry_response:
                    facts = self._parse_facts(retry_response, session_id, session_date, batch_idx)
            except Exception as exc:
                logger.debug("[EventExtractor] Retry failed: %s", exc)

        # 最终兜底：如果仍然没有提取到事实，使用规则回退
        if not facts:
            logger.info("[EventExtractor] LLM extraction produced no facts, using rule fallback")
            return _extract_facts_rule_based(conversation, session_id, session_date)

        return facts

    def _call_ollama(self, prompt: str) -> str:
        """调用Ollama API"""
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 2000,
                },
            },
            timeout=self.timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            answer = (data.get("response") or "").strip()
            # 思考模型回退
            if not answer and data.get("thinking"):
                answer = _extract_from_thinking(data["thinking"])
            return answer
        else:
            logger.warning("[EventExtractor] Ollama error %d: %s", resp.status_code, resp.text[:200])
        return ""

    def _call_deepseek(self, prompt: str) -> str:
        """调用DeepSeek API"""
        if not self._deepseek_api_key:
            logger.warning("[EventExtractor] DeepSeek API key not set")
            return ""
        resp = requests.post(
            f"{self._deepseek_base_url}/chat/completions",
            json={
                "model": self.model or "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 2000,
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
                return (choices[0].get("message", {}).get("content", "") or "").strip()
        else:
            logger.warning("[EventExtractor] DeepSeek error %d: %s", resp.status_code, resp.text[:200])
        return ""

    def _call_openai(self, prompt: str) -> str:
        """v4.1: 调用OpenAI API"""
        if not self._openai_api_key:
            logger.warning("[EventExtractor] OpenAI API key not set")
            return ""
        resp = requests.post(
            f"{self._openai_base_url}/chat/completions",
            json={
                "model": self.model or "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 2000,
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
                return (choices[0].get("message", {}).get("content", "") or "").strip()
        else:
            logger.warning("[EventExtractor] OpenAI error %d: %s", resp.status_code, resp.text[:200])
        return ""

    def _call_minimax(self, prompt: str) -> str:
        """v4.4.0: 调用MiniMax chat API"""
        if not self._minimax_api_key:
            logger.warning("[EventExtractor] MiniMax API key not set")
            return ""
        resp = requests.post(
            f"{self._minimax_base_url}/text/chatcompletion_v2",
            json={
                "model": self.model or "abab6.5s-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.01,
                "tokens_to_generate": 2000,
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
                return (choices[0].get("message", {}).get("content", "") or "").strip()
            return ""
        else:
            logger.warning("[EventExtractor] MiniMax error %d: %s", resp.status_code, resp.text[:200])
        return ""

    def _call_glm(self, prompt: str) -> str:
        """v4.4.0: 调用GLM (智谱) API"""
        if not self._glm_api_key:
            logger.warning("[EventExtractor] GLM/Zhipu API key not set")
            return ""
        resp = requests.post(
            f"{self._glm_base_url}/chat/completions",
            json={
                "model": self.model or "glm-4-flash",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.01,
                "max_tokens": 2000,
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
                return (choices[0].get("message", {}).get("content", "") or "").strip()
        else:
            logger.warning("[EventExtractor] GLM error %d: %s", resp.status_code, resp.text[:200])
        return ""

    def _parse_facts(
        self,
        raw_response: str,
        session_id: str,
        session_date: str | None,
        batch_idx: int,
    ) -> list[ExtractedFact]:
        """解析LLM返回的JSON为ExtractedFact列表 — v4.1: 多策略鲁棒解析"""
        # 尝试提取JSON数组
        json_text = self._extract_json(raw_response)
        if json_text:
            items = self._robust_json_parse(json_text)
            if items is not None:
                return self._build_facts_from_items(items, session_id, session_date, batch_idx)

        # v4.1: JSON 解析完全失败 → regex key-value 兜底
        logger.warning("[EventExtractor] JSON parse failed, trying regex fallback")
        regex_facts = self._extract_facts_regex(raw_response, session_id, session_date, batch_idx)
        if regex_facts:
            return regex_facts

        logger.warning("[EventExtractor] All extraction strategies failed, returning empty")
        return []

    def _robust_json_parse(self, json_text: str) -> list[dict] | None:
        """v4.1: 多策略JSON解析 — 尝试多种修复方式"""
        strategies = [
            self._parse_json_direct,
            self._parse_json_fix_trailing_comma,
            self._parse_json_fix_escapes,
            self._parse_json_fix_brackets,
        ]
        for strategy in strategies:
            try:
                result = strategy(json_text)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def _parse_json_direct(self, text: str) -> list[dict] | None:
        """直接解析JSON"""
        items = json.loads(text)
        if not isinstance(items, list):
            items = [items]
        return items

    def _parse_json_fix_trailing_comma(self, text: str) -> list[dict] | None:
        """修复尾部逗号"""
        fixed = re.sub(r',\s*}', '}', text)
        fixed = re.sub(r',\s*]', ']', fixed)
        items = json.loads(fixed)
        if not isinstance(items, list):
            items = [items]
        return items

    def _parse_json_fix_escapes(self, text: str) -> list[dict] | None:
        """修复转义字符问题（未转义的换行/制表符等）"""
        # 替换字符串值中的未转义换行符
        fixed = re.sub(r'(?<=[\"\'])\n(?=[\"\'])', '\\n', text)
        fixed = re.sub(r'(?<=:)\s*\n\s*', ' ', fixed)
        items = json.loads(fixed)
        if not isinstance(items, list):
            items = [items]
        return items

    def _parse_json_fix_brackets(self, text: str) -> list[dict] | None:
        """修复括号不匹配 — 补全缺失的关闭括号"""
        # 计算开闭括号数量
        open_brackets = text.count('[') + text.count('{')
        close_brackets = text.count(']') + text.count('}')
        diff = open_brackets - close_brackets
        if diff > 0:
            # 补全关闭括号
            fixed = text + ']' * diff
            items = json.loads(fixed)
            if not isinstance(items, list):
                items = [items]
            return items
        return None

    def _extract_facts_regex(
        self,
        raw_response: str,
        session_id: str,
        session_date: str | None,
        batch_idx: int,
    ) -> list[ExtractedFact]:
        """v4.1: Regex key-value 兜底提取 — 从非结构化LLM输出中提取关键字段"""
        facts: list[ExtractedFact] = []

        # 提取 narrative 字段
        narrative_patterns = [
            re.compile(r'"narrative"\s*:\s*"(.*?)"', re.DOTALL),
            re.compile(r'narrative\s*[:=]\s*"(.*?)"', re.DOTALL),
            re.compile(r'Narrative\s*:\s*(.+?)(?:\n|$)'),
        ]

        narratives: list[str] = []
        for pat in narrative_patterns:
            for m in pat.finditer(raw_response):
                n = m.group(1).strip()
                if n and len(n) > 5:
                    narratives.append(n)

        for i, narrative in enumerate(narratives):
            # 提取实体
            entities: list[str] = []
            for m in _PERSON_PATTERN.finditer(narrative):
                name = m.group(1)
                if name.lower() not in _STOP_WORDS:
                    entities.append(name)

            # 检测 preference
            pref_signal = None
            for pat, sig in [(r'\b(?:love|like|enjoy|prefer)\b', 'preference'),
                             (r'\b(?:hate|dislike|don\'t like)\b', 'anti_preference')]:
                if re.search(pat, narrative, re.IGNORECASE):
                    pref_signal = sig
                    break

            facts.append(ExtractedFact(
                fact_id=str(uuid.uuid4())[:12],
                narrative=narrative,
                fact_type="observation",
                event_date_start=session_date,
                mention_date=session_date,
                entities=entities,
                source_session_id=session_id,
                source_chunk_id=f"regex_batch{batch_idx}_fact{i}",
                confidence=0.5,
                preference_signal=pref_signal,
            ))

        return facts

    def _build_facts_from_items(
        self,
        items: list,
        session_id: str,
        session_date: str | None,
        batch_idx: int,
    ) -> list[ExtractedFact]:
        """v4.1: 从解析成功的 JSON items 构建 ExtractedFact 列表"""
        if not isinstance(items, list):
            items = [items]

        facts: list[ExtractedFact] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            try:
                fact = ExtractedFact(
                    fact_id=str(uuid.uuid4())[:12],
                    narrative=str(item.get("narrative", "")).strip(),
                    fact_type=str(item.get("fact_type", "observation")).strip().lower(),
                    event_date_start=item.get("event_date_start"),
                    event_date_end=item.get("event_date_end"),
                    mention_date=session_date,
                    entities=item.get("entities", []) or [],
                    subject=item.get("subject"),
                    verb=item.get("verb"),
                    object=item.get("object"),
                    lexical_aliases=item.get("lexical_aliases", []) or [],
                    source_session_id=session_id,
                    source_chunk_id=f"batch{batch_idx}_fact{i}",
                    confidence=1.0,
                    preference_signal=item.get("preference_signal"),
                )
                # 验证必要字段
                if fact.narrative and len(fact.narrative) > 5:
                    if fact.fact_type not in ("world", "experience", "opinion", "observation"):
                        fact.fact_type = "observation"
                    if fact.preference_signal and fact.preference_signal not in ("preference", "anti_preference"):
                        fact.preference_signal = None
                    facts.append(fact)
            except Exception as exc:
                logger.warning("[EventExtractor] Failed to parse fact %d: %s", i, exc)
                continue

        return facts

    def _extract_json(self, text: str) -> str | None:
        """从LLM响应中提取JSON数组 — v4.1: 多策略鲁棒提取"""
        text = text.strip()

        # 策略1: 直接解析（响应本身就是 JSON 数组）
        if text.startswith("[") and text.endswith("]"):
            return text

        # 策略2: Markdown 代码块 ```json ... ```
        m = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
        if m:
            return m.group(1)

        # 策略3: 查找最外层数组
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return text[start:end + 1]

        # 策略4: v4.1 — 查找 JSON 对象 (非数组)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj_text = text[start:end + 1]
            # 包装为数组
            return f"[{obj_text}]"

        return None


def _extract_from_thinking(thinking: str) -> str:
    """从思考模型的thinking字段提取最终答案"""
    m = re.search(r'<answer>(.*?)</answer>', thinking, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'Answer:\s*(.+?)(?:\n|$)', thinking, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in thinking.split('\n') if l.strip()]
    if lines:
        return lines[-1]
    return ""


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def create_event_extractor(
    provider: str = "ollama",
    model: str = "gemma4",
    mode: str = "full",
    **kwargs: Any,
) -> EventExtractor:
    """工厂函数：创建EventExtractor实例"""
    return EventExtractor(provider=provider, model=model, mode=mode, **kwargs)
