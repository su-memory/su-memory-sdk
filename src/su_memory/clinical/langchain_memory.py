"""
langchain_memory — LangChain Agent 临床语义记忆组件

将 su_memory.clinical.ClinicalMemoryClient 作为 LangChain Agent 的记忆组件。
与 SDK 已有的 SuMemoryChatMemory 互补：后者是通用 chat history，
本组件增加患者隔离 + 语义召回 + 临床知识注入。

Example:
  >>> from su_memory.clinical import SemanticAgentMemory
  >>> memory = SemanticAgentMemory(patient_id="P001")
  >>> memory.save_context({"input": "患者白蛋白偏低"}, {"output": "建议高蛋白饮食"})
  >>> # 下次对话时 load_memory_variables 自动语义召回相关历史
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# LangChain 可选依赖
try:
    from langchain.memory import BaseChatMemory
    from langchain_core.messages import AIMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    BaseChatMemory = object  # type: ignore
    LANGCHAIN_AVAILABLE = False


class SemanticAgentMemory(BaseChatMemory):
    """LangChain Agent 临床语义记忆组件。

    特性：
      - 患者隔离：每个 patient_id 独立记忆空间
      - 语义召回：load_memory_variables 时按当前输入语义检索相关历史
      - 临床上下文：自动附带患者检验趋势/异常值摘要
      - LangChain 兼容：实现 BaseChatMemory 接口

    Args:
        patient_id: 患者 ID（隔离维度）
        client: ClinicalMemoryClient 实例（不传则惰性创建）
        memory_key: 记忆变量名（默认 "chat_history"）
        max_semantic_results: 语义召回最大条数
        include_lab_summary: 是否附带检验趋势摘要
    """

    def __init__(
        self,
        patient_id: str,
        client: Any | None = None,
        memory_key: str = "chat_history",
        max_semantic_results: int = 5,
        include_lab_summary: bool = True,
        **kwargs: Any,
    ) -> None:
        self._patient_id = patient_id
        self._memory_key = memory_key
        self._max_results = max_semantic_results
        self._include_lab = include_lab_summary
        self._client = client
        self._session_id = f"agent_{patient_id}_{int(time.time())}"

        # 如果有 BaseChatMemory 的初始化需求
        if LANGCHAIN_AVAILABLE and hasattr(BaseChatMemory, "__init__"):
            try:
                super().__init__(**kwargs)
            except TypeError as e:
                logger.debug("BaseChatMemory 初始化降级: %s", e)
                pass

    def _ensure_client(self):
        """惰性初始化 ClinicalMemoryClient"""
        if self._client is None:
            from su_memory.clinical import ClinicalMemoryClient
            self._client = ClinicalMemoryClient(
                embedding_backend="none",
                compliance_level="mask",
            )
        return self._client

    # ── BaseChatMemory 接口实现 ───────────────────────────

    @property
    def memory_variables(self) -> list[str]:
        """返回记忆变量名列表"""
        return [self._memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """加载记忆变量 — 语义召回相关历史。

        根据当前输入语义检索患者相关记忆，
        替代简单 buffer 的全量历史返回。
        """
        client = self._ensure_client()
        query = inputs.get("input", inputs.get(self._memory_key, ""))

        # 语义召回
        hits = client.recall(
            patient_id=self._patient_id,
            query=str(query),
            top_k=self._max_results,
        )

        # 组装记忆文本
        parts: list[str] = []

        if hits:
            parts.append("## 相关历史记忆")
            for h in hits:
                content = h.get("content", "")
                event_type = (h.get("metadata") or {}).get("event_type", "")
                parts.append(f"- [{event_type}] {content[:200]}")

        # 附带检验趋势摘要
        if self._include_lab:
            lab_summary = self._build_lab_summary(client)
            if lab_summary:
                parts.append(f"\\n## 检验趋势\\n{lab_summary}")

        memory_text = "\\n".join(parts) if parts else "暂无历史记忆"
        return {self._memory_key: memory_text}

    def save_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> None:
        """保存对话上下文到语义记忆。

        将用户输入和 AI 输出都写入 su-memory，
        附带 patient_id 和 session_id 实现隔离。
        """
        client = self._ensure_client()
        user_input = inputs.get("input", inputs.get(self._memory_key, ""))
        ai_output = outputs.get("output", outputs.get(self._memory_key, ""))

        if user_input:
            client.add_patient_event(
                patient_id=self._patient_id,
                content=f"用户提问: {user_input}",
                event_type="agent_query",
                metadata={"session_id": self._session_id},
            )

        if ai_output:
            client.add_patient_event(
                patient_id=self._patient_id,
                content=f"AI回复: {ai_output}",
                event_type="agent_response",
                metadata={"session_id": self._session_id},
            )

    def clear(self) -> None:
        """清除当前会话的工作记忆（不影响持久化的语义记忆）"""
        pass  # 语义记忆持久化，不主动清除

    # ── 内部方法 ──────────────────────────────────────────

    def _build_lab_summary(self, client: Any) -> str:
        """构建检验趋势摘要"""
        try:
            abnormal = client.find_abnormal_labs(self._patient_id)
            if not abnormal:
                return ""
            parts: list[str] = []
            for lab in abnormal[:5]:
                parts.append(
                    f"- {lab['lab_name']}={lab['value']}{lab.get('unit','')}"
                    f" (参考{lab.get('reference_range','')})"
                )
            return "\\n".join(parts)
        except Exception as e:
            logger.debug("lab summary 降级为空: %s", e)
            return ""

    # ── 兼容属性 ──────────────────────────────────────────

    @property
    def return_messages(self) -> bool:
        return False  # 返回字符串而非消息对象

    @property
    def input_key(self) -> str:
        return "input"

    @property
    def output_key(self) -> str:
        return "output"
