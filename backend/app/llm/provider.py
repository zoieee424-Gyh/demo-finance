"""
LLMProvider（云端大模型适配层）

Uses LangChain's DeepSeek integration for `deepseek-v4-flash`.
No base URL is required in this project path.

Default behavior is mock mode: no network calls are made unless the caller
explicitly sets enabled=True and mock_mode=False.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMProvider:
    """LangChain DeepSeek adapter.

    Attributes:
        model: DeepSeek model identifier.
        api_key: DeepSeek API key. Falls back to DEEPSEEK_API_KEY.
        enabled: Whether real LLM calls are allowed.
        mock_mode: Whether to force mock responses even when configured.
    """

    model: str = "deepseek-v4-flash"
    api_key: str | None = None
    enabled: bool = False
    mock_mode: bool = True

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.model or self.model == "deepseek-v4-flash":
            env_model = os.getenv("FIN_AGENT_LLM_MODEL")
            if env_model:
                self.model = env_model
        env_enabled = os.getenv("FIN_AGENT_LLM_ENABLED")
        if env_enabled is not None:
            self.enabled = env_enabled.lower() in ("1", "true", "yes", "on")
        env_mock = os.getenv("FIN_AGENT_LLM_MOCK_MODE")
        if env_mock is not None:
            self.mock_mode = env_mock.lower() in ("1", "true", "yes", "on")

    def is_configured(self) -> bool:
        """Return True when real LLM calls are allowed and credentials exist."""
        return self.enabled and bool(self.api_key)

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        """Complete chat messages through LangChain ChatDeepSeek.

        Mock mode is the default and returns a safe fallback string.
        Real calls require:
          - enabled=True
          - mock_mode=False
          - DEEPSEEK_API_KEY or api_key provided
          - langchain-deepseek installed
        """
        if not self.is_configured():
            return (
                "[LLM Mock] LangChain DeepSeek 适配层未启用或缺少 DEEPSEEK_API_KEY。"
                "当前返回 mock 响应，不调用真实 API。"
            )
        if self.mock_mode:
            return (
                "[LLM Mock] LangChain DeepSeek 适配层已配置但处于 mock 模式。"
                "设置 FIN_AGENT_LLM_MOCK_MODE=false 后才会调用真实 API。"
            )

        try:
            from langchain_deepseek import ChatDeepSeek
        except ImportError:
            return (
                "[LLM Mock] 未安装 langchain-deepseek，无法调用真实 DeepSeek。"
                "请先安装依赖：pip install langchain-deepseek"
            )

        if self.api_key:
            os.environ.setdefault("DEEPSEEK_API_KEY", self.api_key)

        llm = ChatDeepSeek(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = llm.invoke(messages)
        content = getattr(response, "content", "")
        # deepseek-v4-flash may put the real answer in reasoning_content
        # when content is empty (common with reasoning models).
        if not content or not str(content).strip():
            reasoning = getattr(response, "additional_kwargs", {}) or {}
            reasoning_text = reasoning.get("reasoning_content", "")
            if reasoning_text:
                return str(reasoning_text)
        if not content:
            return str(response)
        return str(content)


_default_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """Get or create the default LLMProvider instance."""
    global _default_provider
    if _default_provider is None:
        _default_provider = LLMProvider()
    return _default_provider
