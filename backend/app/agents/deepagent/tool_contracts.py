"""
DeepAgent tool contracts — unified protocol for deterministic financial tools.

Every tool exposed to a DeepAgent MUST implement this contract so the agent
(and its LLM orchestrator) can discover, describe, and invoke the tool safely.

Key principles:
  - Deterministic tools own financial decisions (risk level, allocation, compliance).
  - The DeepAgent orchestrator can READ tool outputs but MUST NOT alter them.
  - Each tool declares hard_constraints that the orchestrator must respect.
  - All tools carry a Chinese business name for reporting and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class DeepAgentTool:
    """Unified contract for a tool usable by a DeepAgent orchestrator.

    Attributes:
        tool_id: Unique identifier (snake_case), e.g. "profile_analyzer".
        name_cn: Chinese business name, e.g. "用户画像解析器".
        description: Human-readable description for the LLM orchestrator.
        hard_constraints: List of rules the orchestrator MUST NOT violate.
        execute: Callable that performs the actual work.
                 Signature depends on the tool but MUST accept keyword arguments
                 and return a dict or structured result.
        input_schema: Optional description of expected inputs (for docs/logging).
        output_schema: Optional description of expected outputs (for docs/logging).
        requires_llm: Whether this tool internally calls an LLM.
    """

    tool_id: str
    name_cn: str
    description: str
    hard_constraints: list[str] = field(default_factory=list)
    execute: Callable[..., Any] | None = None
    input_schema: dict[str, str] | None = None
    output_schema: dict[str, str] | None = None
    requires_llm: bool = False

    def __call__(self, **kwargs: Any) -> Any:
        """Invoke the tool's execute function."""
        if self.execute is None:
            raise NotImplementedError(
                f"Tool '{self.tool_id}' has no execute function bound."
            )
        return self.execute(**kwargs)

    def get_system_prompt_fragment(self) -> str:
        """Generate a compact prompt fragment describing this tool."""
        lines = [
            f"- {self.tool_id}（{self.name_cn}）: {self.description}",
        ]
        if self.hard_constraints:
            lines.append("  硬约束：")
            for c in self.hard_constraints:
                lines.append(f"    • {c}")
        return "\n".join(lines)
