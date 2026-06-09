"""
DeepAgent architecture layer for the financial services platform.

Provides:
  - DeepAgentTool: unified tool contract for deterministic financial tools
  - ToolRegistry: tool registration and discovery
  - DeepAgentWrapper: model-agnostic DeepAgent runner with fallback
  - DeepAgentInvestmentAdvisor: DeepAgent-powered investment advisor agent
  - FinancialEducationDeepAgent: DeepAgent-powered financial education agent
"""

from app.agents.deepagent.tool_contracts import DeepAgentTool
from app.agents.deepagent.registry import ToolRegistry
from app.agents.deepagent.base import DeepAgentWrapper
from app.agents.deepagent.investment_advisor_deepagent import (
    DeepAgentInvestmentAdvisor,
)
from app.agents.deepagent.education_deepagent import (
    FinancialEducationDeepAgent,
)

__all__ = [
    "DeepAgentTool",
    "ToolRegistry",
    "DeepAgentWrapper",
    "DeepAgentInvestmentAdvisor",
    "FinancialEducationDeepAgent",
]
