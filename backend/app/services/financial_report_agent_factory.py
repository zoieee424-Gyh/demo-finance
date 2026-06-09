"""
Financial Report agent factory with process-level singleton caching.

Provides:
  - get_financial_report_agent(mode) → cached agent instance
  - reset_financial_report_agent_cache() → clear cache (for tests)
  - warmup_financial_report_agent(mode) → eager-init

Design:
  - Each mode string maps to one cached agent instance.
  - "pipeline" mode NEVER imports or initialises DeepAgent.
  - "deepagent"/"auto" modes create FinancialReportDeepAgent once.
  - Tool traces are reset per-request inside DeepAgentWrapper.run().
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Process-level cache: mode → agent instance
_FINANCIAL_REPORT_CACHE: dict[str, Any] = {}


def get_financial_report_agent(mode: str = "deepagent") -> Any:
    """Return a cached financial report agent for the given mode.

    Args:
        mode: "deepagent" / "pipeline" / "auto".

    Returns:
        FinancialReportAgent (pipeline) or
        FinancialReportDeepAgent (deepagent/auto).
    """
    mode = mode.strip().lower()
    if mode not in ("deepagent", "pipeline", "auto"):
        logger.warning("Unknown financial_report mode '%s', falling back to deepagent", mode)
        mode = "deepagent"

    if mode in _FINANCIAL_REPORT_CACHE:
        return _FINANCIAL_REPORT_CACHE[mode]

    agent = _create_agent(mode)
    _FINANCIAL_REPORT_CACHE[mode] = agent
    return agent


def _create_agent(mode: str) -> Any:
    """Build a new agent instance. NOT cached — use get_financial_report_agent."""
    from app.agents.financial_report import FinancialReportAgent

    if mode == "pipeline":
        logger.info("FinancialReportAgentFactory: creating pipeline agent")
        return FinancialReportAgent()

    # deepagent / auto
    logger.info("FinancialReportAgentFactory: creating DeepAgent financial report agent (mode=%s)", mode)
    from app.agents.deepagent.financial_report_deepagent import FinancialReportDeepAgent

    return FinancialReportDeepAgent(
        enabled=True,
        fallback_agent=FinancialReportAgent(),
        configured_mode=mode,
    )


def reset_financial_report_agent_cache() -> None:
    """Clear the agent cache. Intended for tests that mutate config."""
    _FINANCIAL_REPORT_CACHE.clear()
    logger.debug("FinancialReportAgentFactory: cache reset")


def warmup_financial_report_agent(mode: str = "deepagent") -> bool:
    """Eagerly initialise the financial report agent.

    Does NOT crash on failure — agent will lazy-init on first request.

    Returns:
        True if warmup succeeded, False otherwise.
    """
    try:
        agent = get_financial_report_agent(mode)
        if hasattr(agent, "debug_info"):
            info = agent.debug_info
            logger.info(
                "FinancialReportAgentFactory: warmup OK — arch=%s available=%s",
                info.get("agent_architecture", "?"),
                info.get("deepagent_available", False),
            )
        else:
            logger.info("FinancialReportAgentFactory: warmup OK — pipeline agent")
        return True
    except Exception as exc:
        logger.warning("FinancialReportAgentFactory: warmup failed — %s", exc)
        return False
