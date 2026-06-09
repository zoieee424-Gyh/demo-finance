"""
Advisory agent factory with process-level singleton caching.

Provides:
  - get_advisory_agent(mode) → cached agent instance
  - reset_advisory_agent_cache() → clear cache (for tests)
  - warmup_advisory_agent(mode) → eager-init (for FastAPI startup)

Design:
  - Each mode string maps to one cached agent instance.
  - "pipeline" mode NEVER imports or initialises DeepAgent.
  - "deepagent"/"auto" modes create DeepAgentInvestmentAdvisor once.
  - Tool traces are reset per-request inside DeepAgentWrapper.run().
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Process-level cache: mode → agent instance
_ADVISORY_AGENT_CACHE: dict[str, Any] = {}


def get_advisory_agent(mode: str | None = None) -> Any:
    """Return a cached advisory agent for the given mode.

    Args:
        mode: "deepagent" / "pipeline" / "auto".
              If None, reads from settings.advisor_mode.

    Returns:
        InvestmentAdvisorAgent (pipeline) or
        DeepAgentInvestmentAdvisor (deepagent/auto).
    """
    if mode is None:
        from app.core.config import settings
        mode = settings.advisor_mode

    # Normalize
    mode = mode.strip().lower()
    if mode not in ("deepagent", "pipeline", "auto"):
        logger.warning("Unknown advisor_mode '%s', falling back to deepagent", mode)
        mode = "deepagent"

    if mode in _ADVISORY_AGENT_CACHE:
        return _ADVISORY_AGENT_CACHE[mode]

    agent = _create_agent(mode)
    _ADVISORY_AGENT_CACHE[mode] = agent
    return agent


def _create_agent(mode: str) -> Any:
    """Build a new agent instance. NOT cached — use get_advisory_agent."""
    from app.agents.investment_advisor import InvestmentAdvisorAgent

    if mode == "pipeline":
        logger.info("AdvisoryAgentFactory: creating pipeline agent")
        return InvestmentAdvisorAgent()

    # deepagent / auto
    logger.info("AdvisoryAgentFactory: creating DeepAgent advisor (mode=%s)", mode)
    from app.agents.deepagent import DeepAgentInvestmentAdvisor

    return DeepAgentInvestmentAdvisor(
        enabled=True,
        fallback_agent=InvestmentAdvisorAgent(),
        configured_mode=mode,
    )


def reset_advisory_agent_cache() -> None:
    """Clear the agent cache. Intended for tests that mutate config."""
    _ADVISORY_AGENT_CACHE.clear()
    logger.debug("AdvisoryAgentFactory: cache reset")


def warmup_advisory_agent(mode: str | None = None) -> bool:
    """Eagerly initialise the advisory agent.

    Catches all exceptions — failure to warm up MUST NOT crash the
    application (the agent will lazy-init on first request anyway).

    Returns:
        True if warmup succeeded, False otherwise.
    """
    try:
        agent = get_advisory_agent(mode)
        if hasattr(agent, "debug_info"):
            info = agent.debug_info
            logger.info(
                "AdvisoryAgentFactory: warmup OK — arch=%s available=%s fallback=%s",
                info.get("agent_architecture", "?"),
                info.get("deepagent_available", False),
                info.get("fallback_used", False),
            )
        else:
            logger.info("AdvisoryAgentFactory: warmup OK — pipeline agent")
        return True
    except Exception as exc:
        logger.warning("AdvisoryAgentFactory: warmup failed — %s", exc)
        return False
