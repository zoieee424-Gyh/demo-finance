"""
Education agent factory with process-level singleton caching.

Provides:
  - get_education_agent(mode) → cached agent instance
  - reset_education_agent_cache() → clear cache (for tests)
  - warmup_education_agent(mode) → eager-init (for FastAPI startup)

Design:
  - Each mode string maps to one cached agent instance.
  - "pipeline" mode returns EducationAgent (never imports DeepAgent).
  - "deepagent"/"auto" modes create FinancialEducationDeepAgent once.
  - Tool traces are reset per-request inside DeepAgentWrapper.run().
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Process-level cache: mode → agent instance
_EDUCATION_AGENT_CACHE: dict[str, Any] = {}


def get_education_agent(mode: str = "deepagent") -> Any:
    """Return a cached education agent for the given mode.

    Args:
        mode: "deepagent" / "pipeline" / "auto".

    Returns:
        EducationAgent (pipeline) or
        FinancialEducationDeepAgent (deepagent/auto).
    """
    mode = mode.strip().lower()
    if mode not in ("deepagent", "pipeline", "auto"):
        logger.warning("Unknown education mode '%s', falling back to deepagent", mode)
        mode = "deepagent"

    if mode in _EDUCATION_AGENT_CACHE:
        return _EDUCATION_AGENT_CACHE[mode]

    agent = _create_education_agent(mode)
    _EDUCATION_AGENT_CACHE[mode] = agent
    return agent


def _create_education_agent(mode: str) -> Any:
    """Build a new agent instance. NOT cached — use get_education_agent."""
    from app.agents.education import EducationAgent

    if mode == "pipeline":
        logger.info("EducationAgentFactory: creating pipeline agent")
        return EducationAgent()

    # deepagent / auto
    logger.info("EducationAgentFactory: creating DeepAgent education agent (mode=%s)", mode)
    from app.agents.deepagent.education_deepagent import FinancialEducationDeepAgent

    return FinancialEducationDeepAgent(
        enabled=True,
        fallback_agent=EducationAgent(),
        configured_mode=mode,
    )


def reset_education_agent_cache() -> None:
    """Clear the agent cache. Intended for tests that mutate config."""
    _EDUCATION_AGENT_CACHE.clear()
    logger.debug("EducationAgentFactory: cache reset")


def warmup_education_agent(mode: str = "deepagent") -> bool:
    """Eagerly initialise the education agent.

    Does NOT crash on failure — agent will lazy-init on first request.

    Returns:
        True if warmup succeeded, False otherwise.
    """
    try:
        agent = get_education_agent(mode)
        if hasattr(agent, "debug_info"):
            info = agent.debug_info
            logger.info(
                "EducationAgentFactory: warmup OK — arch=%s available=%s",
                info.get("agent_architecture", "?"),
                info.get("deepagent_available", False),
            )
        else:
            logger.info("EducationAgentFactory: warmup OK — pipeline agent")
        return True
    except Exception as exc:
        logger.warning("EducationAgentFactory: warmup failed — %s", exc)
        return False
