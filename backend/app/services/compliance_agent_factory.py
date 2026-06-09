"""
Compliance agent factory with process-level singleton caching.

Provides:
  - get_compliance_agent(mode) → cached agent instance
  - reset_compliance_agent_cache() → clear cache (for tests)
  - warmup_compliance_agent(mode) → eager-init

Design:
  - Each mode string maps to one cached agent instance.
  - "pipeline" mode NEVER imports or initialises DeepAgent.
  - "deepagent"/"auto" modes create ComplianceDeepAgent once.
  - Tool traces are reset per-request inside DeepAgentWrapper.run().
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Process-level cache: mode → agent instance
_COMPLIANCE_CACHE: dict[str, Any] = {}


def get_compliance_agent(mode: str = "deepagent") -> Any:
    """Return a cached compliance agent for the given mode.

    Args:
        mode: "deepagent" / "pipeline" / "auto".

    Returns:
        ComplianceAgent (pipeline) or
        ComplianceDeepAgent (deepagent/auto).
    """
    mode = mode.strip().lower()
    if mode not in ("deepagent", "pipeline", "auto"):
        logger.warning("Unknown compliance mode '%s', falling back to deepagent", mode)
        mode = "deepagent"

    if mode in _COMPLIANCE_CACHE:
        return _COMPLIANCE_CACHE[mode]

    agent = _create_agent(mode)
    _COMPLIANCE_CACHE[mode] = agent
    return agent


def _create_agent(mode: str) -> Any:
    """Build a new agent instance. NOT cached — use get_compliance_agent."""
    from app.agents.compliance import ComplianceAgent

    if mode == "pipeline":
        logger.info("ComplianceAgentFactory: creating pipeline agent")
        return ComplianceAgent()

    # deepagent / auto
    logger.info("ComplianceAgentFactory: creating DeepAgent compliance agent (mode=%s)", mode)
    from app.agents.deepagent.compliance_deepagent import ComplianceDeepAgent

    return ComplianceDeepAgent(
        enabled=True,
        fallback_agent=ComplianceAgent(),
        configured_mode=mode,
    )


def reset_compliance_agent_cache() -> None:
    """Clear the agent cache. Intended for tests that mutate config."""
    _COMPLIANCE_CACHE.clear()
    logger.debug("ComplianceAgentFactory: cache reset")


def warmup_compliance_agent(mode: str = "deepagent") -> bool:
    """Eagerly initialise the compliance agent.

    Does NOT crash on failure — agent will lazy-init on first request.

    Returns:
        True if warmup succeeded, False otherwise.
    """
    try:
        agent = get_compliance_agent(mode)
        if hasattr(agent, "debug_info"):
            info = agent.debug_info
            logger.info(
                "ComplianceAgentFactory: warmup OK — arch=%s available=%s",
                info.get("agent_architecture", "?"),
                info.get("deepagent_available", False),
            )
        else:
            logger.info("ComplianceAgentFactory: warmup OK — pipeline agent")
        return True
    except Exception as exc:
        logger.warning("ComplianceAgentFactory: warmup failed — %s", exc)
        return False
