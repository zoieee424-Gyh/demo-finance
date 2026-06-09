"""
Risk Control agent factory with process-level singleton caching.

Provides:
  - get_risk_control_agent(mode) → cached agent instance
  - reset_risk_control_agent_cache() → clear cache (for tests)
  - warmup_risk_control_agent(mode) → eager-init

Design:
  - Each mode string maps to one cached agent instance.
  - "pipeline" mode NEVER imports or initialises DeepAgent.
  - "deepagent"/"auto" modes create RiskControlDeepAgent once.
  - Tool traces are reset per-request inside DeepAgentWrapper.run().
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Process-level cache: mode → agent instance
_RISK_CONTROL_CACHE: dict[str, Any] = {}


def get_risk_control_agent(mode: str = "deepagent") -> Any:
    """Return a cached risk control agent for the given mode.

    Args:
        mode: "deepagent" / "pipeline" / "auto".

    Returns:
        RiskControlAgent (pipeline) or
        RiskControlDeepAgent (deepagent/auto).
    """
    mode = mode.strip().lower()
    if mode not in ("deepagent", "pipeline", "auto"):
        logger.warning("Unknown risk_control mode '%s', falling back to deepagent", mode)
        mode = "deepagent"

    if mode in _RISK_CONTROL_CACHE:
        return _RISK_CONTROL_CACHE[mode]

    agent = _create_agent(mode)
    _RISK_CONTROL_CACHE[mode] = agent
    return agent


def _create_agent(mode: str) -> Any:
    """Build a new agent instance. NOT cached — use get_risk_control_agent."""
    from app.agents.risk_control import RiskControlAgent

    if mode == "pipeline":
        logger.info("RiskControlAgentFactory: creating pipeline agent")
        return RiskControlAgent()

    # deepagent / auto
    logger.info("RiskControlAgentFactory: creating DeepAgent risk control agent (mode=%s)", mode)
    from app.agents.deepagent.risk_control_deepagent import RiskControlDeepAgent

    return RiskControlDeepAgent(
        enabled=True,
        fallback_agent=RiskControlAgent(),
        configured_mode=mode,
    )


def reset_risk_control_agent_cache() -> None:
    """Clear the agent cache. Intended for tests that mutate config."""
    _RISK_CONTROL_CACHE.clear()
    logger.debug("RiskControlAgentFactory: cache reset")


def warmup_risk_control_agent(mode: str = "deepagent") -> bool:
    """Eagerly initialise the risk control agent.

    Does NOT crash on failure — agent will lazy-init on first request.

    Returns:
        True if warmup succeeded, False otherwise.
    """
    try:
        agent = get_risk_control_agent(mode)
        if hasattr(agent, "debug_info"):
            info = agent.debug_info
            logger.info(
                "RiskControlAgentFactory: warmup OK — arch=%s available=%s",
                info.get("agent_architecture", "?"),
                info.get("deepagent_available", False),
            )
        else:
            logger.info("RiskControlAgentFactory: warmup OK — pipeline agent")
        return True
    except Exception as exc:
        logger.warning("RiskControlAgentFactory: warmup failed — %s", exc)
        return False
