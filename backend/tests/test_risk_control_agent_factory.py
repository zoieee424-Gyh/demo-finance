"""Tests for the risk control agent factory."""

from __future__ import annotations

import pytest


class TestRiskControlAgentFactory:
    """Factory creates and caches RiskControlDeepAgent or RiskControlAgent."""

    def test_factory_returns_agent(self):
        from app.services.risk_control_agent_factory import (
            get_risk_control_agent,
            reset_risk_control_agent_cache,
        )
        reset_risk_control_agent_cache()
        agent = get_risk_control_agent("deepagent")
        assert agent is not None
        assert hasattr(agent, "answer")

    def test_same_mode_returns_cached_instance(self):
        from app.services.risk_control_agent_factory import (
            get_risk_control_agent,
            reset_risk_control_agent_cache,
        )
        reset_risk_control_agent_cache()
        a = get_risk_control_agent("deepagent")
        b = get_risk_control_agent("deepagent")
        assert a is b

    def test_different_modes_return_different_instances(self):
        from app.services.risk_control_agent_factory import (
            get_risk_control_agent,
            reset_risk_control_agent_cache,
        )
        reset_risk_control_agent_cache()
        deep = get_risk_control_agent("deepagent")
        pipe = get_risk_control_agent("pipeline")
        assert deep is not pipe

    def test_reset_clears_cache(self):
        from app.services.risk_control_agent_factory import (
            get_risk_control_agent,
            reset_risk_control_agent_cache,
        )
        reset_risk_control_agent_cache()
        a = get_risk_control_agent("deepagent")
        reset_risk_control_agent_cache()
        b = get_risk_control_agent("deepagent")
        assert a is not b

    def test_pipeline_mode_creates_pipeline_agent(self):
        from app.services.risk_control_agent_factory import (
            get_risk_control_agent,
            reset_risk_control_agent_cache,
        )
        reset_risk_control_agent_cache()
        agent = get_risk_control_agent("pipeline")
        from app.agents.risk_control import RiskControlAgent
        assert isinstance(agent, RiskControlAgent)

    def test_deepagent_mode_has_debug_info(self):
        from app.services.risk_control_agent_factory import (
            get_risk_control_agent,
            reset_risk_control_agent_cache,
        )
        reset_risk_control_agent_cache()
        agent = get_risk_control_agent("deepagent")
        info = agent.debug_info
        assert "agent_architecture" in info
        assert "tool_traces" in info
