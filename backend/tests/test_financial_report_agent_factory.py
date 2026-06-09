"""Tests for the financial report agent factory."""

from __future__ import annotations

import pytest


class TestFinancialReportAgentFactory:
    """Factory creates and caches FinancialReportDeepAgent or FinancialReportAgent."""

    def test_factory_returns_agent(self):
        from app.services.financial_report_agent_factory import (
            get_financial_report_agent,
            reset_financial_report_agent_cache,
        )
        reset_financial_report_agent_cache()
        agent = get_financial_report_agent("deepagent")
        assert agent is not None
        assert hasattr(agent, "answer")

    def test_same_mode_returns_cached_instance(self):
        from app.services.financial_report_agent_factory import (
            get_financial_report_agent,
            reset_financial_report_agent_cache,
        )
        reset_financial_report_agent_cache()
        a = get_financial_report_agent("deepagent")
        b = get_financial_report_agent("deepagent")
        assert a is b

    def test_different_modes_return_different_instances(self):
        from app.services.financial_report_agent_factory import (
            get_financial_report_agent,
            reset_financial_report_agent_cache,
        )
        reset_financial_report_agent_cache()
        deep = get_financial_report_agent("deepagent")
        pipe = get_financial_report_agent("pipeline")
        assert deep is not pipe

    def test_reset_clears_cache(self):
        from app.services.financial_report_agent_factory import (
            get_financial_report_agent,
            reset_financial_report_agent_cache,
        )
        reset_financial_report_agent_cache()
        a = get_financial_report_agent("deepagent")
        reset_financial_report_agent_cache()
        b = get_financial_report_agent("deepagent")
        assert a is not b

    def test_pipeline_mode_creates_pipeline_agent(self):
        from app.services.financial_report_agent_factory import (
            get_financial_report_agent,
            reset_financial_report_agent_cache,
        )
        reset_financial_report_agent_cache()
        agent = get_financial_report_agent("pipeline")
        from app.agents.financial_report import FinancialReportAgent
        assert isinstance(agent, FinancialReportAgent)

    def test_deepagent_mode_has_debug_info(self):
        from app.services.financial_report_agent_factory import (
            get_financial_report_agent,
            reset_financial_report_agent_cache,
        )
        reset_financial_report_agent_cache()
        agent = get_financial_report_agent("deepagent")
        info = agent.debug_info
        assert "agent_architecture" in info
        assert "tool_traces" in info

    def test_warmup_returns_bool(self):
        from app.services.financial_report_agent_factory import (
            reset_financial_report_agent_cache,
            warmup_financial_report_agent,
        )
        reset_financial_report_agent_cache()
        result = warmup_financial_report_agent("pipeline")
        assert result is True

    def test_unknown_mode_falls_back_to_deepagent(self):
        from app.services.financial_report_agent_factory import (
            get_financial_report_agent,
            reset_financial_report_agent_cache,
        )
        reset_financial_report_agent_cache()
        agent = get_financial_report_agent("bogus")
        assert agent is not None
        assert hasattr(agent, "debug_info")
