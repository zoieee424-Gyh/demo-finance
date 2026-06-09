"""
Tests for the unified Agent Router (services/agent_router.py).

Covers:
  - 15+ routing cases covering all 5 intents
  - Confidence normalization
  - Priority-based conflict resolution
  - Candidate ranking
  - Edge cases (empty, unknown, no keywords)
"""

from __future__ import annotations

import pytest

from app.services.agent_router import route_intent, INTENT_PRIORITY, INTENT_DISPLAY_NAMES


# ── Basic routing cases ──────────────────────────────────────────

@pytest.mark.parametrize(
    "question, expected_intent, min_confidence",
    [
        # advisory
        ("我想做资产配置", "advisory", 0.15),
        ("基金定投适合我吗", "advisory", 0.15),
        ("我的投资组合怎么调整", "advisory", 0.15),
        # financial_report
        ("营业收入增长但净利润下降", "financial_report", 0.15),
        ("毛利率和经营现金流怎么看", "financial_report", 0.15),
        ("EBITDA持续下滑怎么回事", "financial_report", 0.15),
        # risk_control
        ("组合回撤太大怎么办", "risk_control", 0.15),
        ("仓位集中度风险怎么管理", "risk_control", 0.15),
        ("杠杆太高了怎么降风险", "risk_control", 0.15),
        # compliance
        ("这段话术保证收益合规吗", "compliance", 0.15),
        ("能不能写目标价", "compliance", 0.15),
        ("荐股违法吗", "compliance", 0.15),
        # education
        ("什么是基金定投", "education", 0.15),
        ("新手怎么理解债券基金", "education", 0.15),
        ("防诈骗有什么要注意的", "education", 0.15),
    ],
)
def test_router_basic_routing(question, expected_intent, min_confidence):
    """Verify basic keyword routing for all 5 intents."""
    decision = route_intent(question)
    assert decision.selected_intent == expected_intent, (
        f"Expected '{expected_intent}' but got '{decision.selected_intent}' "
        f"for: '{question}'. Reason: {decision.reason}"
    )
    assert decision.confidence >= min_confidence or decision.confidence == 0.0, (
        f"Confidence {decision.confidence} below threshold {min_confidence}"
    )


# ── Priority resolution cases ────────────────────────────────────

@pytest.mark.parametrize(
    "question, expected_intent",
    [
        # compliance should win over advisory when "保证收益" appears
        ("这个理财保证年化8%收益合规吗", "compliance"),
        # compliance should win when "违规" appears
        ("这种违规话术怎么处理", "compliance"),
        # financial_report wins over advisory when clear financial terms appear
        ("财报不错能不能买", "financial_report"),
        # risk_control wins for risk exposure questions
        ("风险暴露太大需要预警", "risk_control"),
    ],
)
def test_router_priority_resolution(question, expected_intent):
    """Verify priority-based conflict resolution for multi-intent questions."""
    decision = route_intent(question)
    assert decision.selected_intent == expected_intent, (
        f"Expected '{expected_intent}' but got '{decision.selected_intent}' "
        f"for: '{question}'. Reason: {decision.reason}"
    )


# ── Candidates list ──────────────────────────────────────────────

def test_router_returns_candidates():
    """Verify the decision includes a ranked candidate list."""
    decision = route_intent("我想配置资产并控制风险")
    assert len(decision.candidates) == 5, "Should have 5 candidates"
    # First candidate should match selected intent
    assert decision.candidates[0].intent == decision.selected_intent
    # Candidates should be sorted by score descending
    scores = [c.score for c in decision.candidates]
    assert scores == sorted(scores, reverse=True), "Candidates not sorted"


# ── Confidence ───────────────────────────────────────────────────

def test_router_confidence_range():
    """Verify confidence is always between 0 and 1."""
    questions = [
        "资产配置",
        "营收增长",
        "合规审查风险揭示不足",
        "什么是基金",
        "回撤控制",
    ]
    for q in questions:
        decision = route_intent(q)
        assert 0.0 <= decision.confidence <= 1.0, (
            f"Confidence out of range: {decision.confidence} for '{q}'"
        )


def test_router_strong_signal_high_confidence():
    """Strong keywords should yield higher confidence."""
    decision = route_intent("投资者适当性管理合规红线保证收益")
    assert decision.selected_intent == "compliance"
    # Multiple strong compliance keywords → confidence should be moderate+
    assert decision.confidence >= 0.15, (
        f"Expected higher confidence for strong compliance signal, "
        f"got {decision.confidence}"
    )


def test_router_sole_match_confidence_boost():
    """A question that only matches one intent should have decent confidence."""
    decision = route_intent("帮我做个压力测试看看杠杆风险")
    assert decision.selected_intent in ("risk_control",)
    # Should have at least some confidence since keywords match
    assert decision.confidence > 0, "Sole match should have non-zero confidence"


# ── Empty / unknown ──────────────────────────────────────────────

def test_router_empty_question_defaults_education():
    """Empty question should default to education with 0 confidence."""
    decision = route_intent("")
    assert decision.selected_intent == "education"
    assert decision.confidence == 0.0


def test_router_no_keywords_defaults_education():
    """Question with no matched keywords defaults to education."""
    decision = route_intent("你好啊今天天气不错")
    assert decision.selected_intent == "education"
    assert decision.confidence == 0.0


# ── Reason ───────────────────────────────────────────────────────

def test_router_includes_reason():
    """Verify the decision includes a non-empty reason string."""
    decision = route_intent("我的基金定投组合怎么优化")
    assert len(decision.reason) > 0, "Reason should not be empty"


def test_router_reason_mentions_display_name():
    """Reason should contain the Chinese display name of the selected intent."""
    decision = route_intent("这个推荐是不是合规的")
    display_name = INTENT_DISPLAY_NAMES.get(decision.selected_intent, "")
    assert display_name in decision.reason, (
        f"Reason should mention '{display_name}': {decision.reason}"
    )


# ── Priority constants ───────────────────────────────────────────

def test_intent_priority_order():
    """Verify the priority ordering: compliance > financial_report > risk_control > advisory > education."""
    assert INTENT_PRIORITY["compliance"] > INTENT_PRIORITY["financial_report"]
    assert INTENT_PRIORITY["financial_report"] > INTENT_PRIORITY["risk_control"]
    assert INTENT_PRIORITY["risk_control"] > INTENT_PRIORITY["advisory"]
    assert INTENT_PRIORITY["advisory"] > INTENT_PRIORITY["education"]


def test_all_display_names_present():
    """Verify all 5 intents have Chinese display names."""
    for intent in ["advisory", "financial_report", "risk_control", "compliance", "education"]:
        assert intent in INTENT_DISPLAY_NAMES
        assert len(INTENT_DISPLAY_NAMES[intent]) > 0
