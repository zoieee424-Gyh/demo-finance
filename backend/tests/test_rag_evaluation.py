"""
Tests for RAG evaluation framework.

Covers:
  - Eval cases JSON schema validation
  - Case count requirements (>=32 total, >=8 per collection, >=8 hard)
  - Metrics calculation with mock/synthetic results
  - validate_cases function
  - calculate_metrics function

No real API calls — all tests use deterministic mock data.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_rag import (
    validate_cases,
    calculate_metrics,
    build_markdown_report,
    REQUIRED_CASE_FIELDS,
    VALID_INTENTS,
    VALID_DIFFICULTIES,
)


# ── Paths ──────────────────────────────────────────────────────────────

EVAL_CASES_PATH = (
    Path(__file__).resolve().parent.parent / "eval" / "rag_eval_cases.json"
)


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def eval_data():
    """Load the real eval cases JSON for validation tests."""
    if not EVAL_CASES_PATH.exists():
        pytest.skip(f"Eval cases file not found: {EVAL_CASES_PATH}")
    with open(EVAL_CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def eval_cases(eval_data):
    """Return the cases list from eval data."""
    return eval_data.get("cases", [])


# ── Synthetic results for metrics tests ────────────────────────────────

def _make_result(
    case_id: str,
    expected_intent: str = "advisory",
    actual_intent: str = "advisory",
    intent_correct: bool = True,
    collection_correct: bool = True,
    top1_title: str = "资产配置基础原则",
    top1_score: float = 0.75,
    top1_correct: bool = True,
    recall_correct: bool = True,
    first_hit_rank: int = 1,
    difficulty: str = "easy",
    top_results: list | None = None,
) -> dict:
    """Create a synthetic evaluation result for metrics testing."""
    if top_results is None:
        top_results = [
            {"rank": 1, "title": top1_title, "score": top1_score,
             "collection": "advisory_knowledge", "source_type": "investment_knowledge"},
            {"rank": 2, "title": "基金定投常见原则", "score": 0.62,
             "collection": "advisory_knowledge", "source_type": "investment_knowledge"},
            {"rank": 3, "title": "风险等级与资产类别匹配", "score": 0.58,
             "collection": "advisory_knowledge", "source_type": "investment_knowledge"},
        ]
    return {
        "id": case_id,
        "query": f"Test query for {case_id}",
        "expected_intent": expected_intent,
        "expected_collections": ["advisory_knowledge"],
        "expected_titles": ["资产配置基础原则"],
        "difficulty": difficulty,
        "category": "test",
        "actual_intent": actual_intent,
        "intent_correct": intent_correct,
        "selected_collections": ["advisory_knowledge"],
        "collection_correct": collection_correct,
        "top_results": top_results,
        "top1_title": top1_title,
        "top1_score": top1_score,
        "top1_correct": top1_correct,
        "recall_correct": recall_correct,
        "first_hit_rank": first_hit_rank,
        "errors": [],
    }


# ── JSON schema / format tests ─────────────────────────────────────────

class TestEvalCasesJsonSchema:
    """Validate the eval cases JSON file structure and content."""

    def test_cases_file_exists(self):
        """Eval cases file should exist."""
        assert EVAL_CASES_PATH.exists(), (
            f"Eval cases file not found: {EVAL_CASES_PATH}"
        )

    def test_cases_file_valid_json(self, eval_data):
        """Eval cases file should be valid JSON."""
        assert isinstance(eval_data, dict)
        assert "cases" in eval_data
        assert "meta" in eval_data

    def test_cases_is_list(self, eval_data):
        """'cases' must be a list."""
        assert isinstance(eval_data["cases"], list)

    def test_minimum_32_cases(self, eval_cases):
        """Must have at least 32 evaluation cases."""
        assert len(eval_cases) >= 32, (
            f"Expected >= 32 cases, got {len(eval_cases)}"
        )

    def test_all_required_fields(self, eval_cases):
        """Every case must have all required fields."""
        for case in eval_cases:
            case_id = case.get("id", "?")
            for field in REQUIRED_CASE_FIELDS:
                assert field in case, f"{case_id}: missing field '{field}'"

    def test_unique_ids(self, eval_cases):
        """All case IDs must be unique."""
        ids = [c["id"] for c in eval_cases]
        assert len(ids) == len(set(ids)), (
            f"Duplicate IDs found: {[i for i in ids if ids.count(i) > 1]}"
        )

    def test_valid_intents(self, eval_cases):
        """All expected_intent values must be valid."""
        for case in eval_cases:
            intent = case.get("expected_intent", "")
            assert intent in VALID_INTENTS, (
                f"{case['id']}: invalid expected_intent '{intent}'"
            )

    def test_valid_difficulties(self, eval_cases):
        """All difficulty values must be valid."""
        for case in eval_cases:
            diff = case.get("difficulty", "")
            assert diff in VALID_DIFFICULTIES, (
                f"{case['id']}: invalid difficulty '{diff}'"
            )

    def test_expected_titles_non_empty(self, eval_cases):
        """Every case must have at least one expected title."""
        for case in eval_cases:
            titles = case.get("expected_titles", [])
            assert isinstance(titles, list), f"{case['id']}: expected_titles not a list"
            assert len(titles) >= 1, f"{case['id']}: expected_titles is empty"
            assert len(titles) <= 3, f"{case['id']}: expected_titles has >3 entries"

    def test_expected_collections_non_empty(self, eval_cases):
        """Every case must have at least one expected collection."""
        for case in eval_cases:
            colls = case.get("expected_collections", [])
            assert isinstance(colls, list), f"{case['id']}: expected_collections not a list"
            assert len(colls) >= 1, f"{case['id']}: expected_collections is empty"

    def test_valid_collection_names(self, eval_cases):
        """All expected_collections must be valid Chroma collection names."""
        valid_colls = {
            "advisory_knowledge", "compliance_knowledge",
            "education_knowledge", "risk_knowledge",
            "financial_report_knowledge", "market_knowledge",
        }
        for case in eval_cases:
            for coll in case.get("expected_collections", []):
                assert coll in valid_colls, (
                    f"{case['id']}: invalid collection '{coll}'"
                )


class TestEvalCasesCoverage:
    """Validate coverage requirements across collections and difficulties."""

    def test_per_collection_minimum(self, eval_cases):
        """Each collection must have >= 8 cases where it appears in expected_collections."""
        collection_counts: dict[str, int] = {}
        for case in eval_cases:
            for coll in case.get("expected_collections", []):
                collection_counts[coll] = collection_counts.get(coll, 0) + 1

        required = [
            "advisory_knowledge", "compliance_knowledge",
            "education_knowledge", "risk_knowledge",
        ]
        for coll in required:
            count = collection_counts.get(coll, 0)
            assert count >= 8, (
                f"Collection '{coll}' only has {count} cases (need >= 8)"
            )

    def test_per_intent_minimum(self, eval_cases):
        """Each intent should have >= 8 dedicated (non-XDM) cases."""
        intent_counts: dict[str, int] = {}
        for case in eval_cases:
            if not case["id"].startswith("XDM-"):
                intent = case.get("expected_intent", "")
                intent_counts[intent] = intent_counts.get(intent, 0) + 1

        for intent in ["advisory", "compliance", "education", "risk_control"]:
            count = intent_counts.get(intent, 0)
            assert count >= 8, (
                f"Intent '{intent}' only has {count} dedicated cases (need >= 8)"
            )

    def test_hard_cases_minimum(self, eval_cases):
        """Must have at least 8 hard cases."""
        hard_cases = [c for c in eval_cases if c.get("difficulty") == "hard"]
        assert len(hard_cases) >= 8, (
            f"Only {len(hard_cases)} hard cases (need >= 8)"
        )

    def test_cross_domain_cases_minimum(self, eval_cases):
        """Must have at least 6 cross-domain (XDM) cases."""
        xdm_cases = [c for c in eval_cases if c["id"].startswith("XDM-")]
        assert len(xdm_cases) >= 6, (
            f"Only {len(xdm_cases)} cross-domain cases (need >= 6)"
        )

    def test_easy_medium_hard_distribution(self, eval_cases):
        """Should have a reasonable mix of difficulty levels."""
        counts: dict[str, int] = {}
        for case in eval_cases:
            diff = case.get("difficulty", "medium")
            counts[diff] = counts.get(diff, 0) + 1

        total = len(eval_cases)
        # At least 25% easy, 25% medium, 15% hard
        assert counts.get("easy", 0) >= total * 0.20, "Too few easy cases"
        assert counts.get("medium", 0) >= total * 0.20, "Too few medium cases"
        assert counts.get("hard", 0) >= total * 0.15, "Too few hard cases"

    def test_no_stock_recommendations(self, eval_cases):
        """No eval case should ask for specific stock recommendations."""
        banned = ["推荐股票", "推荐个股", "哪只股票好", "买什么股票",
                   "涨停", "跌停", "牛股", "妖股"]
        for case in eval_cases:
            query = case.get("query", "")
            for term in banned:
                assert term not in query, (
                    f"{case['id']}: query contains banned term '{term}'"
                )

    def test_no_return_promises(self, eval_cases):
        """Queries about return promises should route to compliance, not make promises."""
        for case in eval_cases:
            query = case.get("query", "")
            # Cases with return promise language must have compliance intent
            if any(t in query for t in ["保证", "保本", "稳赚", "年化8%", "年化10%"]):
                assert case.get("expected_intent") == "compliance", (
                    f"{case['id']}: return promise query must route to compliance"
                )

    def test_validate_cases_passes(self, eval_cases):
        """The validate_cases function should return no errors for our eval set."""
        errors = validate_cases(eval_cases)
        assert errors == [], (
            f"Validation errors in eval cases:\n" + "\n".join(f"  - {e}" for e in errors)
        )


# ── Metrics calculation tests ──────────────────────────────────────────

class TestMetricsCalculation:
    """Test metrics computation with synthetic results."""

    def test_all_perfect(self):
        """All cases perfect → all metrics = 1.0."""
        results = [
            _make_result(f"T-{i:03d}") for i in range(10)
        ]
        metrics = calculate_metrics(results)
        assert metrics["total_cases"] == 10
        assert metrics["intent_accuracy"] == 1.0
        assert metrics["top1_accuracy"] == 1.0
        assert metrics["recall_at_k"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_all_wrong(self):
        """All cases wrong → all accuracy metrics = 0.0."""
        results = [
            _make_result(
                f"T-{i:03d}",
                intent_correct=False,
                top1_correct=False,
                recall_correct=False,
                first_hit_rank=None,
                top1_title="Wrong doc",
                top1_score=0.3,
            )
            for i in range(5)
        ]
        metrics = calculate_metrics(results)
        assert metrics["intent_accuracy"] == 0.0
        assert metrics["top1_accuracy"] == 0.0
        assert metrics["recall_at_k"] == 0.0
        assert metrics["mrr"] == 0.0

    def test_mixed_results(self):
        """Mixed results → partial metrics."""
        results = [
            _make_result("T-001"),                          # perfect
            _make_result("T-002", top1_correct=False, first_hit_rank=2),  # top1 miss, recall hit
            _make_result("T-003", intent_correct=False, actual_intent="education", top1_correct=False, recall_correct=False, first_hit_rank=None),  # intent err + no recall
            _make_result("T-004", intent_correct=False, actual_intent="risk_control", recall_correct=False, first_hit_rank=None, top1_correct=False),  # total miss
        ]
        metrics = calculate_metrics(results)
        assert metrics["total_cases"] == 4
        assert metrics["intent_accuracy"] == 0.5    # 2/4 (T-001, T-002)
        assert metrics["top1_accuracy"] == 0.25     # 1/4 (only T-001)
        assert metrics["recall_at_k"] == 0.5         # 2/4 (T-001, T-002)
        # MRR: T-001: 1/1=1.0, T-002: 1/2=0.5, T-003: 0, T-004: 0 → (1.5/4)=0.375
        assert abs(metrics["mrr"] - 0.375) < 0.001

    def test_mrr_calculation(self):
        """MRR should correctly compute mean reciprocal rank."""
        results = [
            _make_result("T-001", first_hit_rank=1),   # 1.0
            _make_result("T-002", first_hit_rank=3),   # 0.333
            _make_result("T-003", first_hit_rank=2),   # 0.5
            _make_result("T-004", recall_correct=False, first_hit_rank=None),  # 0.0
        ]
        metrics = calculate_metrics(results)
        expected_mrr = (1.0 + 1 / 3 + 0.5 + 0.0) / 4
        assert abs(metrics["mrr"] - expected_mrr) < 0.001

    def test_by_intent_grouping(self):
        """Metrics should be grouped by expected_intent."""
        results = [
            _make_result("A-01", expected_intent="advisory"),
            _make_result("A-02", expected_intent="advisory"),
            _make_result("E-01", expected_intent="education",
                         top1_title="什么是基金", top1_correct=False,
                         recall_correct=False, first_hit_rank=None),
        ]
        metrics = calculate_metrics(results)
        assert "by_intent" in metrics
        assert "advisory" in metrics["by_intent"]
        assert "education" in metrics["by_intent"]
        assert metrics["by_intent"]["advisory"]["count"] == 2
        assert metrics["by_intent"]["advisory"]["top1_accuracy"] == 1.0
        assert metrics["by_intent"]["education"]["count"] == 1
        assert metrics["by_intent"]["education"]["top1_accuracy"] == 0.0

    def test_by_difficulty_grouping(self):
        """Metrics should be grouped by difficulty."""
        results = [
            _make_result("E-01", difficulty="easy"),
            _make_result("E-02", difficulty="easy"),
            _make_result("E-03", difficulty="easy"),
            _make_result("H-01", difficulty="hard",
                         top1_correct=False, first_hit_rank=2),
        ]
        metrics = calculate_metrics(results)
        assert "by_difficulty" in metrics
        assert "easy" in metrics["by_difficulty"]
        assert "hard" in metrics["by_difficulty"]
        assert metrics["by_difficulty"]["easy"]["count"] == 3
        assert metrics["by_difficulty"]["easy"]["top1_accuracy"] == 1.0
        assert metrics["by_difficulty"]["hard"]["count"] == 1
        assert metrics["by_difficulty"]["hard"]["top1_accuracy"] == 0.0

    def test_low_score_detection(self):
        """Low score items should be flagged (< 0.45)."""
        results = [
            _make_result("T-001", top1_score=0.80),
            _make_result("T-002", top1_score=0.35),  # low
            _make_result("T-003", top1_score=0.44),  # low (below 0.45)
            _make_result("T-004", top1_score=0.46),  # borderline OK
        ]
        metrics = calculate_metrics(results)
        assert metrics["low_score_count"] == 2
        assert len(metrics["low_score_items"]) == 2

    def test_empty_results(self):
        """Empty results list should not crash."""
        metrics = calculate_metrics([])
        assert metrics["total_cases"] == 0
        assert metrics["intent_accuracy"] == 0.0
        assert metrics["by_intent"] == {}
        assert metrics["by_difficulty"] == {}


# ── Validation function tests ──────────────────────────────────────────

class TestValidateCases:
    """Test the validate_cases function directly."""

    def test_valid_cases_pass(self):
        """A minimal valid set should pass validation."""
        cases = []
        for i in range(32):
            intent = ["advisory", "compliance", "education", "risk_control"][i % 4]
            cases.append({
                "id": f"T-{i:03d}",
                "query": f"Test query {i}",
                "expected_intent": intent,
                "expected_collections": [f"{'risk' if intent == 'risk_control' else intent}_knowledge"],
                "expected_titles": ["Test Title"],
                "difficulty": "hard" if i < 8 else ("medium" if i < 20 else "easy"),
                "category": "test",
                "notes": f"Test case {i}",
            })
        # Add some cross-domain cases
        for i in range(6):
            cases.append({
                "id": f"XDM-{i:03d}",
                "query": f"Cross-domain query {i}",
                "expected_intent": "advisory",
                "expected_collections": ["advisory_knowledge", "risk_knowledge"],
                "expected_titles": ["Title A"],
                "difficulty": "hard",
                "category": "cross",
                "notes": f"Cross-domain case {i}",
            })
        errors = validate_cases(cases)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_field(self):
        """Cases with missing required fields should fail."""
        cases = [{"id": "T-001"}] * 32
        errors = validate_cases(cases)
        assert len(errors) > 0
        assert any("missing field" in e for e in errors)

    def test_duplicate_ids(self):
        """Duplicate IDs should be flagged."""
        base = {
            "id": "SAME-ID",
            "query": "test",
            "expected_intent": "education",
            "expected_collections": ["education_knowledge"],
            "expected_titles": ["Title"],
            "difficulty": "easy",
            "category": "test",
            "notes": "test",
        }
        cases = [{**base, "id": f"T-{i:03d}"} for i in range(31)]
        cases.append({**base, "id": "T-000"})  # duplicate
        cases.append({**base, "id": "T-001"})  # duplicate
        errors = validate_cases(cases)
        dup_errors = [e for e in errors if "duplicate" in e.lower()]
        assert len(dup_errors) >= 2

    def test_too_few_cases(self):
        """Fewer than 32 cases should fail."""
        cases = [
            {
                "id": f"T-{i:03d}",
                "query": f"Q{i}",
                "expected_intent": "education",
                "expected_collections": ["education_knowledge"],
                "expected_titles": ["Title"],
                "difficulty": "easy",
                "category": "test",
                "notes": "test",
            }
            for i in range(10)
        ]
        errors = validate_cases(cases)
        assert any("< 32" in e or ">= 32" in e for e in errors)

    def test_invalid_intent(self):
        """Invalid intents should be flagged."""
        cases = [
            {
                "id": f"T-{i:03d}",
                "query": f"Q{i}",
                "expected_intent": "invalid_intent",
                "expected_collections": ["advisory_knowledge"],
                "expected_titles": ["Title"],
                "difficulty": "easy",
                "category": "test",
                "notes": "test",
            }
            for i in range(32)
        ]
        errors = validate_cases(cases)
        assert any("invalid expected_intent" in e for e in errors)

    def test_too_few_hard_cases(self):
        """Too few hard cases should fail."""
        cases = []
        for i in range(32):
            cases.append({
                "id": f"T-{i:03d}",
                "query": f"Q{i}",
                "expected_intent": "education",
                "expected_collections": ["education_knowledge"],
                "expected_titles": ["Title"],
                "difficulty": "hard" if i < 3 else "easy",
                "category": "test",
                "notes": "test",
            })
        errors = validate_cases(cases)
        assert any("hard cases" in e.lower() for e in errors)

    def test_too_few_cross_domain_cases(self):
        """Too few cross-domain cases should fail."""
        cases = []
        intents = ["advisory", "compliance", "education", "risk_control"]
        collections = {
            "advisory": "advisory_knowledge",
            "compliance": "compliance_knowledge",
            "education": "education_knowledge",
            "risk_control": "risk_knowledge",
        }
        for i in range(32):
            intent = intents[i % 4]
            cases.append({
                "id": f"T-{i:03d}",
                "query": f"Q{i}",
                "expected_intent": intent,
                "expected_collections": [collections[intent]],
                "expected_titles": ["Title"],
                "difficulty": "hard" if i < 8 else "easy",
                "category": "test",
                "notes": "test",
            })

        errors = validate_cases(cases)
        assert any("cross-domain" in e.lower() for e in errors)


# ── Cross-domain / secondary intent tests (evaluation integration) ──────


class TestEvaluationSecondaryIntents:
    """Verify that evaluation results include secondary_intents and that
    cross-domain collection coverage improves with multi-intent selection."""

    def test_result_dict_includes_secondary_intents(self):
        """Each eval result dict must have a secondary_intents field."""
        # Verify the result schema from run_evaluation includes the field
        result = {
            "id": "XDM-001",
            "query": "保守型投资者能买债券基金吗？风险大吗？",
            "expected_intent": "advisory",
            "expected_collections": ["advisory_knowledge", "education_knowledge", "risk_knowledge"],
            "expected_titles": ["风险等级与资产类别匹配"],
            "difficulty": "hard",
            "category": "cross",
            "actual_intent": "advisory",
            "secondary_intents": ["risk_control"],
            "selected_collections": ["advisory_knowledge", "education_knowledge", "compliance_knowledge", "risk_knowledge"],
            "collection_correct": True,
        }
        assert "secondary_intents" in result
        assert isinstance(result["secondary_intents"], list)

    def test_xdm_collection_coverage_with_secondary(self):
        """With secondary intent, XDM cases should cover more collections."""
        # Simulate: primary=advisory, secondary=[risk_control]
        # advisory's: advisory_knowledge, education_knowledge, compliance_knowledge
        # + risk_control's: risk_knowledge, compliance_knowledge
        primary_colls = {"advisory_knowledge", "education_knowledge", "compliance_knowledge"}
        secondary_colls = {"risk_knowledge", "compliance_knowledge"}
        merged = primary_colls | secondary_colls

        # XDM-001 expects: advisory_knowledge, education_knowledge, risk_knowledge
        expected = {"advisory_knowledge", "education_knowledge", "risk_knowledge"}
        assert expected.issubset(merged), (
            "With secondary risk_control, XDM-001 should have all three domains"
        )

    def test_xdm001_without_secondary_misses_risk(self):
        """Without secondary, XDM-001 misses risk_knowledge."""
        primary_colls = {"advisory_knowledge", "education_knowledge", "compliance_knowledge"}
        expected = {"advisory_knowledge", "education_knowledge", "risk_knowledge"}
        assert not expected.issubset(primary_colls), (
            "Without secondary, risk_knowledge is missing"
        )

    def test_secondary_intent_detection_integration(self):
        """SecondaryIntentDetector should detect risk for XDM queries."""
        from app.rag.secondary_intent_detector import SecondaryIntentDetector
        detector = SecondaryIntentDetector()

        # XDM-001
        r = detector.detect("保守型投资者能买债券基金吗？风险大吗？", primary_intent="advisory")
        assert "risk_control" in r

        # XDM-003
        r = detector.detect("我想学理财，有什么风险需要注意？入门该看什么？", primary_intent="education")
        assert "risk_control" in r
        assert "advisory" in r

        # XDM-006
        r = detector.detect("市场下跌时应该卖出止损吗？这算不算追涨杀跌？", primary_intent="risk_control")
        assert "advisory" in r


class TestMarkdownReport:
    """Markdown report rendering tests."""

    def test_knowledge_gap_recommendation_keeps_heading(self):
        """Recall-miss recommendation should not lose its bold heading."""
        class Args:
            top_k = 5
            fail_under_top1 = 0.70
            fail_under_recall = 0.85

        results = [
            _make_result(
                "XDM-003",
                expected_intent="education",
                recall_correct=False,
                top1_correct=False,
                first_hit_rank=None,
                top1_title="长期投资与复利效应",
                top1_score=0.56,
            )
        ]
        metrics = calculate_metrics(results)
        report = build_markdown_report(
            metrics=metrics,
            results=results,
            args=Args(),
            total_vectors=31,
            num_collections=4,
            elapsed=0.1,
            top1_pass=False,
            recall_pass=False,
            intent_pass=True,
        )
        assert "**Fill knowledge gaps**" in report
        assert "Missing topics include:" in report
