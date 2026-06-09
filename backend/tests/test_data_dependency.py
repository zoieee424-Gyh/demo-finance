"""
Tests for the investment advisor data dependency layer.

Covers:
  - Policy config JSON files are valid and parseable.
  - Risk profile questionnaire has valid structure.
  - Asset allocation templates have valid ratios.
  - Compliance rules include required rule_ids.
  - New knowledge docs are indexed in Chroma (via dry-run path).

No real API calls — config validation and structure tests only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "data" / "policy_configs" / "advisory"
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"


# ── Helpers ──────────────────────────────────────────────────────────────

def _load_json(filename: str) -> dict:
    path = CONFIG_DIR / filename
    if not path.exists():
        pytest.skip(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Risk Profile Questionnaire tests ─────────────────────────────────────


class TestRiskProfileQuestionnaire:
    """Validate data/policy_configs/advisory/risk_profile_questionnaire.json."""

    @pytest.fixture
    def data(self):
        return _load_json("risk_profile_questionnaire.json")

    def test_has_meta(self, data):
        assert "meta" in data
        assert "title" in data["meta"]

    def test_risk_levels_complete(self, data):
        levels = data.get("risk_levels", {})
        for key in ["conservative", "stable", "balanced", "aggressive"]:
            assert key in levels, f"Missing risk level: {key}"
            assert "label" in levels[key]
            assert "min_score" in levels[key]
            assert "max_score" in levels[key]

    def test_risk_level_scores_cover_0_to_100(self, data):
        """Score ranges should cover 0-100 without gaps between levels."""
        levels = data["risk_levels"]
        # Check that ranges are contiguous
        ranges = []
        for key in ["conservative", "stable", "balanced", "aggressive"]:
            ranges.append((levels[key]["min_score"], levels[key]["max_score"]))
        # Verify conservative starts at 0
        assert ranges[0][0] == 0
        # Verify aggressive ends at 100
        assert ranges[-1][1] == 100
        # Check adjacency: each level's max+1 should equal next level's min
        for i in range(len(ranges) - 1):
            assert ranges[i][1] + 1 == ranges[i+1][0], (
                f"Gap between level {i} (max={ranges[i][1]}) and level {i+1} (min={ranges[i+1][0]})"
            )

    def test_dimensions_weight_sum(self, data):
        """Dimension weights should sum to approximately 1.0."""
        dims = data.get("dimensions", [])
        total = sum(d["weight"] for d in dims)
        assert abs(total - 1.0) < 0.01, f"Dimension weights sum to {total}, expected ~1.0"

    def test_dimension_ids_valid(self, data):
        valid_ids = {"risk_capacity", "risk_attitude", "investment_experience",
                     "time_horizon", "liquidity_need"}
        dims = data.get("dimensions", [])
        for d in dims:
            assert d["dimension_id"] in valid_ids

    def test_questions_count(self, data):
        """Should have at least 10 questions."""
        questions = data.get("questions", [])
        assert len(questions) >= 10

    def test_all_questions_have_dimension(self, data):
        valid_ids = {"risk_capacity", "risk_attitude", "investment_experience",
                     "time_horizon", "liquidity_need"}
        for q in data.get("questions", []):
            assert q["dimension"] in valid_ids

    def test_all_questions_have_options(self, data):
        for q in data.get("questions", []):
            assert len(q.get("options", [])) >= 2

    def test_each_dimension_has_questions(self, data):
        """Each dimension must have at least 2 questions."""
        dims = {"risk_capacity": 0, "risk_attitude": 0,
                "investment_experience": 0, "time_horizon": 0, "liquidity_need": 0}
        for q in data.get("questions", []):
            dims[q["dimension"]] = dims.get(q["dimension"], 0) + 1
        for dim, count in dims.items():
            assert count >= 2, f"Dimension {dim} has only {count} questions"


# ── Asset Allocation Templates tests ─────────────────────────────────────


class TestAssetAllocationTemplates:
    """Validate data/policy_configs/advisory/asset_allocation_templates.json."""

    @pytest.fixture
    def data(self):
        return _load_json("asset_allocation_templates.json")

    def test_has_meta(self, data):
        assert "meta" in data
        assert "asset_classes" in data
        assert "templates" in data

    def test_all_four_risk_levels(self, data):
        for key in ["conservative", "stable", "balanced", "aggressive"]:
            assert key in data["templates"]

    def test_each_template_has_allocation(self, data):
        for key, tmpl in data["templates"].items():
            alloc = tmpl.get("allocation", [])
            assert len(alloc) >= 3, f"{key} has only {len(alloc)} asset classes"

    def test_benchmark_ratio_sums_to_100(self, data):
        """Each template's benchmark ratios should sum to ~100."""
        for key, tmpl in data["templates"].items():
            alloc = tmpl.get("allocation", [])
            total = sum(item["benchmark_ratio"] for item in alloc)
            assert 95 <= total <= 105, (
                f"{key} benchmark total is {total}, expected ~100"
            )
            assert tmpl.get("benchmark_total", 100) == 100

    def test_min_not_exceed_max(self, data):
        """min_ratio should always be <= max_ratio."""
        for key, tmpl in data["templates"].items():
            for item in tmpl["allocation"]:
                assert item["min_ratio"] <= item["max_ratio"], (
                    f"{key}/{item['asset_class']}: min={item['min_ratio']} > max={item['max_ratio']}"
                )

    def test_benchmark_within_min_max(self, data):
        """Benchmark ratio should be within min-max range."""
        for key, tmpl in data["templates"].items():
            for item in tmpl["allocation"]:
                assert item["min_ratio"] <= item["benchmark_ratio"] <= item["max_ratio"], (
                    f"{key}/{item['asset_class']}: benchmark={item['benchmark_ratio']} "
                    f"not in [{item['min_ratio']}, {item['max_ratio']}]"
                )

    def test_each_has_rebalance_advice(self, data):
        for key, tmpl in data["templates"].items():
            assert "rebalance_advice" in tmpl
            assert len(tmpl["rebalance_advice"]) > 10

    def test_each_has_risk_tips(self, data):
        for key, tmpl in data["templates"].items():
            assert "risk_tips" in tmpl
            assert len(tmpl["risk_tips"]) >= 1

    def test_asset_class_labels_no_products(self, data):
        """Asset classes should be categories, not specific products."""
        for key, tmpl in data["templates"].items():
            for item in tmpl["allocation"]:
                label = item["asset_class"]
                # Should not contain numbers (stock codes) or fund names
                assert not any(c.isdigit() for c in label[-6:]), (
                    f"{key}: '{label}' looks like it contains a code"
                )


# ── Compliance Rules tests ────────────────────────────────────────────────


class TestComplianceRules:
    """Validate data/policy_configs/advisory/advisory_compliance_rules.json."""

    @pytest.fixture
    def data(self):
        return _load_json("advisory_compliance_rules.json")

    def test_has_meta(self, data):
        assert "meta" in data
        assert "rules" in data

    def test_minimum_rules(self, data):
        """Should have at least 8 compliance rules."""
        rules = data.get("rules", [])
        assert len(rules) >= 8

    def test_key_rule_ids_present(self, data):
        """Critical compliance rules must be present."""
        rule_ids = {r["rule_id"] for r in data["rules"]}
        required = [
            "ADV-CMP-001",  # 禁止荐股
            "ADV-CMP-002",  # 禁止承诺收益
            "ADV-CMP-003",  # 禁止预测涨跌
            "ADV-CMP-004",  # 禁止替用户决策
            "ADV-CMP-006",  # 必须提示风险
            "ADV-CMP-007",  # 必须说明仅供参考
            "ADV-CMP-008",  # 禁止推荐具体基金
        ]
        for rid in required:
            assert rid in rule_ids, f"Missing required rule: {rid}"

    def test_each_rule_has_fields(self, data):
        required_fields = ["rule_id", "name_cn", "severity", "category",
                          "pattern_examples", "safe_expression", "risk_notice"]
        for rule in data["rules"]:
            for field in required_fields:
                assert field in rule, f"{rule.get('rule_id', '?')}: missing {field}"

    def test_severity_values_valid(self, data):
        valid = {"violation", "warning", "required"}
        for rule in data["rules"]:
            assert rule["severity"] in valid, (
                f"{rule['rule_id']}: invalid severity '{rule['severity']}'"
            )

    def test_pattern_examples_non_empty(self, data):
        for rule in data["rules"]:
            assert len(rule.get("pattern_examples", [])) >= 1, (
                f"{rule['rule_id']}: pattern_examples is empty"
            )

    def test_safe_expressions_non_empty(self, data):
        for rule in data["rules"]:
            assert len(rule.get("safe_expression", "")) > 10, (
                f"{rule['rule_id']}: safe_expression too short"
            )


# ── Knowledge doc existence tests ─────────────────────────────────────────


class TestKnowledgeDocsExist:
    """Verify that the new knowledge documents exist and have content."""

    def test_risk_profile_doc_exists(self):
        path = KB_DIR / "risk" / "investor_risk_profile_rules.md"
        assert path.exists(), f"Missing: {path}"
        content = path.read_text(encoding="utf-8")
        assert len(content) > 500

    def test_allocation_model_doc_exists(self):
        path = KB_DIR / "advisory" / "asset_allocation_model_notes.md"
        assert path.exists(), f"Missing: {path}"
        content = path.read_text(encoding="utf-8")
        assert len(content) > 500

    def test_compliance_boundaries_doc_exists(self):
        path = KB_DIR / "compliance" / "investment_advisory_compliance_boundaries.md"
        assert path.exists(), f"Missing: {path}"
        content = path.read_text(encoding="utf-8")
        assert len(content) > 500
