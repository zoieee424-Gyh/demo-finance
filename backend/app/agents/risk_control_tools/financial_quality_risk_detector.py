"""
FinancialQualityRiskDetector（财务质量风险识别器）

Detects financial quality risks from financial report text.
Rule-based — no LLM.
"""

from __future__ import annotations

import re


def detect(*, financial_text: str = "", **kwargs) -> dict:
    """Detect financial quality risks from text.

    Args:
        financial_text: Financial report text/summary.

    Returns:
        dict with financial_quality_level, risk_flags, evidence_points.
    """
    text = financial_text or ""
    risk_flags: list[str] = []
    evidence: list[str] = []

    # ── Check 1: Profit up but OCF negative ────────────────────
    has_profit_growth = bool(re.search(r"净利润?[增涨长]*[约达]?\s*-?\d+[%％]", text))
    has_negative_ocf = bool(re.search(r"经营[活动性]?现金流[量为净]?\s*-[\d.]+", text)) or \
                       "经营现金流为负" in text or "-3亿元" in text

    if has_profit_growth and has_negative_ocf:
        risk_flags.append(
            "净利润增长但经营现金流为负：利润质量存在隐忧，"
            "可能存在应收账款大幅增加或收入确认偏激进的情况。"
        )
        evidence.append("净利润与经营现金流背离")

    # ── Check 2: High debt ratio ───────────────────────────────
    debt_match = re.search(r"资产负债率[约为]?\s*(-?[\d.]+)\s*%", text)
    if debt_match:
        debt_val = float(debt_match.group(1))
        if debt_val > 70:
            risk_flags.append(
                f"资产负债率{debt_val}%处于高位，超过70%警戒线，"
                "财务杠杆偏高，偿债能力需关注。"
            )
            evidence.append(f"资产负债率{debt_val}%")

    # ── Check 3: AR spike ──────────────────────────────────────
    ar_match = re.search(r"应收[账款][约为余额]*\s*(-?[\d,]+\.?\d*)\s*(?:亿|万)?元?", text)
    rev_match = re.search(r"营业[总]?收入[约为]*\s*(-?[\d,]+\.?\d*)\s*(?:亿|万)?元?", text)
    if ar_match and rev_match:
        # Both found — qualitative flag
        risk_flags.append(
            "应收账款规模需关注：若占营收比例偏高，可能影响回款速度和坏账计提。"
        )
        evidence.append("应收账款需关注")

    # ── Check 4: Inventory high ────────────────────────────────
    inv_match = re.search(r"存货[余额约为]*\s*(-?[\d,]+\.?\d*)\s*(?:亿|万)?元?", text)
    if inv_match:
        risk_flags.append(
            "存货规模需关注：若占营收比例偏高，存在存货跌价风险。"
        )
        evidence.append("存货需关注")

    # ── Check 5: Goodwill risk ─────────────────────────────────
    gw_match = re.search(r"商誉[余额约为]*\s*(-?[\d,]+\.?\d*)\s*(?:亿|万)?元?", text)
    if gw_match:
        gw_val = float(gw_match.group(1).replace(",", ""))
        # Look for net profit to compare
        np_match = re.search(r"净利润?[约为]*\s*(-?[\d,]+\.?\d*)\s*(?:亿|万)?元?", text)
        if np_match:
            np_val = float(np_match.group(1).replace(",", ""))
            if np_val > 0 and gw_val > np_val * 2:
                risk_flags.append(
                    "商誉规模显著高于净利润，若被并购标的业绩不达预期，"
                    "存在商誉减值风险。"
                )
                evidence.append("商誉减值风险")

    # ── Determine level ────────────────────────────────────────
    if len(risk_flags) >= 3:
        level = "high_risk"
    elif len(risk_flags) >= 1:
        level = "elevated_risk"
    else:
        level = "no_signal"

    return {
        "financial_quality_level": level,
        "risk_flags": risk_flags,
        "evidence_points": evidence,
    }
