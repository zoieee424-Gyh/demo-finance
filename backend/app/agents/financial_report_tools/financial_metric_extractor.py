"""
FinancialMetricExtractor（财务指标提取器）

Extracts key financial metrics from Chinese financial text using regex.
Rule-based — no LLM.

Supports:
  - Chinese numeric units: 亿元, 万元, %, 同比增长, 环比
  - Fallback: returns None / "unknown" when not found
"""

from __future__ import annotations

import re


def _parse_number(text: str) -> float | None:
    """Parse a numeric value from Chinese financial text.

    Handles: "120亿元", "18.5万元", "62%", "同比增长15%", "-3亿元"
    """
    # Match numeric patterns with optional unit
    patterns = [
        (r"(-?[\d,]+\.?\d*)\s*亿\s*[元人]", 100_000_000),  # 亿元
        (r"(-?[\d,]+\.?\d*)\s*万\s*[元人]", 10_000),       # 万元
        (r"(-?[\d,]+\.?\d*)%", 0.01),                       # 百分比
        (r"(-?[\d,]+\.?\d*)\s*[元]", 1),                     # 元
    ]
    for pat, multiplier in patterns:
        m = re.search(pat, text)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                return float(raw) * multiplier
            except ValueError:
                continue
    return None


def _parse_ratio(text: str) -> float | None:
    """Parse a percentage value, e.g. '62%' → 0.62"""
    m = re.search(r"(-?[\d,]+\.?\d*)\s*%", text)
    if m:
        try:
            return float(m.group(1).replace(",", "")) / 100.0
        except ValueError:
            pass
    return None


def _parse_growth(text: str) -> str | None:
    """Parse YoY growth, e.g. '同比增长15%' → '+15%'"""
    m = re.search(r"同比[增长加]*\s*(-?[\d,]+\.?\d*)\s*%", text)
    if m:
        try:
            raw_val = m.group(1).replace(",", "")
            val = float(raw_val)
            # Format without .0 for integer values
            if val == int(val):
                return f"{'+' if val >= 0 else ''}{int(val)}%"
            return f"{'+' if val >= 0 else ''}{val}%"
        except ValueError:
            pass
    return None


def extract_metrics(
    *,
    financial_text: str = "",
    **kwargs,
) -> dict:
    """Extract financial metrics from text.

    Args:
        financial_text: Chinese financial report text or summary.

    Returns:
        dict with revenue, net_profit, gross_margin, net_margin,
        debt_ratio, operating_cash_flow, accounts_receivable,
        inventory, goodwill, and parsed growth rates.
    """
    text = financial_text or ""

    # ── Extract individual metrics ────────────────────────────
    revenue = _parse_number(
        _extract_context(text, r"(?:营业[总]?收入|营收|营业收入)[约达为]?\s*")
    )
    net_profit = _parse_number(
        _extract_context(text, r"(?:净利润?|归[属于母]?净利润?)[约达为]?\s*")
    )
    debt_ratio = _parse_ratio(
        _extract_context(text, r"(?:资产负债率|负债率|总负债[率]?)[约达为]?\s*")
    )
    operating_cash_flow = _parse_number(
        _extract_context(text, r"(?:经营[活动性]?现金流[量净额]?)[约为达]?\s*")
    )
    accounts_receivable = _parse_number(
        _extract_context(text, r"(?:应收账款|应收)[约为余额达]?\s*")
    )
    inventory = _parse_number(
        _extract_context(text, r"(?:存货[余额]?|库存[余额]?)[约为达]?\s*")
    )
    goodwill = _parse_number(
        _extract_context(text, r"(?:商誉[余额]?)[约为达]?\s*")
    )
    gross_margin = _parse_ratio(
        _extract_context(text, r"(?:毛[利]率)[约为达]?\s*")
    )
    net_margin = _parse_ratio(
        _extract_context(text, r"(?:净利[润]率)[约为达]?\s*")
    )

    # ── Growth rates ──────────────────────────────────────────
    revenue_growth = _parse_growth(
        _extract_context(text, r"营业[总]?收入", suffix_len=30)
    )
    profit_growth = _parse_growth(
        _extract_context(text, r"净利润?", suffix_len=30)
    )

    return {
        "revenue": revenue,
        "revenue_unit": "yuan",
        "revenue_growth": revenue_growth,
        "net_profit": net_profit,
        "net_profit_unit": "yuan",
        "profit_growth": profit_growth,
        "gross_margin": gross_margin,
        "net_margin": net_margin,
        "debt_ratio": debt_ratio,
        "operating_cash_flow": operating_cash_flow,
        "operating_cash_flow_unit": "yuan",
        "accounts_receivable": accounts_receivable,
        "accounts_receivable_unit": "yuan",
        "inventory": inventory,
        "inventory_unit": "yuan",
        "goodwill": goodwill,
        "goodwill_unit": "yuan",
    }


def _extract_context(
    text: str,
    prefix: str,
    suffix_len: int = 40,
) -> str:
    """Extract a context window around a regex prefix match."""
    m = re.search(prefix, text)
    if m:
        start = m.end()
        return text[start:start + suffix_len]
    return text
