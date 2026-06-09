"""
FinancialTextParser（财报文本解析器）

Parses user question and optional financial_text into a structured
financial report context.  Rule-based — no LLM.

Output:
  {
    "company_name": str | None,
    "report_period": str | None,
    "report_type": str | None,
    "extracted_sections": list[str],
    "missing_fields": list[str]
  }
"""

from __future__ import annotations

import re


def parse(
    *,
    question: str,
    financial_text: str | None = None,
    **kwargs,
) -> dict:
    """Parse financial report text from user input.

    Args:
        question: User's question text.
        financial_text: Optional financial report text/summary.

    Returns:
        dict with company_name, report_period, report_type,
        extracted_sections, missing_fields.
    """
    text = (financial_text or "") + " " + question

    result: dict = {
        "company_name": None,
        "report_period": None,
        "report_type": None,
        "extracted_sections": [],
        "missing_fields": [],
    }

    # ── Company name detection ────────────────────────────────
    company_patterns = [
        (r"([\w一-鿿]{2,12}(?:公司|集团|股份|科技|控股|有限))", "company"),
        (r"(某公司|某集团|本公司|该公司)", "generic_company"),
    ]
    for pat, _ in company_patterns:
        m = re.search(pat, text)
        if m:
            result["company_name"] = m.group(1)
            break

    # ── Report period detection ───────────────────────────────
    period_patterns = [
        (r"(20\d{2})\s*年(?:度|半年度|一季度|二季度|三季度|四季度|中期|年报)", "annual"),
        (r"(20\d{2})\s*年年报", "annual"),
        (r"(20\d{2})\s*Q[1-4]", "quarterly"),
        (r"(20\d{2}年(?:第[一二三四]季|Q[1-4]))", "quarterly"),
        (r"(20\d{2})年(?!\S*[天月日])", "year"),
    ]
    for pat, ptype in period_patterns:
        m = re.search(pat, text)
        if m:
            result["report_period"] = m.group(1)
            if "年报" in m.group(0) or "年度" in m.group(0):
                result["report_type"] = "annual_report"
            elif "Q" in m.group(0) or "季" in m.group(0):
                result["report_type"] = "quarterly_report"
            elif "半年" in m.group(0):
                result["report_type"] = "semi_annual_report"
            else:
                result["report_type"] = "period_unspecified"
            break

    # ── Section detection ─────────────────────────────────────
    section_keywords = {
        "income_statement": ["营业", "收入", "利润", "成本", "毛利率", "净利率", "营收"],
        "balance_sheet": ["资产", "负债", "权益", "资产负债率", "净资产"],
        "cash_flow": ["现金流", "经营活动", "筹资活动", "投资活动"],
        "notes": ["附注", "说明", "会计政策"],
    }
    for section, keywords in section_keywords.items():
        if any(kw in text for kw in keywords):
            result["extracted_sections"].append(section)

    # ── Missing fields ────────────────────────────────────────
    if not result["company_name"]:
        result["missing_fields"].append("company_name")
    if not result["report_period"]:
        result["missing_fields"].append("report_period")
    if not result["report_type"]:
        result["missing_fields"].append("report_type")
    if not financial_text or len(financial_text.strip()) < 20:
        result["missing_fields"].append("detailed_financial_text")

    return result
