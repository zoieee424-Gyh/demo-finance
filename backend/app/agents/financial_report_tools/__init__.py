"""Financial report analysis tool modules (财报分析工具模块).

Seven deterministic tools for financial report analysis.
All tools are rule-based (requires_llm=False).
"""

from app.agents.financial_report_tools.financial_text_parser import parse
from app.agents.financial_report_tools.financial_metric_extractor import extract_metrics
from app.agents.financial_report_tools.profitability_analyzer import analyze_profitability
from app.agents.financial_report_tools.solvency_liquidity_analyzer import analyze_solvency
from app.agents.financial_report_tools.growth_efficiency_analyzer import analyze_growth
from app.agents.financial_report_tools.anomaly_risk_detector import detect_risks
from app.agents.financial_report_tools.financial_report_compliance_policy import review

__all__ = [
    "parse",
    "extract_metrics",
    "analyze_profitability",
    "analyze_solvency",
    "analyze_growth",
    "detect_risks",
    "review",
]
