"""
ProhibitedExpressionDetector（违规话术检测器）

Detects regulated/prohibited financial expressions.
Rule-based — no LLM.
"""

import re

_RULES: list[tuple[str, str, str]] = [
    # (regex, severity, category)
    (r"推荐买入|建议买入|强烈推荐|建议购买", "high", "个股推荐"),
    (r"建议卖出|建议清仓|尽快卖出", "high", "个股推荐"),
    (r"目标价\s*\d+", "high", "目标价预测"),
    (r"保本|稳赚不赔|无风险|绝对安全|稳赚", "high", "收益承诺"),
    (r"保证收益|保证年化|稳赚", "high", "收益承诺"),
    (r"(一定|肯定|势必|必然|百分百)(?:涨|跌|上涨|下跌|赚钱)", "high", "确定性预测"),
    (r"加仓|减仓|满仓|空仓|建仓|平仓", "medium", "交易指令"),
    (r"买入评级|卖出评级|增持评级|减持评级", "high", "投资评级"),
    (r"年化收益[率]?\s*\d+%", "medium", "收益暗示"),
    (r"史上最|最高|绝对第一|无人能及", "medium", "夸大宣传"),
    (r"赶快|马上|立刻|现在就|错过就", "medium", "诱导性营销"),
    (r"(?:60\d{4}|00\d{4}|30\d{4}|68\d{8})", "high", "股票代码"),
]

def detect(*, review_content: str = "", **kwargs) -> dict:
    text = review_content or ""
    flags: list[dict] = []
    high_count = 0

    for pattern, severity, category in _RULES:
        for m in re.finditer(pattern, text):
            flags.append({
                "expression": m.group(0),
                "severity": severity,
                "category": category,
            })
            if severity == "high":
                high_count += 1

    overall = "high" if high_count >= 2 else ("medium" if high_count >= 1 or len(flags) >= 3 else "low")
    return {
        "prohibited_flags": flags,
        "overall_severity": overall,
        "high_count": high_count,
        "total_flags": len(flags),
        "matched_expressions": [f["expression"] for f in flags],
    }
