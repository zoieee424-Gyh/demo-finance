"""
ComplianceOutputPolicy（合规输出审查器）

Reviews compliance report output for violations.
Rule-based — no LLM.
"""

import re

def review(*, answer: str, **kwargs) -> dict:
    warnings: list[str] = []

    rules: list[tuple[str, str]] = [
        (r"完全合规|绝对合法|保证通过监管|肯定合规", "使用绝对化合规结论"),
        (r"建议买入|推荐买入|强烈推荐|建议购买", "投资推荐表述"),
        (r"目标价\s*\d+", "目标价预测"),
        (r"(一定|肯定|势必|必然)(?:涨|跌)", "确定性市场预测"),
        (r"保证收益|稳赚|年化收益\s*\d+%", "收益承诺"),
        (r"本律师|本法律顾问|法律意见书|以律师身份", "替代律师/法律顾问判断"),
        (r"(?:60\d{4}|00\d{4}|30\d{4}|68\d{8})", "股票代码"),
    ]

    for pattern, desc in rules:
        if re.search(pattern, answer):
            warnings.append(f"[违规] {desc}")

    risk_notice = (
        "本报告仅供合规风险识别参考，不构成正式法律意见，"
        "不替代律师、持牌机构或合规部门判断，"
        "请结合具体业务规则和适用法律法规审慎判断。"
        "合规审查基于有限输入信息，可能未覆盖全部监管要求。"
    )

    return {"warnings": warnings, "risk_notice": risk_notice, "is_compliant": len(warnings) == 0}
