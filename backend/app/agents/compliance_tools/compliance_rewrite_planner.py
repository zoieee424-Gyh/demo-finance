"""
ComplianceRewritePlanner（合规整改建议生成器）

Generates remediation actions and alternative phrasings.
Rule-based — no LLM.
Does NOT output formal legal opinions.
"""

_FORBIDDEN_TERMS = {"买入", "卖出", "加仓", "减仓", "满仓", "空仓", "目标价"}

def plan(*, flags_result: dict | None = None, suitability_result: dict | None = None,
         disclosure_result: dict | None = None, basis_result: dict | None = None, **kwargs) -> dict:
    actions: list[str] = []
    alternatives: list[dict] = []

    flags = (flags_result or {}).get("prohibited_flags", [])

    for f in flags:
        expr = f.get("expression", "")
        cat = f.get("category", "")
        if cat == "个股推荐":
            actions.append("删除全部个股推荐表述，替换为通用资产类别说明。")
            alternatives.append({"original": expr, "suggestion": "可改为'关注相关行业板块'或'宽基指数类资产'"})
        elif cat == "目标价预测":
            actions.append("删除目标价表述，不提供确定性价格预测。")
            alternatives.append({"original": expr, "suggestion": "可改为'基于历史估值区间分析，当前价格处于[X]分位'"})
        elif cat == "收益承诺":
            actions.append("删除收益承诺表述，不承诺收益或暗示无风险。")
            alternatives.append({"original": expr, "suggestion": "可补充'投资有风险，过往业绩不代表未来表现'"})
        elif cat == "交易指令":
            actions.append("删除具体交易操作指令，改为风险管理建议语气。")
            alternatives.append({"original": expr, "suggestion": "可改为'可考虑定期复核组合结构，关注资产配置比例是否偏离目标'"})
        elif cat == "夸大宣传":
            actions.append("删除夸大/绝对化宣传表述，改为中性客观描述。")
            alternatives.append({"original": expr, "suggestion": "可改为基于数据和事实的客观描述"})
        elif cat == "股票代码":
            actions.append("删除具体股票代码，不指向具体证券。")
            alternatives.append({"original": expr, "suggestion": "可改为行业分类或资产类别名称"})

    # Disclosure-based actions
    missing = (disclosure_result or {}).get("missing_disclosures", [])
    if "risk_disclaimer" in missing:
        actions.append("补充风险提示：'投资有风险，入市需谨慎。'")
    if "no_investment_advice" in missing:
        actions.append("补充声明：'本内容不构成投资建议。'")
    if "data_source" in missing:
        actions.append("补充数据来源和依据说明。")
    if "past_performance" in missing:
        actions.append("补充声明：'过往业绩不代表未来表现。'")

    # Suitability-based actions
    if (suitability_result or {}).get("suitability_level") in ("high_risk", "elevated"):
        actions.append("建议评估目标受众适当性，必要时增加风险揭示和投资者适当性声明。")

    if not actions:
        actions.append("当前未发现明显合规整改项，建议定期复核合规状态。")

    # Safety check
    for action in actions:
        for term in _FORBIDDEN_TERMS:
            if term in action:
                raise RuntimeError(f"Forbidden term '{term}' in remediation action")

    return {"remediation_actions": actions, "alternative_phrasings": alternatives}
