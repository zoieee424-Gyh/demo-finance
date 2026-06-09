"""
ScamRiskDetector (金融诈骗风险识别器)

Detects common financial scam signals in user questions or content.
Used for fraud-prevention education — NOT for legal judgment.

Rule-based — no LLM.
"""

from __future__ import annotations

import re


# ── Scam signal patterns ──────────────────────────────────────────

_SCAM_PATTERNS: list[tuple[str, str, str]] = [
    # (regex, risk_type, description)
    (r"稳赚不赔|保本高收益|保本保息|无风险.*收益", "保本高收益承诺",
     "承诺保本且高收益是典型骗局特征。正规金融产品不承诺保本且高收益。"),
    (r"内幕消息|内部消息|独家消息|庄家消息", "内幕信息诱导",
     "声称有内幕消息或独家消息诱导投资，是常见的证券欺诈手法。利用内幕信息交易属于违法行为。"),
    (r"老师带单|跟着老师|跟单|跟老师|带单老师|群里跟单|群里.*荐股|群里.*推荐", "跟单/带单骗局",
     "老师带单/群里跟单是常见投资骗局模式。正规投资顾问不会在群内带领买卖操作。"),
    (r"荐股群|炒股群|群推荐|群里推荐|群里荐股|荐股.*群", "荐股群骗局",
     "通过社交群组推荐股票是常见的非法证券活动。合法投顾服务需持牌且遵守适当性管理规定。"),
    (r"充值返利|充值.*返|返利.*充值|充.*送", "充值返利骗局",
     "充值返利是常见金融诈骗手法，往往在获取信任后骗取大额资金。"),
    (r"虚假平台|假平台|黑平台|克隆平台|冒牌", "虚假交易平台",
     "虚假交易平台会伪造交易记录和盈利数据，用户投入资金后无法取出。请通过正规渠道验证平台资质。"),
    (r"高额回报|超高收益|短期.*倍|翻倍.*收益|月收益.*%", "高收益承诺",
     "承诺高额/超高收益是金融诈骗的典型特征。高收益必然伴随高风险，不存在低风险高收益产品。"),
    (r"包赚|包赢|必赚|只赚不赔|绝不亏损", "绝对盈利承诺",
     "承诺包赚/只赚不赔是典型的欺诈信号。任何投资都存在亏损可能。"),
    (r"拉人头|发展下线|多层分销|会员.*级|代理.*佣金", "传销/庞氏特征",
     "拉人头、多层返佣模式可能涉及传销或庞氏骗局。正规金融机构不以发展下线为盈利模式。"),
    (r"原始股|未上市|股权投资.*原始|认购.*原始股", "原始股骗局",
     "普通投资者很难获得真正的原始股。声称销售原始股的往往是骗局。"),
    (r"刷流水|刷单|任务.*收益|接单.*收益", "刷单/刷流水骗局",
     "刷单返利是常见电信诈骗手法，前期小额返利后诱导大额投入。"),
]

# ── Safe action recommendations ───────────────────────────────────

_SAFE_ACTIONS = [
    "核实机构资质：通过中国证监会、基金业协会、银保监会官网查询机构是否持牌。",
    "不轻信高收益承诺：任何保证稳赚不赔/低风险高收益的说法都是警示信号。",
    "不随意转账：不向陌生账户或老师/客服提供的个人账户转账。",
    "保留证据：保存聊天记录、转账凭证、合同文件，必要时向公安机关报案。",
    "通过正规渠道投资：选择银行、证券公司、基金公司等持牌机构的官方渠道。",
    "咨询专业人士：如有疑虑，咨询正规金融机构或律师。",
]


def detect(*, question: str, **kwargs) -> dict:
    """Detect scam signals in user's question or content.

    Args:
        question: User's question or content to analyze.

    Returns:
        dict with scam_signals, risk_level, warnings, safe_actions.
    """
    q = question or ""
    scam_signals: list[dict] = []

    for pattern, risk_type, description in _SCAM_PATTERNS:
        if re.search(pattern, q):
            scam_signals.append({
                "risk_type": risk_type,
                "description": description,
            })

    # Determine risk level
    if len(scam_signals) >= 3:
        risk_level = "high"
    elif len(scam_signals) >= 1:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Build warnings
    warnings: list[str] = []
    if scam_signals:
        warnings.append(
            "检测到疑似金融诈骗信号。请注意：以下内容为风险教育和防骗提示，"
            "不构成法律判断。如涉及实际财产损失，请及时向公安机关报案。"
        )
        for sig in scam_signals:
            warnings.append(f"[警告] {sig['risk_type']}: {sig['description']}")

    if not scam_signals:
        warnings.append(
            "当前问题中未检测到明显的金融诈骗信号。"
            "但请注意：投资有风险，任何投资决策都可能造成损失。"
        )

    return {
        "scam_signals": scam_signals,
        "signal_count": len(scam_signals),
        "risk_level": risk_level,
        "warnings": warnings,
        "safe_actions": _SAFE_ACTIONS,
        "education_note": "以上为金融安全和防骗教育内容，不构成法律意见。",
    }
