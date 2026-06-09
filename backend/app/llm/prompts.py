"""
LLM Prompt Templates（提示词模板）

Defines prompt templates for the two permitted LLM use cases:
  1. Profile enrichment — natural language completion of missing profile fields.
  2. Advisory report rewrite — polishing the final report language.

All prompts enforce financial compliance boundaries:
  - No stock tipping (不荐股)
  - No price prediction (不预测涨跌)
  - No return promise (不承诺收益)
  - No investment decision substitution (不替用户做投资决策)
  - Do not modify structured facts, ratios, or risk levels.
"""

from __future__ import annotations

# ── Profile Enrichment Prompt ────────────────────────────────────
# Used by: ProfileAnalyzer.enrich_profile_with_llm()
# Purpose: Suggest completions for missing profile fields based on
#   the user's natural language question.  The LLM output is treated
#   as a SUGGESTION only — the rule engine makes the final decision.

PROFILE_ENRICHMENT_PROMPT = """\
不要思考，不要解释，只输出一行JSON。你是金融用户画像分析助手，根据用户自然语言问题推断缺失的画像字段。

## 严格禁止
- 不得推荐具体股票、基金代码、基金名称或交易平台。
- 不得预测市场涨跌。
- 不得承诺任何投资收益。
- 不得替用户做出投资决策。
- 只能基于用户已说出的信息推断，不得凭空编造。
- 不得推断 risk_preference，该字段必须由用户明确表达，统一填 "unknown"。

## 当前已知画像
{current_profile}

## 缺失字段（仅补全这些）
{missing_fields}

## 用户原始问题
{user_question}

## 只输出这行 JSON，不要 markdown，不要思考过程：
{{"income_level":"low|medium|high|unknown","investment_experience":"beginner|experienced|unknown","liquidity_need":"low|medium|high|unknown","investment_horizon":"short|medium|long|unknown","constraints":[],"risk_preference":"unknown"}}"""


# ── Advisory Report Rewrite Prompt ───────────────────────────────
# Used by: InvestmentAdvisorAgent.rewrite_report_with_llm()
# Purpose: Polish the language of the final report WITHOUT changing
#   any structured facts, ratios, risk levels, or conclusions.

ADVISORY_REPORT_REWRITE_PROMPT = """\
不要思考，不要解释，直接输出润色后的完整报告。你是金融投顾报告的润色助手，优化以下报告的语言表达，使其更流畅、更易读。

## 严格禁止
- 不得新增任何投资建议、结论或观点。
- 不得推荐具体股票、基金代码、基金名称或交易平台。
- 不得预测市场涨跌。
- 不得承诺任何投资收益。
- 不得替用户做出投资决策。
- 不得修改以下任何数据：资产配置比例、风险等级、投资期限、金额比例。
- 不得修改报告的章节结构（一、二、三...九）。
- 不得修改任何数字、百分比或数值型内容。
- 不得删除风险提示内容。

## 允许的操作
- 优化措辞，使表达更流畅自然。
- 调整语序，使逻辑更清晰。
- 统一术语表达。
- 修正错别字和语法错误（如果有）。

## 原始报告
{original_report}

## 输出要求
直接输出润色后的完整报告，不要添加任何解释、前言或后缀。
输出必须保持原报告的九个章节结构不变。
"""


# ── Compliance notice embedded in every prompt ───────────────────

_COMPLIANCE_FOOTER = """
---
重要提醒：你是一个金融辅助工具，不是持牌投资顾问。你的所有输出仅供参考，不构成投资决策依据。
"""


def build_profile_prompt(current_profile: str, missing_fields: str, user_question: str) -> str:
    """Build the profile enrichment prompt with filled placeholders."""
    return PROFILE_ENRICHMENT_PROMPT.format(
        current_profile=current_profile,
        missing_fields=missing_fields,
        user_question=user_question,
    )


def build_report_rewrite_prompt(original_report: str) -> str:
    """Build the advisory report rewrite prompt with the original report."""
    return ADVISORY_REPORT_REWRITE_PROMPT.format(
        original_report=original_report,
    )
