"""
Tests for compliance validator context-aware logic.

Key principle: quoting violation expressions in a review report
is NOT a violation — the validator should distinguish between
"reviewing/quoted" violations and "model outputting" violations.
"""
from __future__ import annotations

import pytest


class TestComplianceValidatorReviewContext:

    def test_quoting_return_promise_does_not_fail(self):
        """合规报告引用'保证收益'作为审查对象——不应失败。"""
        from app.agents.deepagent.compliance_deepagent import validate_compliance_output

        answer = (
            "一、审查对象与场景说明\n审查营销话术「本产品保证年化收益8%」。\n"
            "二、合规风险等级\n高风险——被审查内容涉及收益承诺。\n"
            "三、违规或高风险表述识别\n"
            "检测到以下违规表述：\n"
            "- 违规示例：「保证年化收益8%」——承诺收益，违反适当性管理规定\n"
            "- 原文表述「保证收益」属于明确违规\n"
            "四、适用监管原则与依据\n"
            "根据《证券投资顾问业务暂行规定》，不得承诺收益。\n"
            "五、整改建议\n"
            "需要删除「保证年化收益8%」等承诺性表述。\n"
            "六、可替代表述示例\n"
            "应改为「历史业绩不代表未来表现」。\n"
            "七、审查边界与不确定性\n"
            "本次审查仅基于给定文本，未考虑完整合规流程。\n"
            "八、合规提示\n"
            "本报告仅供合规风险识别参考，不构成正式法律意见，"
            "请结合具体业务规则审慎判断。"
        )
        is_valid, failures = validate_compliance_output(answer)
        assert is_valid, f"Should be valid but got: {failures}"

    def test_quoting_buy_recommendation_does_not_fail(self):
        """合规报告引用'推荐买入'作为审查对象——不应失败。"""
        from app.agents.deepagent.compliance_deepagent import validate_compliance_output

        answer = (
            "一、审查对象与场景说明\n审查内容：「推荐买入某股票，目标价30元」。\n"
            "二、合规风险等级\n高风险。\n"
            "三、违规或高风险表述识别\n"
            "- 需删除「推荐买入」——构成投资建议\n"
            "- 需删除「目标价30元」——构成价格预测\n"
            "- 需删除「必涨」——构成确定性涨跌预测\n"
            "四、适用监管原则与依据\n"
            "根据法规，不得推荐个股或预测价格。\n"
            "五、整改建议\n"
            "将所有买入推荐和价格预测内容删除。\n"
            "六、可替代表述示例\n"
            "建议改为「投资者应基于自身判断做出决策」。\n"
            "七、审查边界与不确定性\n"
            "本次审查基于给定文本。\n"
            "八、合规提示\n"
            "本报告仅供合规风险识别参考，不构成正式法律意见，"
            "请结合具体业务规则审慎判断。"
        )
        is_valid, failures = validate_compliance_output(answer)
        assert is_valid, f"Should be valid but got: {failures}"

    def test_quoting_price_prediction_does_not_fail(self):
        """合规报告引用'必涨'、'目标价'作为审查对象——不应失败。"""
        from app.agents.deepagent.compliance_deepagent import validate_compliance_output

        answer = (
            "一、审查对象与场景说明\n审查「预计下个月必涨，目标价30元」。\n"
            "二、合规风险等级\n高风险。\n"
            "三、违规或高风险表述识别\n"
            "被审查内容中检测到以下问题：\n"
            "- 违规表述1：「必涨」——确定性涨跌预测\n"
            "- 违规表述2：「目标价30元」——价格预测\n"
            "四、适用监管原则与依据\n"
            "根据法规，不得预测涨跌。\n"
            "五、整改建议\n"
            "必须删除所有涨跌预测表述。\n"
            "六、可替代表述示例\n"
            "建议强调风险提示而非预测。\n"
            "七、审查边界与不确定性\n"
            "本次审查基于给定文本。\n"
            "八、合规提示\n"
            "本报告仅供合规风险识别参考，不构成正式法律意见，请审慎判断。"
        )
        is_valid, failures = validate_compliance_output(answer)
        assert is_valid, f"Should be valid but got: {failures}"

    def test_model_making_return_promise_fails(self):
        """模型自身建议'你可以这样保证收益宣传'——应失败。"""
        from app.agents.deepagent.compliance_deepagent import validate_compliance_output

        answer = (
            "一、审查对象与场景说明\n营销文案审查。\n"
            "二、合规风险等级\n低。\n"
            "三、违规或高风险表述识别\n无。\n"
            "四、适用监管原则与依据\n无。\n"
            "五、整改建议\n"
            "你可以这样写宣传语：「本产品保证年化收益8%」。"  # MODEL suggesting this!
            "六、可替代表述示例\n同上。\n"
            "七、审查边界与不确定性\n无。\n"
            "八、合规提示\n本报告仅供合规风险识别参考，不构成正式法律意见，审慎判断。"
        )
        is_valid, failures = validate_compliance_output(answer)
        # "保证年化收益8%" appears in section 五 (整改建议),
        # not in section 三 (违规识别). No review context keywords nearby.
        # This SHOULD fail.
        assert not is_valid, f"Should fail because model is suggesting violation wording"
        assert any("收益承诺" in f for f in failures), f"Expected 收益承诺 failure, got: {failures}"


class TestComplianceValidatorStructure:

    def test_missing_sections_fails(self):
        from app.agents.deepagent.compliance_deepagent import validate_compliance_output
        is_valid, failures = validate_compliance_output("只是一些合规分析。")
        assert not is_valid
        assert any("Missing required sections" in f for f in failures)

    def test_missing_disclaimer_fails(self):
        from app.agents.deepagent.compliance_deepagent import validate_compliance_output
        answer = (
            "一、审查对象与场景说明\nx\n"
            "二、合规风险等级\nx\n"
            "三、违规或高风险表述识别\nx\n"
            "四、适用监管原则与依据\nx\n"
            "五、整改建议\nx\n"
            "六、可替代表述示例\nx\n"
            "七、审查边界与不确定性\nx\n"
            "八、合规提示\nx"
        )
        is_valid, failures = validate_compliance_output(answer)
        assert not is_valid
        assert any("disclaimer" in f.lower() or "法律意见" in f for f in failures)

    def test_stock_code_fails(self):
        from app.agents.deepagent.compliance_deepagent import validate_compliance_output
        answer = (
            "一、审查对象与场景说明\n审查600519相关内容。\n"
            "二、合规风险等级\n高\n"
            "三、违规或高风险表述识别\n\n"
            "四、适用监管原则与依据\n\n"
            "五、整改建议\n\n"
            "六、可替代表述示例\n\n"
            "七、审查边界与不确定性\n\n"
            "八、合规提示\n"
            "本报告仅供合规风险识别参考，不构成正式法律意见，审慎判断。"
        )
        is_valid, failures = validate_compliance_output(answer)
        assert not is_valid
        assert any("股票代码" in f for f in failures)

    def test_complete_valid_report_passes(self):
        from app.agents.deepagent.compliance_deepagent import validate_compliance_output
        answer = (
            "一、审查对象与场景说明\n审查营销文案。\n"
            "二、合规风险等级\n中等风险。\n"
            "三、违规或高风险表述识别\n"
            "被审查内容包含「保证收益」表述，该表述需要整改。\n"
            "四、适用监管原则与依据\n"
            "根据投资者适当性管理规定，不得承诺收益。\n"
            "五、整改建议\n"
            "应删除「保证收益」表述，替换为风险提示。\n"
            "六、可替代表述示例\n"
            "建议修改为「历史业绩不代表未来表现」。\n"
            "七、审查边界与不确定性\n"
            "本次审查基于给定文本，未覆盖完整合规审查流程。\n"
            "八、合规提示\n"
            "本报告仅供合规风险识别参考，不构成正式法律意见，"
            "不替代律师、持牌机构或合规部门判断，请审慎判断。"
        )
        is_valid, failures = validate_compliance_output(answer)
        assert is_valid, f"Should be valid but got: {failures}"

    def test_absolute_legal_claim_fails(self):
        """模型声称'完全合规'——应失败。"""
        from app.agents.deepagent.compliance_deepagent import validate_compliance_output
        answer = (
            "一、审查对象与场景说明\nx\n"
            "二、合规风险等级\n该内容完全合规，无任何风险。\n"
            "三、违规或高风险表述识别\n无\n"
            "四、适用监管原则与依据\n无\n"
            "五、整改建议\n无\n"
            "六、可替代表述示例\n无\n"
            "七、审查边界与不确定性\n无\n"
            "八、合规提示\n"
            "本报告仅供合规风险识别参考，不构成正式法律意见，审慎判断。"
        )
        is_valid, failures = validate_compliance_output(answer)
        assert not is_valid
        assert any("绝对化合规" in f for f in failures)

    def test_absolute_legal_claim_in_review_context_passes(self):
        """审查报告中提示'不得宣称完全合规'——不应失败。"""
        from app.agents.deepagent.compliance_deepagent import validate_compliance_output
        answer = (
            "一、审查对象与场景说明\n审查营销文案。\n"
            "二、合规风险等级\n中等风险。\n"
            "三、违规或高风险表述识别\n未发现直接收益承诺。\n"
            "四、适用监管原则与依据\n营销材料不得使用绝对化合规结论。\n"
            "五、整改建议\n避免宣称完全合规，应改为提示仍需结合具体规则审慎判断。\n"
            "六、可替代表述示例\n建议写作：本材料已按内部流程初步审查。\n"
            "七、审查边界与不确定性\n本次审查基于有限文本。\n"
            "八、合规提示\n本报告仅供合规风险识别参考，不构成正式法律意见，请审慎判断。"
        )
        is_valid, failures = validate_compliance_output(answer)
        assert is_valid, f"Should be valid but got: {failures}"

    def test_output_review_status_passes(self):
        """工具状态'合规输出审查通过'——不等于承诺材料完全合规。"""
        from app.agents.deepagent.compliance_deepagent import validate_compliance_output
        answer = (
            "合规输出审查通过。\n"
            "一、审查对象与场景说明\n审查营销文案。\n"
            "二、合规风险等级\n低风险。\n"
            "三、违规或高风险表述识别\n未检测到明确违规表述。\n"
            "四、适用监管原则与依据\n遵循适当性与充分披露原则。\n"
            "五、整改建议\n继续保留风险提示。\n"
            "六、可替代表述示例\n过往业绩不代表未来表现。\n"
            "七、审查边界与不确定性\n不替代完整合规流程。\n"
            "八、合规提示\n本报告仅供合规风险识别参考，不构成正式法律意见，请审慎判断。"
        )
        is_valid, failures = validate_compliance_output(answer)
        assert is_valid, f"Should be valid but got: {failures}"


class TestComplianceRepair:

    def test_repair_adds_missing_sections(self):
        from app.agents.deepagent.compliance_deepagent import (
            repair_compliance_output,
            _REQUIRED_SECTIONS,
        )
        raw = "该内容存在合规风险。"
        repaired = repair_compliance_output(raw)
        for section in _REQUIRED_SECTIONS:
            assert section in repaired, f"Missing section: {section}"

    def test_repair_adds_compliance_notice(self):
        from app.agents.deepagent.compliance_deepagent import (
            repair_compliance_output,
            _COMPLIANCE_NOTICE,
        )
        raw = "合规分析结果。"
        repaired = repair_compliance_output(raw)
        assert _COMPLIANCE_NOTICE in repaired

    def test_repair_preserves_existing_content(self):
        from app.agents.deepagent.compliance_deepagent import repair_compliance_output
        raw = "一、审查对象与场景说明\n这是营销文案审查。\n二、合规风险等级\n高\n"
        repaired = repair_compliance_output(raw)
        assert "营销文案" in repaired


class TestReviewContextHelper:

    def test_review_context_detected_for_violation_section(self):
        from app.agents.deepagent.compliance_deepagent import _is_review_context
        text = "三、违规或高风险表述识别\n被审查内容包含「保证年化收益8%」，该表述需要整改。"
        pos = text.find("保证年化收益")
        assert _is_review_context(text, pos, pos + 8) is True

    def test_review_context_detected_for_remediation_with_quotes(self):
        from app.agents.deepagent.compliance_deepagent import _is_review_context
        text = "五、整改建议\n需要删除「推荐买入」等表述，改为风险提示。"
        pos = text.find("推荐买入")
        assert _is_review_context(text, pos, pos + 4) is True

    def test_not_review_context_for_suggestion(self):
        from app.agents.deepagent.compliance_deepagent import _is_review_context
        # Model making a suggestion NOT in review context
        text = "你可以这样写宣传语：「本产品保证年化收益8%」。"
        pos = text.find("保证年化收益")
        assert _is_review_context(text, pos, pos + 8) is False
