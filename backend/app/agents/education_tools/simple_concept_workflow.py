"""
SimpleConceptWorkflow (简单概念投教工作流)

Composite deterministic tool that bundles the common education pipeline
for simple concept explanation questions into a SINGLE tool call.

This reduces LLM round-trips from ~4 to 1 for the most common
education scenario (e.g. "什么是XX？").

Internal chain:
  learner_profile_analyzer → concept_explainer →
  education_evidence_builder → education_compliance_policy →
  draft 7-section report skeleton

ALL deterministic — no LLM calls.
"""
from __future__ import annotations

from app.agents.education_tools.learner_profile_analyzer import analyze as analyze_profile
from app.agents.education_tools.concept_explainer import explain as explain_concept
from app.agents.education_tools.education_evidence_builder import build as build_evidence
from app.agents.education_tools.education_compliance_policy import review as review_compliance


_RISK_NOTICE = (
    "本内容仅供金融知识学习和投资者教育参考，不构成任何投资建议、"
    "产品推荐或投资决策依据。投资有风险，入市需谨慎。"
    "请根据自身风险承受能力独立判断，必要时咨询专业投资顾问。"
)

_LEARNING_NOTE = (
    "建议先从基础概念入手，逐步深入学习。如需了解更具体的投资产品，"
    "可进一步咨询，但请注意：投教智能体不推荐具体产品。"
)


def run_simple_concept_workflow(
    *,
    question: str,
    user_profile: dict | None = None,
    sources: list | None = None,
    **kwargs,
) -> dict:
    """Run the full simple-concept education workflow in one deterministic pass.

    Args:
        question: User's question (e.g. "什么是指数基金？")
        user_profile: Optional user profile dict.
        sources: Optional RAG sources list.

    Returns:
        dict with workflow_type, learner_profile, concepts, evidence,
        compliance, draft_sections, tool_chain, risk_notice.
    """
    # ── Step 1: Profile analysis ───────────────────────────────
    learner_profile = analyze_profile(question=question, user_profile=user_profile)

    # ── Step 2: Concept explanation ────────────────────────────
    concepts = explain_concept(question=question, learner_profile=learner_profile)

    # ── Step 3: Evidence building ──────────────────────────────
    evidence = build_evidence(sources=sources or [])

    # ── Step 4: Build draft sections ───────────────────────────
    draft_sections = _build_draft_sections(question, learner_profile, concepts, evidence)

    # ── Step 5: Compliance review of draft ─────────────────────
    draft_text = _sections_to_text(draft_sections)
    compliance = review_compliance(answer=draft_text)

    return {
        "workflow_type": "simple_concept",
        "learner_profile": learner_profile,
        "concepts": concepts,
        "evidence": evidence,
        "draft_sections": draft_sections,
        "compliance": compliance,
        "risk_notice": _RISK_NOTICE,
        "tool_chain": [
            "learner_profile_analyzer",
            "concept_explainer",
            "education_evidence_builder",
            "education_compliance_policy",
        ],
    }


def _build_draft_sections(
    question: str,
    learner_profile: dict,
    concepts: dict,
    evidence: dict,
) -> dict[str, str]:
    """Build the 7-section education report skeleton from tool outputs."""

    knowledge_level = learner_profile.get("knowledge_level", "beginner")
    learning_goal = learner_profile.get("learning_goal", "概念解释")
    preferred_style = learner_profile.get("preferred_style", "通俗解释")

    # Section 1: 问题理解与学习目标
    section_1 = (
        f"用户咨询：「{question}」\n"
        f"知识水平：{knowledge_level}｜学习目标：{learning_goal}｜偏好：{preferred_style}\n"
        f"该问题属于简单金融概念解释，目标帮助用户建立正确的认知框架。"
    )

    # Section 2: 用户知识水平判断
    level_descriptions = {
        "beginner": "初学者，需要最通俗易懂的解释，避免术语堆砌。",
        "basic": "有一定基础认知，可用比喻和简单类比。",
        "intermediate": "有一定投资经验，可引入适度专业概念。",
        "advanced": "专业知识背景，可用术语和深入分析。",
    }
    level_desc = level_descriptions.get(knowledge_level, level_descriptions["beginner"])
    section_2 = f"知识水平：{knowledge_level}\n{level_desc}"

    # Section 3: 核心概念通俗解释
    concept_items = concepts.get("concepts", [])
    if concept_items:
        lines = []
        for c in concept_items:
            lines.append(f"**{c.get('concept', '金融概念')}**：{c.get('explanation', '')}")
            for kp in c.get("key_points", []):
                lines.append(f"- {kp}")
        section_3 = "\n".join(lines)
    else:
        section_3 = f"关于「{question}」的基本概念解释：建议从最基础的金融知识入手，逐步建立理解。"

    # Section 4: 关键风险与常见误区
    if concept_items:
        lines = []
        for c in concept_items:
            misunderstandings = c.get("common_misunderstandings", [])
            for m in misunderstandings:
                lines.append(f"- {m}")
        if lines:
            section_4 = "\n".join(lines)
        else:
            section_4 = "当前知识库未提供具体误区信息。建议关注：概念混淆、以偏概全、忽视风险等常见问题。"
    else:
        section_4 = "当前知识库未提供具体误区信息。建议关注：概念混淆、以偏概全、忽视风险等常见问题。"

    # Section 5: 学习路径建议
    section_5 = (
        f"1. 先理解当前知识点的基本概念和核心要点。\n"
        f"2. 再了解相关产品或资产类别的实际运作方式。\n"
        f"3. 逐步学习风险管理和资产配置原则。\n"
        f"{_LEARNING_NOTE}"
    )

    # Section 6: 参考依据与延伸阅读
    evidence_items = evidence.get("items", [])
    if evidence_items:
        lines = ["参考资料来源："]
        for item in evidence_items[:5]:
            lines.append(f"- {item.get('title', '参考资料')}")
        section_6 = "\n".join(lines)
    else:
        section_6 = (
            "当前知识库暂无直接相关参考资料。"
            "建议查阅证监会、基金业协会等权威机构发布的投资者教育材料。"
        )

    # Section 7: 风险提示与适用边界
    section_7 = _RISK_NOTICE

    return {
        "一、问题理解与学习目标": section_1,
        "二、用户知识水平判断": section_2,
        "三、核心概念通俗解释": section_3,
        "四、关键风险与常见误区": section_4,
        "五、学习路径建议": section_5,
        "六、参考依据与延伸阅读": section_6,
        "七、风险提示与适用边界": section_7,
    }


def _sections_to_text(sections: dict[str, str]) -> str:
    """Convert draft_sections dict to a flat text for compliance review."""
    lines = []
    for title in [
        "一、问题理解与学习目标",
        "二、用户知识水平判断",
        "三、核心概念通俗解释",
        "四、关键风险与常见误区",
        "五、学习路径建议",
        "六、参考依据与延伸阅读",
        "七、风险提示与适用边界",
    ]:
        lines.append(title)
        lines.append(sections.get(title, "当前材料未提供充分信息。"))
        lines.append("")
    return "\n".join(lines)
