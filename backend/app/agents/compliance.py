from app.agents.base import FinancialAgent
from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Source


class ComplianceAgent(FinancialAgent):
    name = "regulatory_compliance"
    intent = "compliance"

    def answer(self, request: ConsultationRequest, sources: list[Source]) -> ConsultationResponse:
        return ConsultationResponse(
            intent=self.intent,
            agent=self.name,
            answer="已识别为监管合规咨询。MVP 阶段将基于法规材料输出解释、自查项和合规风险。",
            risk_notice="合规解读仅供参考，具体业务应咨询具备资质的专业机构。",
            sources=sources,
            warnings=[],
        )

