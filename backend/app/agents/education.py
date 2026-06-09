from app.agents.base import FinancialAgent
from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Source


class EducationAgent(FinancialAgent):
    name = "financial_education"
    intent = "education"

    def answer(self, request: ConsultationRequest, sources: list[Source]) -> ConsultationResponse:
        return ConsultationResponse(
            intent=self.intent,
            agent=self.name,
            answer="已识别为金融科普任务。MVP 阶段将根据用户知识水平输出通俗化投教内容。",
            risk_notice="投教内容仅用于知识学习，不构成投资建议。",
            sources=sources,
            warnings=[],
        )

