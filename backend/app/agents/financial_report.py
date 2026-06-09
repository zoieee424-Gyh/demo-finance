from app.agents.base import FinancialAgent
from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Source


class FinancialReportAgent(FinancialAgent):
    name = "financial_report_analyst"
    intent = "financial_report"

    def answer(self, request: ConsultationRequest, sources: list[Source]) -> ConsultationResponse:
        return ConsultationResponse(
            intent=self.intent,
            agent=self.name,
            answer="已识别为财报分析任务。MVP 阶段将提取关键指标、同比环比、现金流和风险提示。",
            risk_notice="财报分析仅供研究参考，不构成投资建议或收益承诺。",
            sources=sources,
            warnings=[],
        )

