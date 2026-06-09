from app.agents.base import FinancialAgent
from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Source


class RiskControlAgent(FinancialAgent):
    name = "risk_control_reviewer"
    intent = "risk_control"

    def answer(self, request: ConsultationRequest, sources: list[Source]) -> ConsultationResponse:
        return ConsultationResponse(
            intent=self.intent,
            agent=self.name,
            answer="已识别为风控审查任务。MVP 阶段将输出政策、市场、财务和行为风险维度。",
            risk_notice="风险评估结果仅作辅助参考，不代表确定性判断。",
            sources=sources,
            warnings=[],
        )

