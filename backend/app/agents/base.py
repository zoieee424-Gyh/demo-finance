from abc import ABC, abstractmethod

from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Intent, Source


class FinancialAgent(ABC):
    name: str
    intent: Intent

    @abstractmethod
    def answer(self, request: ConsultationRequest, sources: list[Source]) -> ConsultationResponse:
        """Return a compliant, source-aware answer."""

