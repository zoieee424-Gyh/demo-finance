from fastapi import APIRouter

from app.schemas.consultation import ConsultationRequest, ConsultationResponse
from app.services.consultation_service import ConsultationService


router = APIRouter(tags=["consultations"])
service = ConsultationService()


@router.post("/consultations", response_model=ConsultationResponse)
def create_consultation(payload: ConsultationRequest) -> ConsultationResponse:
    return service.handle(payload)

