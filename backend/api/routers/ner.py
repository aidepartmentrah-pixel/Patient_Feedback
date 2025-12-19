"""
AUTO-GENERATED API CONTRACT

Source: Obsidian – Insert Record Page (NER)
Iteration: 1
Status: API skeleton only – no implementation
"""

from typing import List, Optional, Dict
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/ner", tags=["NER"])


# -------------------------
# Request Models
# -------------------------

class NerExtractRequest(BaseModel):
    text: str


# -------------------------
# Response Models
# -------------------------

class NerEntity(BaseModel):
    type: str
    text: str
    confidence: float


class NerExtractSuccessResponse(BaseModel):
    success: bool
    patient_name: str
    doctor_name: str
    confidence: Dict[str, float]
    other_entities: List[NerEntity]


class NerErrorResponse(BaseModel):
    success: bool
    error: str
    message: str


# -------------------------
# Routes
# -------------------------

@router.post(
    "/extract",
    response_model=NerExtractSuccessResponse,
)
def extract_entities(payload: NerExtractRequest):
    """
    Extract patient and doctor names using NLP/NER.
    """
    raise NotImplementedError
