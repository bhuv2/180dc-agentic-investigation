from pydantic import BaseModel
from typing import List

class Citation(BaseModel):
    document_id: str
    claim: str

class InvestigationResult(BaseModel):
    theory: str
    confidence: float
    citations: List[Citation]
    needs_more_evidence: bool
    execution_trace: List[str] = []

class FactCheckResult(BaseModel):
    contradictions_found: bool
    details: str