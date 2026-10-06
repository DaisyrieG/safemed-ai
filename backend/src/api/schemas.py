"""Pydantic schemas for the SafeMed AI Verified Clinical Search Assistant API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentEvaluationEntry(BaseModel):
    id: Optional[str] = None
    title: str
    source: str
    category: str = Field(description="GD (Ground-Truth) | HD (Harmful) | MD (Mediocre)")
    status: str = Field(description="TRUSTED SOURCE | BLOCKED: UNSAFE")
    is_blocked: bool
    block_reason: Optional[str] = None
    harmful_probability: float
    snippet: str
    text: Optional[str] = None
    pmid: Optional[str] = None
    reinstated: bool = False
    scorer: Optional[str] = None
    retrieval_score: Optional[float] = None


class SafetyScanSummary(BaseModel):
    total_scanned: int = Field(description="Total candidate literature passages screened in Stage 2")
    blocked_count: int = Field(description="Total harmful or misleading documents blocked")
    safe_count: int = Field(description="Total safe passages retained for ranking")
    evaluation_log: List[DocumentEvaluationEntry]


class TrustedDocument(BaseModel):
    id: Optional[str] = None
    title: str
    source: str
    verification_status: str = "verified safe"
    relevance_score: float
    snippet: str
    text: Optional[str] = None
    pmid: Optional[str] = None
    harmful_probability: Optional[float] = None
    retrieval_score: Optional[float] = None


class SingleQueryDocument(BaseModel):
    id: Optional[str] = None
    title: str
    source: str
    category: str = Field(description="GD (Ground-Truth) | HD (Harmful) | MD (Mediocre)")
    status: Optional[str] = None
    is_blocked_in_proposed: bool = Field(default=False, description="True if Stage 2 filter blocked this document")
    block_reason: Optional[str] = None
    harmful_probability: float = Field(default=0.0)
    relevance_score: Optional[float] = None
    snippet: str


class SingleQueryConditionMetrics(BaseModel):
    condition: str = Field(description="'Control (Standard RAG)' or 'Proposed (SafeMed AI)'")
    answer: str
    top5_documents: List[SingleQueryDocument]
    hd_count: int = Field(description="Count of harmful documents (HD) in Top-5 context")
    hd_at_top5: float = Field(description="Proportion of harmful documents in Top-5 (0.0 to 1.0)")
    gd_count: int = Field(description="Count of ground-truth documents (GD) in Top-5 context")
    hit_at_5: int = Field(description="Hit@5: 1 if at least one GD is retained, 0 otherwise")
    gd_retention_rate: float = Field(description="Proportion of ground-truth documents in Top-5 (0.0 to 1.0)")
    claims: List[str] = Field(default_factory=list)
    has_harmful_exposure: bool = Field(description="True if HD@Top-5 > 0")
    hallucination_risk: str = Field(description="'Elevated' | 'Low' | 'Protected'")


class SingleQueryEvaluationBreakdown(BaseModel):
    query: str
    tau_threshold: float = Field(default=0.5, description="Stage 2 cross-encoder filter decision boundary tau")
    total_screened: int = Field(default=50, description="Total candidate literature passages screened")
    total_blocked: int = Field(description="Total passages exceeding tau that were blocked")
    blocked_document_ids: List[str] = Field(default_factory=list)
    control: SingleQueryConditionMetrics
    proposed: SingleQueryConditionMetrics
    hd_exposure_reduction: float = Field(description="Control HD@Top-5 minus Proposed HD@Top-5")
    fidelity_status: str = Field(description="Clinical verification verdict")


class ClinicalQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Clinical or biomedical search query")
    case_id: Optional[str] = Field(None, description="Optional preset case ID")


class ClinicalResponse(BaseModel):
    query: str
    consensus_status: str = "CLINICAL CONSENSUS"
    clinical_summary: str
    trusted_sources: List[TrustedDocument]
    atomic_claims: List[str]
    safety_scan: SafetyScanSummary
    evaluation_breakdown: Optional[SingleQueryEvaluationBreakdown] = None
    attribution: Optional[Dict[str, Any]] = Field(
        None, description="Answer-sentence to source-sentence links used for highlighting"
    )
    llm_model: Optional[str] = None
    llm_endpoint: Optional[str] = None


class SampleClinicalCase(BaseModel):
    case_id: str
    title: str
    query: str
    description: str

