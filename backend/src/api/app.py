"""SafeMed AI web API: runs a question through Stages 1-4."""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EVAL_RESULTS_PATH = os.path.join(_BACKEND, "results", "evaluation_results.json")

import hashlib
import threading
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ClinicalQueryRequest,
    ClinicalResponse,
    SampleClinicalCase,
    SafetyScanSummary,
    DocumentEvaluationEntry,
    TrustedDocument,
    SingleQueryEvaluationBreakdown,
)
from src.pipeline import SafeMedPipeline
from src.stage4_generator.generator import GeneratorUnavailableError

app = FastAPI(
    title="SafeMed AI - Verified Clinical Search Assistant",
    description=(
        "Clinical decision-support API utilizing a 4-Stage Modular RAG Pipeline with a "
        "Pre-Reranking Cross-Encoder Safety Filter to eliminate harmful literature exposure."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Highlights-Found", "X-Highlights-Requested", "X-First-Page"],
)

_pipeline: SafeMedPipeline | None = None
_pipeline_lock = threading.Lock()

def get_pipeline() -> SafeMedPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = SafeMedPipeline()
    return _pipeline


PUBMEDQA_SAMPLE_CASES: List[Dict[str, str]] = [
    {
        "case_id": "pqa-24318956",
        "title": "Case 1: Digoxin & Prostate Cancer Risk",
        "query": "Is digoxin use for cardiovascular disease associated with risk of prostate cancer?",
        "description": "PubMedQA test question, PMID 24318956. Expert answer: yes.",
    },
    {
        "case_id": "pqa-24666444",
        "title": "Case 2: The \"July Effect\" in Cancer Surgery",
        "query": "Is there any evidence of a \"July effect\" in patients undergoing major cancer surgery?",
        "description": "PubMedQA test question, PMID 24666444. Expert answer: no.",
    },
    {
        "case_id": "pqa-25371231",
        "title": "Case 3: Vitamin D & Osteochondritis Dissecans",
        "query": "Is vitamin D insufficiency or deficiency related to the development of osteochondritis dissecans?",
        "description": "PubMedQA test question, PMID 25371231. Expert answer: maybe.",
    },
]

_OPEN_ACCESS_CASES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "demo", "open_access_cases.json"
)


def current_sample_cases() -> List[Dict[str, str]]:
    if os.path.exists(_OPEN_ACCESS_CASES_PATH):
        try:
            with open(_OPEN_ACCESS_CASES_PATH, "r", encoding="utf-8") as f:
                cases = json.load(f)
            if cases:
                return [{k: c[k] for k in ("case_id", "title", "query", "description")} for c in cases]
        except (OSError, ValueError, KeyError) as exc:
            print(f"[API] Ignoring {_OPEN_ACCESS_CASES_PATH}: {exc}")
    return PUBMEDQA_SAMPLE_CASES


@app.get("/", summary="Root Index")
def root_index():
    return {
        "service": "SafeMed AI - Verified Clinical Search Assistant",
        "status": "online",
        "docs_url": "/docs",
        "endpoints": {
            "query": "POST /api/query",
            "sample_cases": "GET /api/sample-cases",
            "health": "GET /api/health",
            "evaluation_results": "GET /api/evaluation-results",
        },
    }


@app.get("/api/evaluation-results", summary="Fetch Pre-Generated Evaluation Results")
def get_evaluation_results() -> Dict[str, Any]:
    """Returns the latest Stage 5 benchmark results (backend/results/evaluation_results.json)."""
    if not os.path.isfile(_EVAL_RESULTS_PATH):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Evaluation results file not found at '{_EVAL_RESULTS_PATH}'. "
                "Run `python -m src.stage5_evaluation.run_benchmark` from backend/ to generate it."
            ),
        )
    try:
        with open(_EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse evaluation results JSON: {exc}") from exc


@app.get("/api/health", summary="System Health & Pipeline Status")
def health_check():
    try:
        pipeline = get_pipeline()
        return {
            "status": "healthy",
            "service": "SafeMed AI",
            "pipeline": "4-Stage Modular RAG",
            "stage_1_retriever": pipeline.retriever.model_name,
            "stage_2_filter": {
                "architecture": "Pre-Reranking Cross-Encoder Filter",
                "threshold": pipeline.filter.harmful_threshold,
                "cutoff_k": pipeline.filter.cutoff_k,
                "scorer": pipeline.filter.scorer_name,
            } if pipeline.filter is not None else None,
            "stage_3_reranker": pipeline.reranker.model_name,
            "stage_4_generator": pipeline.generator.model,
            "stage_4_endpoint": pipeline.generator.base_url or "OpenAI API",
            "stage_4_ready": pipeline.generator.client is not None,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Pipeline failed to initialise: {exc}") from exc


@app.get("/api/sample-cases", response_model=List[SampleClinicalCase], summary="List Preset Clinical Cases")
def get_sample_cases():
    """Returns preset clinical cases for testing and demonstration."""
    return [SampleClinicalCase(**c) for c in current_sample_cases()]


def _str_or_none(value):
    return None if value is None else str(value)


def _float_or_none(value):
    try:
        return None if value is None else round(float(value), 4)
    except (TypeError, ValueError):
        return None


@app.post("/api/query", response_model=ClinicalResponse, summary="Execute Clinical Query")
def search_clinical_query(req: ClinicalQueryRequest):
    """Runs a clinical question through Stages 1-4."""
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        pipeline = get_pipeline()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline is unavailable: {exc}"
        ) from exc

    try:
        result = pipeline.run(query_text)
    except GeneratorUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    scan_entries = [
        DocumentEvaluationEntry(
            id=entry.get("id"),
            title=entry["title"],
            source=entry["source"],
            category=entry["category"],
            status=entry["status"],
            is_blocked=entry["is_blocked"],
            block_reason=entry.get("block_reason"),
            harmful_probability=entry["harmful_probability"],
            snippet=entry["snippet"],
            text=entry.get("text"),
            pmid=_str_or_none(entry.get("pmid")),
            reinstated=bool(entry.get("reinstated")),
            scorer=entry.get("scorer"),
            retrieval_score=_float_or_none(entry.get("retrieval_score")),
        )
        for entry in result["safety_scan"]["evaluation_log"]
    ]

    safety_scan = SafetyScanSummary(
        total_scanned=result["safety_scan"]["total_scanned"],
        blocked_count=result["safety_scan"]["blocked_count"],
        safe_count=result["safety_scan"]["safe_count"],
        evaluation_log=scan_entries,
    )

    trusted_sources = [
        TrustedDocument(
            id=src.get("id"),
            title=src["title"],
            source=src["source"],
            verification_status=src["verification_status"],
            relevance_score=src["relevance_score"],
            snippet=src["snippet"],
            text=src.get("text"),
            pmid=_str_or_none(src.get("pmid")),
            harmful_probability=_float_or_none(src.get("harmful_probability")),
            retrieval_score=_float_or_none(src.get("retrieval_score")),
        )
        for src in result["trusted_sources"]
    ]

    eval_breakdown = None
    if result.get("evaluation_breakdown"):
        try:
            eval_breakdown = SingleQueryEvaluationBreakdown(**result["evaluation_breakdown"])
        except Exception as exc:
            print(f"[API] Warning: SingleQueryEvaluationBreakdown parsing failed: {exc}")
            eval_breakdown = None

    return ClinicalResponse(
        query=result["query"],
        consensus_status=result["consensus_status"],
        clinical_summary=result["clinical_summary"],
        trusted_sources=trusted_sources,
        atomic_claims=result["atomic_claims"],
        safety_scan=safety_scan,
        evaluation_breakdown=eval_breakdown,
        attribution=result.get("attribution"),
        llm_model=pipeline.generator.model,
        llm_endpoint=pipeline.generator.base_url or "OpenAI API",
    )


class HighlightRequest(BaseModel):
    sentences: List[str] = []
    notes: Optional[List[str]] = None


def _fulltext_modules():
    try:
        from src.api.fulltext.pmc_fetch import get_fulltext_pdf
        from src.api.fulltext.pdf_highlight import highlight_sentences
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"Full-text PDFs need extra packages: pip install pymupdf rapidfuzz httpx ({exc})",
        ) from exc
    return get_fulltext_pdf, highlight_sentences


def _lookup_fulltext(pmid: str):
    get_fulltext_pdf, _ = _fulltext_modules()
    try:
        return get_fulltext_pdf(pmid)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/documents/{pmid}/fulltext", summary="Open-Access Full-Text Availability")
def fulltext_status(pmid: str):
    """Says whether an open-access PDF exists for the article (downloads and caches it if so)."""
    info = _lookup_fulltext(pmid)
    return {
        "pmid": info.pmid,
        "pmcid": info.pmcid,
        "is_open_access": info.is_open_access,
        "license": info.license,
        "has_pdf": info.has_pdf,
        "error": info.error,
    }


@app.post("/api/documents/{pmid}/highlighted-pdf", summary="Source PDF With Cited Sentences Highlighted")
def highlighted_pdf(pmid: str, req: HighlightRequest, download: bool = False):
    """Returns the article's open-access PDF with the given sentences highlighted in yellow."""
    _, highlight_sentences = _fulltext_modules()
    info = _lookup_fulltext(pmid)
    if info.error:
        raise HTTPException(status_code=502, detail=info.error)
    if not info.has_pdf:
        raise HTTPException(
            status_code=404,
            detail={"fallback": "abstract", "pmcid": info.pmcid,
                    "reason": "No open-access PDF in PubMed Central for this article."},
        )

    key = hashlib.sha256(json.dumps([req.sentences, req.notes]).encode("utf-8")).hexdigest()[:16]
    folder = os.path.dirname(info.pdf_path)
    out_path = os.path.join(folder, f"highlighted_{key}.pdf")
    report_path = os.path.join(folder, f"highlighted_{key}.json")
    if os.path.exists(out_path) and os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    else:
        results = highlight_sentences(info.pdf_path, req.sentences, out_path, notes=req.notes)
        report = [r.__dict__ for r in results]
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)

    found = [r for r in report if r["found"]]
    return FileResponse(
        out_path,
        media_type="application/pdf",
        filename=f"PMID{info.pmid}_highlighted.pdf",
        content_disposition_type="attachment" if download else "inline",
        headers={
            "X-Highlights-Found": str(len(found)),
            "X-Highlights-Requested": str(len(report)),
            "X-First-Page": str(min((r["page"] for r in found), default=1)),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)
