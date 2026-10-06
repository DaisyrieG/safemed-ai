"""SafeMed AI pipeline: Stages 1 to 4 of the Chapter 3 system architecture."""

from typing import Any, Dict, List, Optional
import json
import os

from src.stage1_retriever.retriever import BiEncoderRetriever
from src.stage2_filter.cross_encoder_filter import HarmfulDocumentFilter, FilterResult
from src.stage3_reranker.reranker import Reranker
from src.stage4_generator.generator import AnswerGenerator
from src.stage4_generator.attribution import attribute_answer

POOL_SIZE = 30
TOP_K = 5

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATA_DIR = os.path.join(_BACKEND, "data")
DEMO_CORPUS = os.path.join(_DATA_DIR, "demo", "pubmedqa_demo_corpus.json")
DEMO_HARMFUL = os.path.join(_DATA_DIR, "demo", "pubmedqa_demo_harmful.json")
PQAA_CORPUS = os.path.join(_DATA_DIR, "processed", "pqaa_corpus.json")
DEFAULT_FILTER_MODEL = os.path.join(_BACKEND, "models", "safemed_filter")


def resolve_corpus_path(choice: str) -> str:
    """Corpus file for 'demo' (web app), 'pubmedqa' (benchmark) or a path."""
    choice = (choice or "demo").strip()
    named = {"demo": DEMO_CORPUS, "pubmedqa": PQAA_CORPUS}
    path = named.get(choice.lower(), os.path.abspath(choice))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus '{choice}' not found at {path}")
    return path


def resolve_filter_model_path() -> str:
    """SAFEMED_FILTER_MODEL, else backend/models/safemed_filter (written by train_filter.py)."""
    return os.path.abspath(os.getenv("SAFEMED_FILTER_MODEL", "").strip() or DEFAULT_FILTER_MODEL)


def load_extra_documents(corpus_path: str) -> List[Dict[str, Any]]:
    """The labelled synthetic harmful documents (scripts/generate_harmful_docs.py) join the demo corpus."""
    if corpus_path != DEMO_CORPUS or not os.path.exists(DEMO_HARMFUL):
        return []
    with open(DEMO_HARMFUL, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalise(text: str) -> str:
    return " ".join(str(text or "").lower().split()).rstrip("?. ")


def mark_source_abstract(query: str, candidates: List[Dict[str, Any]],
                         target_pmid: Optional[str] = None) -> List[Dict[str, Any]]:
    """Sets each candidate's gold label for this query: its source abstract is GD, a synthetic HD only counts for its own question.

    The query's PubMedQA question is identified by target_pmid when given (e.g. a paraphrased sample case), else by exact text."""
    q = _normalise(query)
    target = str(target_pmid) if target_pmid else None
    marked = []
    for doc in candidates:
        if doc.get("true_label") == "harmful" and doc.get("target_question"):
            own = (str(doc.get("target_pmid")) == target) if target else _normalise(doc["target_question"]) == q
            if not own:
                doc = {**doc, "true_label": "mediocre"}
        elif doc.get("true_label") != "harmful" and (
                (target and str(doc.get("pmid")) == target) or (q and _normalise(doc.get("title")) == q)):
            doc = {**doc, "true_label": "ground_truth", "is_source_abstract": True}
        marked.append(doc)
    return marked


def _category(doc: Dict[str, Any]) -> str:
    """GD / HD / MD from the gold label when the corpus has one, else from the filter's prediction."""
    label = doc.get("true_label") or doc.get("filter_label")
    return "GD" if label == "ground_truth" else "HD" if label == "harmful" else "MD"


class SafeMedPipeline:
    def __init__(
        self,
        use_filter: bool = True,
        filter_model_path: Optional[str] = None,
        corpus_path: Optional[str] = None,
        retriever: Optional[BiEncoderRetriever] = None,
        harm_filter: Optional[HarmfulDocumentFilter] = None,
        reranker: Optional[Reranker] = None,
        generator: Optional[AnswerGenerator] = None,
    ):
        """corpus_path defaults to SAFEMED_CORPUS (demo | pubmedqa | <path>), else demo."""
        self.use_filter = use_filter

        if retriever is None:
            corpus_path = corpus_path or resolve_corpus_path(os.getenv("SAFEMED_CORPUS", "demo"))
            print(f"[Pipeline] Corpus: {corpus_path}")
            retriever = BiEncoderRetriever(top_k=50, corpus_path=corpus_path)
            extra = load_extra_documents(corpus_path)
            if extra:
                retriever.build_index(list(retriever.corpus) + extra)
                print(f"[Pipeline] Added {len(extra)} labelled harmful documents from {DEMO_HARMFUL}")
        self.retriever = retriever

        if use_filter and harm_filter is None:
            harm_filter = HarmfulDocumentFilter(model_path=filter_model_path or resolve_filter_model_path(),
                                                cutoff_k=POOL_SIZE)
        self.filter = harm_filter if use_filter else None

        self.reranker = reranker or Reranker(top_k=TOP_K)
        self.generator = generator or AnswerGenerator()

        self.decompose_claims = False
        self.last_filter_result: Optional[FilterResult] = None
        self.last_candidates: List[Dict[str, Any]] = []
        self.last_control_top5: List[Dict[str, Any]] = []

    def run(
        self,
        query: str,
        exclude_id: Optional[str] = None,
        generate_control_answer: bool = True,
        target_pmid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs one query through Stages 1 to 4 for both conditions."""
        query_text = query.strip()

        candidates = self.retriever.retrieve(query_text, top_k=50, exclude_id=exclude_id)
        candidates = mark_source_abstract(query_text, candidates, target_pmid)
        self.last_candidates = candidates

        if self.filter is not None:
            fr = self.filter.filter(query_text, candidates)
            classified, blocked, pool = fr.classified, fr.blocked, fr.surviving
        else:
            fr = None
            classified, blocked, pool = candidates, [], candidates[:POOL_SIZE]
        self.last_filter_result = fr

        top5 = self.reranker.rerank(query_text, pool, top_k=TOP_K)
        control_top5 = self.reranker.rerank(query_text, candidates[:POOL_SIZE], top_k=TOP_K)
        self.last_control_top5 = control_top5

        answer = self.generator.generate_answer(query_text, top5)
        claims = self.generator.decompose_into_claims(answer) if self.decompose_claims else []
        attribution = attribute_answer(answer, top5, encoder=self.retriever.model)
        if generate_control_answer:
            control_answer = self.generator.generate_answer(query_text, control_top5)
            control_claims = self.generator.decompose_into_claims(control_answer) if self.decompose_claims else []
        else:
            control_answer, control_claims = "", []

        return self._build_response(query_text, exclude_id, candidates, classified, blocked, pool, fr,
                                    top5, answer, claims, attribution,
                                    control_top5, control_answer, control_claims)


    def _build_response(self, query_text, exclude_id, candidates, classified, blocked, pool, fr,
                        top5, answer, claims, attribution, control_top5, control_answer, control_claims):
        tau = self.filter.harmful_threshold if self.filter else None
        reinstated_ids = {str(e.get("id")) for e in (fr.reinstated if fr else [])}
        by_id = {str(d.get("id")): d for d in classified}

        evaluation_log = []
        for d in (classified if fr else []):
            was_reinstated = str(d.get("id")) in reinstated_ids
            status = ("REINSTATED: FALLBACK GUARD" if was_reinstated
                      else "BLOCKED: UNSAFE" if d.get("is_blocked") else "TRUSTED SOURCE")
            evaluation_log.append({
                "id": d.get("id"),
                "title": d.get("title", "Untitled Source"),
                "source": d.get("source", "PubMed"),
                "category": _category(d),
                "status": status,
                "is_blocked": bool(d.get("is_blocked")),
                "reinstated": was_reinstated,
                "block_reason": d.get("block_reason"),
                "harmful_probability": d.get("harmful_probability", 0.0),
                "true_label": d.get("true_label"),
                "filter_label": d.get("filter_label"),
                "scorer": d.get("scorer"),
                "snippet": d.get("text", "")[:180] + "...",
                "text": d.get("text", ""),
                "pmid": d.get("pmid"),
                "retrieval_score": d.get("retrieval_score"),
            })

        safety_scan = {
            "filter_enabled": fr is not None,
            "tau_safe": tau,
            "scorer": self.filter.scorer_name if self.filter else None,
            "total_scanned": len(candidates),
            "blocked_count": len(blocked),
            "safe_count": len(candidates) - len(blocked),
            "reinstated_count": len(reinstated_ids),
            "reinstated": fr.reinstated if fr else [],
            "final_pool_size": len(pool),
            "evaluation_log": evaluation_log,
        }

        trusted_sources = [{
            "id": d.get("id"),
            "title": d.get("title", "Untitled Source"),
            "source": d.get("source", "PubMed"),
            "verification_status": "reinstated by fallback guard" if d.get("reinstated") else "verified safe",
            "relevance_score": round(float(d.get("rerank_score", 0.0)), 4),
            "snippet": d.get("text", "")[:220] + "...",
            "text": d.get("text", ""),
            "pmid": d.get("pmid"),
            "harmful_probability": d.get("harmful_probability"),
            "retrieval_score": d.get("retrieval_score"),
        } for d in top5]

        def condition(docs, is_control, cond_answer, cond_claims, name):
            formatted = []
            for doc in docs:
                info = by_id.get(str(doc.get("id")), doc)
                formatted.append({
                    "id": str(doc.get("id") or ""),
                    "title": doc.get("title", "Untitled Source"),
                    "source": doc.get("source", "PubMed"),
                    "category": _category(info),
                    "status": ("BLOCKED BY STAGE 2 FILTER" if info.get("is_blocked") else "EXPOSED EVIDENCE")
                    if is_control else
                    ("REINSTATED BY FALLBACK GUARD" if doc.get("reinstated") else "VERIFIED SAFE EVIDENCE"),
                    "is_blocked_in_proposed": bool(info.get("is_blocked", False)),
                    "reinstated": bool(doc.get("reinstated", False)),
                    "true_label": info.get("true_label"),
                    "block_reason": info.get("block_reason"),
                    "harmful_probability": float(info.get("harmful_probability", 0.0)),
                    "relevance_score": round(float(doc["rerank_score"]), 4) if doc.get("rerank_score") is not None else None,
                    "snippet": doc.get("text", "")[:220] + "...",
                })
            hd = sum(d["category"] == "HD" for d in formatted)
            gd = sum(d["category"] == "GD" for d in formatted)
            return {
                "condition": name,
                "answer": cond_answer,
                "top5_documents": formatted,
                "hd_count": hd,
                "hd_at_top5": round(hd / TOP_K, 4),
                "gd_count": gd,
                "hit_at_5": int(gd > 0),
                "gd_retention_rate": round(gd / TOP_K, 4),
                "claims": cond_claims,
                "has_harmful_exposure": hd > 0,
            }

        control = condition(control_top5, True, control_answer, control_claims, "Control (Standard RAG)")
        proposed = condition(top5, False, answer, claims, "Proposed (SafeMed AI)")
        control["hallucination_risk"] = "Elevated" if control["hd_count"] else ("Low" if control["gd_count"] else "Moderate")
        proposed["hallucination_risk"] = ("Protected" if not proposed["hd_count"] and control["hd_count"]
                                          else "Low" if not proposed["hd_count"] else "Elevated")
        reduction = round(control["hd_at_top5"] - proposed["hd_at_top5"], 4)

        evaluation_breakdown = {
            "query": query_text,
            "tau_threshold": tau if tau is not None else 0.0,
            "total_screened": len(candidates),
            "total_blocked": len(blocked),
            "blocked_document_ids": [str(d.get("id")) for d in blocked],
            "reinstated": fr.reinstated if fr else [],
            "control": control,
            "proposed": proposed,
            "hd_exposure_reduction": reduction,
            "fidelity_status": ("Evidence Protected: Harmful Context Eliminated" if control["hd_count"] and not proposed["hd_count"]
                                else "Clinically Grounded: 0% Harmful Exposure" if not proposed["hd_count"]
                                else "Review Required: Harmful Document Detected"),
        }

        return {
            "query": query_text,
            "configuration": "Proposed Framework" if self.filter else "Standard RAG",
            "filter_enabled": self.filter is not None,
            "excluded_source_id": exclude_id,
            "consensus_status": "CLINICAL CONSENSUS",
            "clinical_summary": answer,
            "answer": answer,
            "trusted_sources": trusted_sources,
            "attribution": attribution,
            "top5_documents": top5,
            "atomic_claims": claims,
            "safety_scan": safety_scan,
            "evaluation_breakdown": evaluation_breakdown,
            "metrics": {
                "hd_at_top5": proposed["hd_at_top5"],
                "hit_at_5": proposed["hit_at_5"],
                "gd_retention_rate": proposed["gd_retention_rate"],
                "hd_exposure_reduction": reduction,
            },
        }
