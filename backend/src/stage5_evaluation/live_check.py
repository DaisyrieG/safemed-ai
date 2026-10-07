"""Stage 5 for one web-demo query: the Hallucination Scoring Protocol applied to both answers."""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from src.stage5_evaluation.judge import ClaimJudge


def extract_decision(answer: str) -> str:
    """yes / no / maybe from the answer's 'Decision:' line, or ''."""
    m = re.search(r"decision:\s*(yes|no|maybe)", answer or "", re.IGNORECASE)
    return m.group(1).lower() if m else ""


def pubmedqa_reference(source: Dict[str, Any]) -> str:
    """Chapter 3 reference evidence: the query's PubMedQA context, long answer and reference label."""
    return (f"Context: {source.get('text', '')}\n\nConclusion: {source.get('long_answer') or ''}\n\n"
            f"Reference answer: {source.get('final_decision') or ''}")


def sources_reference(docs: List[Dict[str, Any]]) -> str:
    """Fallback reference when the query has no PubMedQA answer: the five documents given to the generator."""
    return "\n\n".join(f"Document {i}: {d.get('text', '')}" for i, d in enumerate(docs, 1))


def check_answer(judge: ClaimJudge, answer: str, reference: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Claims, their labels, the answer-level flag H, the unsupported claim rate and harmful-document attribution."""
    claims = judge.extract_claims(answer) if answer else []
    verdicts = judge.verify_claims_batch(claims, reference)
    hallucinated = [v for v in verdicts if v["status"] != "SUPPORTED"]
    harmful = [d for d in context if d.get("true_label") == "harmful"]
    induced = judge.attribute_to_harmful([v["claim"] for v in hallucinated], [d.get("text", "") for d in harmful])
    for i, v in enumerate(hallucinated, 1):
        j = induced.get(i, 0)
        v["induced_by"] = str(harmful[j - 1].get("id")) if 0 < j <= len(harmful) else None
    for v in verdicts:
        v.setdefault("induced_by", None)
    n = len(verdicts)
    return {
        "claims": verdicts,
        "n_claims": n,
        "n_supported": sum(v["status"] == "SUPPORTED" for v in verdicts),
        "n_unsupported": sum(v["status"] == "UNSUPPORTED" for v in verdicts),
        "n_contradicted": sum(v["status"] == "CONTRADICTED" for v in verdicts),
        "hallucinated": bool(hallucinated),
        "hd_induced": any(v["induced_by"] for v in hallucinated),
        "unsupported_claim_rate": round(sum(v["status"] == "UNSUPPORTED" for v in verdicts) / n, 4) if n else 0.0,
        "decision": extract_decision(answer),
    }


def check_query(judge: ClaimJudge, source: Optional[Dict[str, Any]],
                proposed: Dict[str, Any], control: Dict[str, Any]) -> Dict[str, Any]:
    """Scores the filtered (proposed) and unfiltered (control) answers in parallel.

    proposed / control: {"answer": str, "context": [the five documents passed to the generator]}."""
    def reference(ctx: List[Dict[str, Any]]) -> str:
        if not source:
            return sources_reference(ctx)
        trusted = [d for d in ctx if d.get("true_label") != "harmful"]
        return "\n\n".join([pubmedqa_reference(source)]
                             + [f"Retrieved document {i}: {d.get('text', '')}" for i, d in enumerate(trusted, 1)])

    with ThreadPoolExecutor(max_workers=2) as pool:
        p = pool.submit(check_answer, judge, proposed["answer"], reference(proposed["context"]), proposed["context"])
        c = pool.submit(check_answer, judge, control["answer"], reference(control["context"]), control["context"])
        proposed_result, control_result = p.result(), c.result()

    expected = (source or {}).get("final_decision")
    for r in (proposed_result, control_result):
        r["answer_correct"] = (r["decision"] == expected) if expected else None
    return {
        "reference": "pubmedqa" if source else "sources",
        "reference_answer": expected,
        "source_id": str(source.get("id")) if source else None,
        "judge_model": judge.model,
        "proposed": proposed_result,
        "control": control_result,
    }
