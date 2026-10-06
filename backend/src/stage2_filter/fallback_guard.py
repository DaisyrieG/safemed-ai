"""Safety Fallback Guard (Chapter 3, Stage 2)."""

from typing import Any, Dict, List, Tuple

MIN_CONTEXT_DOCS = 5


def apply_safety_fallback_guard(
    surviving: List[Dict[str, Any]],
    blocked: List[Dict[str, Any]],
    min_docs: int = MIN_CONTEXT_DOCS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reinstates the lowest-P(HD) blocked candidates until at least min_docs remain."""
    S = list(surviving)
    reinstated: List[Dict[str, Any]] = []
    if len(S) >= min_docs:
        return S, reinstated

    lowest_risk_first = sorted(blocked, key=lambda d: float(d.get("harmful_probability", 1.0)))
    for doc in lowest_risk_first:
        if len(S) >= min_docs:
            break
        restored = dict(doc)
        restored["reinstated"] = True
        S.append(restored)
        reinstated.append({
            "id": restored.get("id"),
            "title": restored.get("title"),
            "harmful_probability": float(restored.get("harmful_probability", 0.0)),
            "block_reason": restored.get("block_reason"),
            "size_before": len(S) - 1,
            "size_after": len(S),
        })
    return S, reinstated
