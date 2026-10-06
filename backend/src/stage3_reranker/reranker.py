"""Stage 3: Reranker."""

from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_MODEL, top_k: int = 5):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.top_k = top_k
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns copies of the top-k (default 5) candidates by relevance, each with its rerank_score."""
        if not candidates:
            return []
        scores = self.model.predict([[query, c.get("text", "")] for c in candidates])
        scored = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored[: top_k or self.top_k]
