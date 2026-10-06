"""Stage 1: Dense bi-encoder retriever (all-MiniLM-L6-v2)."""

import json
import os
from typing import List, Dict, Any, Optional

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def index_path_for(corpus_path: str) -> str:
    """The FAISS index that belongs to a corpus file: same folder and name, .faiss extension."""
    return os.path.splitext(corpus_path)[0] + ".faiss"


class BiEncoderRetriever:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_k: int = 50,
        corpus_path: Optional[str] = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.top_k = top_k
        self.model = SentenceTransformer(model_name)
        self.corpus: List[Dict[str, Any]] = []
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.faiss_index = None

        if corpus_path:
            self.load_corpus_from_file(corpus_path)

    def load_corpus_from_file(self, filepath: str):
        """Loads a corpus JSON file and its FAISS index when one was built for it."""
        with open(filepath, "r", encoding="utf-8") as f:
            documents = json.load(f)

        index_path = index_path_for(filepath)
        if os.path.exists(index_path):
            import faiss
            index = faiss.read_index(index_path)
            if index.ntotal == len(documents):
                self.corpus = documents
                self.faiss_index = index
                print(f"[Stage 1] Loaded FAISS index {index_path} ({index.ntotal} vectors).")
                return
            print(f"[Stage 1] {index_path} has {index.ntotal} vectors but the corpus has "
                  f"{len(documents)} documents; encoding in memory instead.")

        self.build_index(documents)

    def build_index(self, corpus: List[Dict[str, Any]]):
        """Encodes the corpus in memory (normalised embeddings, so dot product = cosine)."""
        self.corpus = corpus
        self.faiss_index = None
        texts = [doc.get("text", "") for doc in corpus]
        self.corpus_embeddings = (
            self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True) if texts else None
        )

    @staticmethod
    def _matches_exclude_id(doc: Dict[str, Any], exclude_id: Optional[str]) -> bool:
        """Leave-self-out test: True if the document is the query's own source document."""
        if exclude_id is None or str(exclude_id) == "":
            return False
        target = str(exclude_id)
        if target.startswith("pqa_"):
            target = target[len("pqa_"):]
        doc_id = str(doc.get("id", ""))
        if doc_id.startswith("pqa_"):
            doc_id = doc_id[len("pqa_"):]
        return doc_id == target or str(doc.get("pmid", "")) == target

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        exclude_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Returns the top-k candidates (default 50, Chapter 3), each with its retrieval_score."""
        k = top_k or self.top_k
        if not self.corpus:
            return []
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        if self.faiss_index is not None:
            fetch_k = min(len(self.corpus), k + (16 if exclude_id is not None else 0))
            scores, indices = self.faiss_index.search(query_embedding, fetch_k)
            ranked = [(int(i), float(s)) for s, i in zip(scores[0], indices[0]) if 0 <= i < len(self.corpus)]
        else:
            scores = self.corpus_embeddings @ query_embedding[0]
            order = np.argsort(-scores, kind="stable")
            ranked = [(int(i), float(scores[i])) for i in order]

        results = []
        for idx, score in ranked:
            if self._matches_exclude_id(self.corpus[idx], exclude_id):
                continue
            doc = dict(self.corpus[idx])
            doc["retrieval_score"] = score
            results.append(doc)
            if len(results) == k:
                break
        return results
