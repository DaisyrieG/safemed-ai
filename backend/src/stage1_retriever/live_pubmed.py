"""Stage 1, optional corpus source: live PubMed search through NCBI E-utilities (ESearch + EFetch).

Selected with SAFEMED_CORPUS=live_pubmed. Returns the same candidate dicts as BiEncoderRetriever, so the
filter, reranker and generator run unchanged. Not part of the Chapter 3 experiment, whose corpus is the
pooled pqa_artificial abstracts.
"""

import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import numpy as np

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "with", "by", "at", "from", "as", "is", "are",
    "was", "were", "be", "been", "does", "do", "did", "can", "could", "should", "would", "will", "may", "might",
    "there", "any", "this", "that", "these", "those", "it", "its", "than", "into", "about", "after", "before",
    "between", "during", "patients", "people", "what", "which", "who", "whom", "how", "why", "when", "whether",
}


class LivePubMedError(RuntimeError):
    """Raised when PubMed cannot be reached or returns no usable abstracts."""


def keyword_query(question: str) -> str:
    """The question's content words joined with AND, a fallback when the plain question finds too little."""
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", question) if w.lower() not in _STOPWORDS]
    return " AND ".join(words)


def parse_pubmed_xml(xml_text: str) -> List[Dict[str, Any]]:
    """PubmedArticleSet XML from EFetch -> one document per article that has an abstract."""
    docs = []
    root = ET.fromstring(xml_text)
    for art in root.iter("PubmedArticle"):
        pmid = (art.findtext(".//MedlineCitation/PMID") or "").strip()
        title = "".join(art.find(".//ArticleTitle").itertext()).strip() if art.find(".//ArticleTitle") is not None else ""
        parts = []
        for node in art.findall(".//Abstract/AbstractText"):
            text = " ".join("".join(node.itertext()).split())
            if text:
                label = node.get("Label")
                parts.append(f"{label}: {text}" if label else text)
        if not pmid or not parts:
            continue
        journal = (art.findtext(".//Journal/Title") or "").strip()
        year = (art.findtext(".//JournalIssue/PubDate/Year") or art.findtext(".//JournalIssue/PubDate/MedlineDate") or "").strip()[:4]
        doi = next((e.text for e in art.findall(".//ArticleIdList/ArticleId") if e.get("IdType") == "doi" and e.text), None)
        docs.append({
            "id": f"pubmed_{pmid}",
            "pmid": pmid,
            "title": title,
            "text": " ".join(parts),
            "source": f"PubMed (live){' · ' + journal if journal else ''}{' ' + year if year else ''}",
            "journal": journal,
            "year": year,
            "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "publication_types": [e.text for e in art.findall(".//PublicationTypeList/PublicationType") if e.text],
        })
    return docs


class LivePubMedRetriever:
    """ESearch for PMIDs, EFetch their abstracts, then order them by MiniLM cosine similarity to the query."""

    def __init__(self, top_k: int = 50, model_name: str = DEFAULT_MODEL, api_key: Optional[str] = None,
                 email: Optional[str] = None, timeout: float = 20.0):
        from sentence_transformers import SentenceTransformer

        self.top_k = top_k
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.api_key = api_key or os.getenv("NCBI_API_KEY") or None
        self.email = email or os.getenv("NCBI_EMAIL") or None
        self.tool = os.getenv("SAFEMED_PUBMED_TOOL", "safemed-ai")
        self.timeout = timeout
        self.min_interval = 0.11 if self.api_key else 0.34
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._by_id: Dict[str, Dict[str, Any]] = {}
        print(f"[Stage 1] Live PubMed via NCBI E-utilities ({'with' if self.api_key else 'without'} API key)")

    @property
    def corpus(self) -> List[Dict[str, Any]]:
        """Every abstract fetched so far (the API looks documents up by id here)."""
        return list(self._by_id.values())

    def _get(self, endpoint: str, params: Dict[str, Any]) -> str:
        params = {**params, "tool": self.tool}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        url = EUTILS + endpoint + "?" + urllib.parse.urlencode(params)
        for attempt in range(4):
            with self._lock:
                wait = self.min_interval - (time.time() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                self._last_call = time.time()
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                    return resp.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503) or attempt == 3:
                    raise LivePubMedError(f"PubMed {endpoint} failed: HTTP {e.code}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 3:
                    raise LivePubMedError(f"PubMed unreachable: {e}") from e
            time.sleep(1.0 * (attempt + 1))
        raise LivePubMedError(f"PubMed {endpoint} failed")

    def search(self, term: str, retmax: int) -> List[str]:
        data = json.loads(self._get("esearch.fcgi", {"db": "pubmed", "term": term, "retmax": retmax,
                                                     "sort": "relevance", "retmode": "json"}))
        return list(data.get("esearchresult", {}).get("idlist", []))

    def fetch(self, pmids: List[str]) -> List[Dict[str, Any]]:
        docs = []
        for i in range(0, len(pmids), 200):
            docs += parse_pubmed_xml(self._get("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids[i:i + 200]),
                                                               "retmode": "xml"}))
        return docs

    def retrieve(self, query: str, top_k: Optional[int] = None, exclude_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Top-k live abstracts for the query, each with retrieval_score (cosine to the query)."""
        k = top_k or self.top_k
        pmids = self.search(query, k)
        for term in (keyword_query(query), keyword_query(query).replace(" AND ", " OR ")):
            if len(pmids) >= k or not term:
                break
            pmids += [p for p in self.search(term, 2 * k) if p not in pmids]
        exclude = str(exclude_id or "").replace("pqa_", "").replace("pubmed_", "")
        docs = [d for d in self.fetch(pmids[: 2 * k]) if d["pmid"] != exclude]
        if not docs:
            raise LivePubMedError("PubMed returned no abstracts for this question; try rephrasing it.")
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        e = self.model.encode([d["text"] for d in docs], convert_to_numpy=True, normalize_embeddings=True)
        scores = e @ q
        ranked = []
        for i in np.argsort(-scores, kind="stable")[:k]:
            doc = {**docs[i], "retrieval_score": float(scores[i])}
            self._by_id[doc["id"]] = {key: v for key, v in doc.items() if key != "retrieval_score"}
            ranked.append(doc)
        return ranked
