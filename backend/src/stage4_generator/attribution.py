"""Answer-to-source attribution for the UI."""

import re
from typing import Any, Dict, List, Optional

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")
_CITATION = re.compile(r"\[(\d+(?:\s*[,-]\s*\d+)*)\]")
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "was", "were", "are", "has", "have",
    "had", "not", "but", "its", "their", "there", "these", "those", "than", "then", "which",
    "who", "whom", "into", "onto", "also", "may", "can", "could", "would", "should", "been",
    "being", "such", "both", "each", "other", "more", "most", "less", "per", "via", "between",
    "among", "about", "over", "under", "after", "before", "during", "while", "based", "provided",
    "documents", "document", "source", "sources", "decision", "yes", "maybe",
}

EMBEDDING_THRESHOLD = 0.55
LEXICAL_THRESHOLD = 0.5


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    sentences: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            for part in _SENTENCE_SPLIT.split(line):
                part = part.strip()
                if not part:
                    continue
                if sentences and _CITATION.search(part) and not _content_words(part):
                    sentences[-1] = f"{sentences[-1]} {part}"
                else:
                    sentences.append(part)
    return sentences


def parse_citations(sentence: str, n_sources: int) -> List[int]:
    """Returns the 1-based source numbers cited as [1], [1, 3] or [2-4] in a sentence."""
    cited: List[int] = []
    for group in _CITATION.findall(sentence):
        for part in re.split(r"\s*,\s*", group):
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                numbers = range(lo, hi + 1)
            else:
                numbers = [int(part)]
            for n in numbers:
                if 1 <= n <= n_sources and n not in cited:
                    cited.append(n)
    return cited


def _content_words(text: str) -> set:
    return {w for w in _WORD.findall(_CITATION.sub(" ", text.lower())) if len(w) >= 3 and w not in _STOPWORDS}


def _lexical_similarity(answer_sentence: str, doc_sentence: str) -> float:
    a, d = _content_words(answer_sentence), _content_words(doc_sentence)
    if not a or not d:
        return 0.0
    return len(a & d) / len(a)


def attribute_answer(
    answer: str,
    sources: List[Dict[str, Any]],
    encoder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Links each answer sentence to the source sentences that support it (for UI highlighting)."""
    answer_sentences = split_sentences(answer)
    doc_sentences = [split_sentences(doc.get("text", "")) for doc in sources]
    n_sources = len(sources)

    method, threshold = "lexical", LEXICAL_THRESHOLD
    similarity = None
    if encoder is not None and answer_sentences and any(doc_sentences):
        try:
            flat_docs = [s for sents in doc_sentences for s in sents]
            a_emb = encoder.encode(
                [_CITATION.sub("", s) for s in answer_sentences],
                convert_to_numpy=True, normalize_embeddings=True,
            )
            d_emb = encoder.encode(flat_docs, convert_to_numpy=True, normalize_embeddings=True)
            matrix = a_emb @ d_emb.T
            offsets, start = [], 0
            for sents in doc_sentences:
                offsets.append(start)
                start += len(sents)

            def similarity(i: int, doc_idx: int, sent_idx: int) -> float:
                return float(matrix[i, offsets[doc_idx] + sent_idx])

            method, threshold = "embedding", EMBEDDING_THRESHOLD
        except Exception as exc:
            print(f"[Attribution] Embedding similarity unavailable ({exc}); using word overlap.")
            similarity = None

    if similarity is None:
        def similarity(i: int, doc_idx: int, sent_idx: int) -> float:
            return _lexical_similarity(answer_sentences[i], doc_sentences[doc_idx][sent_idx])

    source_out = [
        {"rank": d + 1, "sentences": [{"text": s, "answer_sentences": [], "score": 0.0} for s in sents]}
        for d, sents in enumerate(doc_sentences)
    ]
    answer_out = []
    for i, sentence in enumerate(answer_sentences):
        cited = parse_citations(sentence, n_sources)
        supported_by = []
        for d, sents in enumerate(doc_sentences):
            if not sents:
                continue
            scores = [similarity(i, d, j) for j in range(len(sents))]
            best = max(range(len(sents)), key=scores.__getitem__)
            if scores[best] >= threshold:
                supported_by.append({"source": d + 1, "sentence": best, "score": round(scores[best], 3)})
                entry = source_out[d]["sentences"][best]
                entry["answer_sentences"].append(i)
                entry["score"] = round(max(entry["score"], scores[best]), 3)
        supported_by.sort(key=lambda s: -s["score"])
        answer_out.append({"index": i, "text": sentence, "cited": cited, "supported_by": supported_by})

    return {
        "method": method,
        "threshold": threshold,
        "answer_sentences": answer_out,
        "sources": source_out,
    }
