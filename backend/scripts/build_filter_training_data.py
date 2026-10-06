"""Builds labelled (query, document) pairs for training the Stage 2 filter."""

import argparse
import json
import os
import random
import sys
from collections import Counter

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND)

DEMO_CORPUS = os.path.join(BACKEND, "data", "demo", "pubmedqa_demo_corpus.json")
HARMFUL = os.path.join(BACKEND, "data", "demo", "pubmedqa_demo_harmful.json")
OUT_DIR = os.path.join(BACKEND, "data", "annotations")
SAMPLE_CASE_PMIDS = {"24318956", "24666444", "25371231"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mediocre-per-question", type=int, default=3)
    parser.add_argument("--only-with-harmful", action="store_true", default=True,
                        help="use only questions that have generated harmful documents (default)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    with open(DEMO_CORPUS, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    harmful = []
    if os.path.exists(HARMFUL):
        with open(HARMFUL, "r", encoding="utf-8") as f:
            harmful = json.load(f)
    if not harmful:
        sys.exit("No harmful documents yet. Run backend/scripts/generate_harmful_docs.py --extra 300 first.")

    by_pmid = {d["pmid"]: d for d in corpus}
    harmful_by_target = {}
    for doc in harmful:
        harmful_by_target.setdefault(str(doc["target_pmid"]), []).append(doc)

    pmids = sorted(harmful_by_target) if args.only_with_harmful else sorted(by_pmid)
    pmids = [p for p in pmids if p in by_pmid]

    from src.stage1_retriever.retriever import BiEncoderRetriever
    retriever = BiEncoderRetriever(top_k=20)
    retriever.corpus = corpus
    retriever.build_index(corpus)

    pairs_by_question = {}
    for pmid in pmids:
        source = by_pmid[pmid]
        query = source["title"]
        rows = [_row(pmid, query, source, "ground_truth", "pubmedqa_source_abstract")]
        for doc in harmful_by_target.get(pmid, []):
            rows.append(_row(pmid, query, doc, "harmful", "synthetic_contradiction"))
        negatives = [d for d in retriever.retrieve(query, top_k=20) if d.get("pmid") != pmid]
        for doc in negatives[: args.mediocre_per_question]:
            rows.append(_row(pmid, query, doc, "mediocre", "retrieved_other_abstract"))
        pairs_by_question[pmid] = rows

    questions = [p for p in pairs_by_question if p not in SAMPLE_CASE_PMIDS]
    rng.shuffle(questions)
    n_val = n_test = max(1, round(0.15 * len(questions)))
    splits = {
        "test": questions[:n_test] + [p for p in pairs_by_question if p in SAMPLE_CASE_PMIDS],
        "val": questions[n_test:n_test + n_val],
        "train": questions[n_test + n_val:],
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    for split, qs in splits.items():
        path = os.path.join(OUT_DIR, f"filter_{split}.jsonl")
        rows = [r for q in qs for r in pairs_by_question[q]]
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split:5s}: {len(qs):4d} questions, {len(rows):5d} pairs {dict(Counter(r['label'] for r in rows))} -> {path}")


def _row(pubid, query, doc, label, source):
    return {
        "pubid": pubid,
        "query": query,
        "doc_id": str(doc.get("id")),
        "title": doc.get("title", ""),
        "text": doc.get("text", ""),
        "label": label,
        "label_source": source,
    }


if __name__ == "__main__":
    main()
