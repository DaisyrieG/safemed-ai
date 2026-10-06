"""Builds the demo retrieval corpus from the 1,000 expert-labelled PubMedQA questions (PQA-L)."""

import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE = os.path.join(ROOT, "third_party", "pubmedqa", "data", "ori_pqal.json")
OUTPUT = os.path.join(ROOT, "backend", "data", "demo", "pubmedqa_demo_corpus.json")


def main() -> None:
    with open(SOURCE, "r", encoding="utf-8") as f:
        pqal = json.load(f)

    corpus = []
    for pmid, item in pqal.items():
        corpus.append({
            "id": f"pqa_{pmid}",
            "pmid": str(pmid),
            "title": item["QUESTION"],
            "text": " ".join(item["CONTEXTS"]),
            "source": "PubMed (PubMedQA expert-labelled)",
            "year": item.get("YEAR"),
            "final_decision": item.get("final_decision"),
            "long_answer": item.get("LONG_ANSWER"),
            "true_label": "mediocre",
        })

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=1)
    print(f"Wrote {len(corpus)} PubMedQA abstracts to {OUTPUT}")


if __name__ == "__main__":
    main()
