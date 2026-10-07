"""Chapter 3 development step 5 input: filter training/validation pairs from the pqa_artificial training and validation queries.

  MD / GD / HD   the LLM (or human) labels of the annotated candidates (scripts/annotate_dataset.py)
  GD             each query's own PubMedQA context, its reference evidence by definition
  HD (optional)  synthetic harmful documents written for these queries (--synthetic), reported as an augmentation
"""

import argparse
import json
import os
import sys
from collections import Counter

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND)

from src.stage1_retriever.build_corpus import SPLITS_PATH, load_pqaa  # noqa: E402

LABELS = os.path.join(BACKEND, "data", "annotations", "pqaa_llm_labels.jsonl")
OUT_DIR = os.path.join(BACKEND, "data", "annotations")
VALID = {"ground_truth", "mediocre", "harmful"}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=LABELS, help="annotated pairs (human_label wins over llm_label)")
    parser.add_argument("--synthetic", nargs="*", default=[], help="synthetic HD files (generate_harmful_docs.py output)")
    parser.add_argument("--no-own-context", action="store_true", help="do not add each query's own context as GD")
    parser.add_argument("--prefix", default="pqaa_filter")
    args = parser.parse_args()

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    split_of = {p: s for s in ("train", "val") for p in splits[s]}
    rows = {"train": [], "val": []}

    for r in read_jsonl(args.labels):
        label = str(r.get("human_label") or r.get("llm_label") or "").lower()
        split = split_of.get(str(r["pubid"]))
        if label in VALID and split:
            rows[split].append({"pubid": r["pubid"], "query": r["query"], "doc_id": r["doc_id"], "text": r["text"],
                                "label": label, "label_source": "human" if r.get("human_label") else r.get("label_source", "llm")})

    if not args.no_own_context:
        full, _ = load_pqaa()
        for pubid, split in split_of.items():
            item = full[pubid]
            rows[split].append({"pubid": pubid, "query": item["QUESTION"], "doc_id": f"pqa_{pubid}",
                                "text": " ".join(item.get("CONTEXTS") or []), "label": "ground_truth",
                                "label_source": "own_context"})
        del full

    for path in args.synthetic:
        for d in read_jsonl(path) if path.endswith(".jsonl") else json.load(open(path, "r", encoding="utf-8")):
            split = split_of.get(str(d.get("target_pmid")))
            if split:
                rows[split].append({"pubid": d["target_pmid"], "query": d["target_question"], "doc_id": d["id"],
                                    "text": d["text"], "label": d.get("true_label", "harmful"),
                                    "label_source": f"synthetic:{d.get('generated_by')}"})

    for split, items in rows.items():
        path = os.path.join(OUT_DIR, f"{args.prefix}_{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split:5s}: {len(items):5d} pairs  labels {dict(Counter(r['label'] for r in items))}  "
              f"sources {dict(Counter(r['label_source'].split(':')[0] for r in items))} -> {path}")
    if not any(r["label"] == "harmful" for r in rows["train"]):
        print("\nNo harmful pairs in train: the filter cannot learn the HD class. Add --synthetic HD for the "
              "training queries and report it as a deviation (Chapter 3, Sources of Data).")


if __name__ == "__main__":
    main()
