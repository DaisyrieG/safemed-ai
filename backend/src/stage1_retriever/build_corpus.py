"""Pools the pqa_artificial abstracts into one corpus and splits the queries 1,000 / 250 / 500."""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT = os.path.dirname(BACKEND)
RAW_CANDIDATES = [
    os.path.join(BACKEND, "data", "raw", "ori_pqaa.json"),
    os.path.join(ROOT, "third_party", "pubmedqa", "data", "ori_pqaa.json"),
]
PROCESSED = os.path.join(BACKEND, "data", "processed")
CORPUS_PATH = os.path.join(PROCESSED, "pqaa_corpus.json")
SPLITS_PATH = os.path.join(PROCESSED, "pqaa_splits.json")

SPLIT_SIZES = {"train": 1000, "val": 250, "test": 500}
SEED = 42


def load_pqaa() -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Returns {pubid: instance} in the ori_pqaa.json format, and where it came from."""
    for path in RAW_CANDIDATES:
        if os.path.isfile(path):
            print(f"Reading {path} ...")
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path

    from datasets import load_dataset
    print("No local ori_pqaa.json; downloading pqa_artificial from Hugging Face (qiaojin/PubMedQA) ...")
    data = {}
    for row in load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train"):
        data[str(row["pubid"])] = {
            "QUESTION": row["question"],
            "CONTEXTS": row["context"]["contexts"],
            "LONG_ANSWER": row["long_answer"],
            "final_decision": row["final_decision"],
        }
    return data, "huggingface:qiaojin/PubMedQA/pqa_artificial"


def build_corpus(pqaa: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    corpus = []
    for pubid, item in pqaa.items():
        text = " ".join(item.get("CONTEXTS") or []).strip()
        if not text:
            continue
        corpus.append({
            "id": f"pqa_{pubid}",
            "pmid": str(pubid),
            "title": item["QUESTION"],
            "text": text,
            "source": "PubMed (PubMedQA pqa_artificial)",
        })
    return corpus


def partition_queries(pqaa: Dict[str, Dict[str, Any]], seed: int = SEED) -> Dict[str, List[str]]:
    """Non-overlapping train / val / test query sets by seeded simple random sampling."""
    eligible = sorted(p for p, item in pqaa.items() if item.get("QUESTION") and item.get("LONG_ANSWER"))
    drawn = random.Random(seed).sample(eligible, sum(SPLIT_SIZES.values()))
    splits, start = {}, 0
    for name, size in SPLIT_SIZES.items():
        splits[name] = drawn[start:start + size]
        start += size
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="keep only the first N documents (0 = all, for a quick test)")
    args = parser.parse_args()

    pqaa, source = load_pqaa()
    corpus = build_corpus(pqaa)
    if args.limit:
        corpus = corpus[: args.limit]
    splits = partition_queries(pqaa)

    os.makedirs(PROCESSED, exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    with open(SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump({"source": source, "seed": SEED, **splits}, f, indent=1)

    print(f"Wrote {len(corpus):,} documents to {CORPUS_PATH}")
    print(f"Wrote query splits {({k: len(v) for k, v in splits.items()})} to {SPLITS_PATH}")


if __name__ == "__main__":
    main()
