"""Sprint step 1 / Chapter 3 development step 4: LLM annotation of (query, document) pairs."""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND)

from src.annotation.llm_labeler import label_pairs, make_labeler_client  # noqa: E402
from src.pipeline import PQAA_CORPUS  # noqa: E402
from src.stage1_retriever.build_corpus import SPLITS_PATH, load_pqaa  # noqa: E402
from src.stage1_retriever.retriever import BiEncoderRetriever, index_path_for  # noqa: E402

OUTPUT = os.path.join(BACKEND, "data", "annotations", "pqaa_llm_labels.jsonl")
RANK_BANDS = [(1, 5), (6, 10), (11, 20), (21, 35), (36, 50)]
SEED = 42


def sample_pairs(retriever, pqaa, splits, rng):
    """One candidate per train/val query, rank band cycling across queries."""
    queries = [(p, "train") for p in splits["train"]] + [(p, "val") for p in splits["val"]]
    pairs = []
    for i, (pubid, split) in enumerate(queries):
        item = pqaa[pubid]
        candidates = retriever.retrieve(item["QUESTION"], top_k=50, exclude_id=pubid)
        lo, hi = RANK_BANDS[i % len(RANK_BANDS)]
        rank = rng.randint(lo, min(hi, len(candidates)))
        doc = candidates[rank - 1]
        pairs.append({
            "pubid": pubid,
            "split": split,
            "query": item["QUESTION"],
            "doc_id": doc["id"],
            "text": doc["text"],
            "retrieval_rank": rank,
            "retrieval_score": round(doc["retrieval_score"], 4),
            "reference_answer": f"{item.get('final_decision', '')}. {item['LONG_ANSWER']}",
        })
        if (i + 1) % 250 == 0:
            print(f"  retrieved candidates for {i + 1}/{len(queries)} queries")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="label only the first N pairs (quick test)")
    parser.add_argument("--batch-size", type=int, default=10, help="pairs per request (default 10)")
    parser.add_argument("--pace", type=float, default=40.0,
                        help="minimum seconds between requests, to stay under the tokens-per-minute limit")
    args = parser.parse_args()

    if not os.path.exists(index_path_for(PQAA_CORPUS)):
        sys.exit(f"FAISS index {index_path_for(PQAA_CORPUS)} not found; run `python -m src.stage1_retriever.build_index` first.")
    client, model = make_labeler_client()
    print(f"Labelling model: {model}")

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    pqaa, _ = load_pqaa()
    retriever = BiEncoderRetriever(corpus_path=PQAA_CORPUS)

    print("Retrieving candidates ...")
    pairs = sample_pairs(retriever, pqaa, splits, random.Random(SEED))
    if args.limit:
        pairs = pairs[: args.limit]

    done = set()
    if os.path.exists(OUTPUT):
        with open(OUTPUT, "r", encoding="utf-8") as f:
            done = {json.loads(line)["pubid"] for line in f if line.strip()}
    todo = [p for p in pairs if p["pubid"] not in done]
    print(f"{len(done)} pairs already labelled, {len(todo)} to go "
          f"(~{len(todo) / args.batch_size * args.pace / 60:.0f} min)")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    for start in range(0, len(todo), args.batch_size):
        began = time.time()
        batch = todo[start:start + args.batch_size]
        labelled = label_pairs(batch, client=client, model=model, batch_size=args.batch_size)
        with open(OUTPUT, "a", encoding="utf-8") as f:
            for row in labelled:
                f.write(json.dumps({**row, "label_source": f"llm:{model}"}, ensure_ascii=False) + "\n")
        print(f"  labelled {min(start + args.batch_size, len(todo))}/{len(todo)}: "
              f"{dict(Counter(r['llm_label'] for r in labelled))}")
        if start + args.batch_size < len(todo):
            time.sleep(max(0.0, args.pace - (time.time() - began)))

    with open(OUTPUT, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"\nDone: {len(rows)} labelled pairs in {OUTPUT}")
    print("Label counts:", dict(Counter(r["llm_label"] for r in rows)))


if __name__ == "__main__":
    main()
