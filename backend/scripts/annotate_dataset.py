"""Sprint step 1 / Chapter 3 development step 4: LLM annotation of (query, document) pairs."""

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import Counter

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND)

from dotenv import load_dotenv  # noqa: E402

from src.annotation.llm_labeler import label_pairs, make_labeler_client  # noqa: E402
from src.stage4_generator.llm_client import make_client  # noqa: E402

load_dotenv(os.path.join(BACKEND, ".env"))
from src.pipeline import PQAA_CORPUS  # noqa: E402
from src.stage1_retriever.build_corpus import SPLITS_PATH, load_pqaa  # noqa: E402
from src.stage1_retriever.retriever import BiEncoderRetriever, index_path_for  # noqa: E402

OUTPUT = os.path.join(BACKEND, "data", "annotations", "pqaa_llm_labels.jsonl")
RANK_BANDS = [(1, 5), (6, 10), (11, 20), (21, 35), (36, 50)]
SEED = 42
N_ANNOT = 1500


def sample_pairs(retriever, pqaa, splits, rng, n_pairs: int = N_ANNOT):
    """n_pairs candidates stratified by retriever rank: one per train/val query, then a second, different-band
    candidate for the first queries until n_pairs is reached (Chapter 3: n_annot = 1,500)."""
    queries = [(p, "train") for p in splits["train"]] + [(p, "val") for p in splits["val"]]
    slots = [(i, q, 0) for i, q in enumerate(queries)]
    slots += [(i, queries[i], 2) for i in range(max(0, n_pairs - len(queries)))]
    pairs, cache, taken = [], {}, set()
    for n, (i, (pubid, split), shift) in enumerate(slots[:n_pairs]):
        item = pqaa[pubid]
        if pubid not in cache:
            cache[pubid] = retriever.retrieve(item["QUESTION"], top_k=50, exclude_id=pubid)
        candidates = cache[pubid]
        lo, hi = RANK_BANDS[(i + shift) % len(RANK_BANDS)]
        options = [r for r in range(lo, min(hi, len(candidates)) + 1) if (pubid, candidates[r - 1]["id"]) not in taken]
        if not options:
            continue
        rank = rng.choice(options)
        doc = candidates[rank - 1]
        taken.add((pubid, doc["id"]))
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
        if (n + 1) % 250 == 0:
            print(f"  sampled {n + 1}/{n_pairs} pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="label only the first N pairs (quick test)")
    parser.add_argument("--openai", action="store_true",
                        help="label with GPT-4o-mini via OPENAI_API_KEY (Chapter 3 labeller), ignoring SAFEMED_LABELER_*")
    parser.add_argument("--output", default=OUTPUT, help="labels JSONL (resumed if it exists)")
    parser.add_argument("--pairs", type=int, default=N_ANNOT, help=f"pairs to sample (default {N_ANNOT}, Chapter 3)")
    parser.add_argument("--batch-size", type=int, default=10, help="pairs per request (default 10)")
    parser.add_argument("--pace", type=float, default=40.0,
                        help="minimum seconds between requests, to stay under the tokens-per-minute limit")
    parser.add_argument("--workers", type=int, default=1, help="batches labelled in parallel (OpenAI: 6 is fine)")
    args = parser.parse_args()

    if not os.path.exists(index_path_for(PQAA_CORPUS)):
        sys.exit(f"FAISS index {index_path_for(PQAA_CORPUS)} not found; run `python -m src.stage1_retriever.build_index` first.")
    if args.openai:
        client, error = make_client(os.getenv("OPENAI_API_KEY"))
        if client is None:
            sys.exit(f"GPT-4o-mini unavailable: {error}. Add OPENAI_API_KEY to backend/.env.")
        model = "gpt-4o-mini"
    else:
        client, model = make_labeler_client()
    print(f"Labelling model: {model}")

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    full, _ = load_pqaa()
    needed = set(splits["train"]) | set(splits["val"])
    pqaa = {k: {f: full[k].get(f) for f in ("QUESTION", "LONG_ANSWER", "final_decision")} for k in needed}
    del full
    gc.collect()
    retriever = BiEncoderRetriever(corpus_path=PQAA_CORPUS)

    print("Retrieving candidates ...")
    pairs = sample_pairs(retriever, pqaa, splits, random.Random(SEED), args.pairs)
    if args.limit:
        pairs = pairs[: args.limit]

    done = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            done = {(r["pubid"], r["doc_id"]) for r in map(json.loads, filter(str.strip, f))}
    todo = [p for p in pairs if (p["pubid"], p["doc_id"]) not in done]
    print(f"{len(done)} pairs already labelled, {len(todo)} to go "
          f"(~{len(todo) / args.batch_size * args.pace / 60:.0f} min)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]

    def run(batch):
        began = time.time()
        labelled = label_pairs(batch, client=client, model=model, batch_size=args.batch_size)
        time.sleep(max(0.0, args.pace - (time.time() - began)))
        return labelled

    from concurrent.futures import ThreadPoolExecutor
    done_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for labelled in pool.map(run, batches):
            with open(args.output, "a", encoding="utf-8") as f:
                for row in labelled:
                    f.write(json.dumps({**row, "label_source": f"llm:{model}"}, ensure_ascii=False) + "\n")
            done_count += len(labelled)
            print(f"  labelled {done_count}/{len(todo)}: {dict(Counter(r['llm_label'] for r in labelled))}")

    with open(args.output, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"\nDone: {len(rows)} labelled pairs in {args.output}")
    print("Label counts:", dict(Counter(r["llm_label"] for r in rows)))


if __name__ == "__main__":
    main()
