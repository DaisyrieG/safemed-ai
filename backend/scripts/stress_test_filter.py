"""Measures the current Stage 2 filter on the old (v1) and realistic (v2) synthetic HD for its held-out test questions."""

import argparse
import json
import os
import statistics
import sys

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

from generate_harmful_docs import CONCLUSION, DEMO_HARMFUL, FILTER_TEST, OUTPUT, PQAL, STYLE_FLAGS  # noqa: E402
from src.pipeline import resolve_filter_model_path  # noqa: E402
from src.stage2_filter.cross_encoder_filter import HarmfulDocumentFilter  # noqa: E402

RESULT = os.path.join(BACKEND, "results", "hd_stress_test.json")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarise(name: str, rows: list) -> dict:
    """Block rate, P(HD) and style statistics for one document set."""
    if not rows:
        return {"set": name, "n": 0}
    return {
        "set": name,
        "n": len(rows),
        "blocked_rate": round(sum(r["blocked"] for r in rows) / len(rows), 4),
        "median_p_hd": round(statistics.median(r["p_hd"] for r in rows), 4),
        "median_words": statistics.median(r["words"] for r in rows),
        "style_word_rate": round(sum(r["style_word"] for r in rows) / len(rows), 4),
        "conclusion_phrase_rate": round(sum(r["conclusion"] for r in rows) / len(rows), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", default=OUTPUT, help="realistic HD file from generate_harmful_docs.py")
    parser.add_argument("--filter-model", default=None, help="filter folder (default SAFEMED_FILTER_MODEL or models/safemed_filter)")
    parser.add_argument("--output", default=RESULT)
    parser.add_argument("--extra", nargs="*", default=[], metavar="NAME=PATH",
                        help="more held-out sets, e.g. hd_gpt4omini=results/diag_test_hd_gpt4omini.json")
    args = parser.parse_args()

    with open(FILTER_TEST, "r", encoding="utf-8") as f:
        test_pmids = set(json.loads(line)["pubid"] for line in f if line.strip())
    pqal = load_json(PQAL)
    sets = {
        "real_source_abstracts (GD)": [
            {"id": f"pqa_{p}", "target_pmid": p, "text": " ".join(pqal[p]["CONTEXTS"])} for p in sorted(test_pmids) if p in pqal
        ],
        "v1_synthetic_HD": [d for d in load_json(DEMO_HARMFUL) if d["target_pmid"] in test_pmids],
    }
    if os.path.exists(args.v2):
        sets["v2_realistic_HD"] = [d for d in load_json(args.v2) if d["target_pmid"] in test_pmids]
    else:
        print(f"No v2 file at {args.v2}; run scripts/generate_harmful_docs.py first. Scoring v1 only.")

    for spec in args.extra:
        name, path = spec.split("=", 1)
        sets[name] = [d for d in load_json(path) if d["target_pmid"] in test_pmids]

    harm_filter = HarmfulDocumentFilter(model_path=args.filter_model or resolve_filter_model_path())
    print(f"Filter: {harm_filter.model_path} | tau_safe = {harm_filter.harmful_threshold}\n")

    summaries, per_document = [], {}
    for name, docs in sets.items():
        rows = []
        for doc in docs:
            scored = harm_filter.classify(pqal[doc["target_pmid"]]["QUESTION"], [doc])[0]
            rows.append({
                "id": doc["id"],
                "target_pmid": doc["target_pmid"],
                "p_hd": scored["harmful_probability"],
                "blocked": bool(scored["is_blocked"]),
                "words": len(doc["text"].split()),
                "style_word": bool(STYLE_FLAGS.search(doc["text"])),
                "conclusion": bool(CONCLUSION.search(doc["text"])),
            })
        summaries.append(summarise(name, rows))
        per_document[name] = rows

    print(f"{'set':28s} {'n':>4s} {'blocked':>8s} {'med P(HD)':>10s} {'words':>6s} {'style':>6s} {'concl.':>7s}")
    for s in summaries:
        if s["n"]:
            print(f"{s['set']:28s} {s['n']:4d} {s['blocked_rate']:8.1%} {s['median_p_hd']:10.3f} "
                  f"{s['median_words']:6.0f} {s['style_word_rate']:6.0%} {s['conclusion_phrase_rate']:7.0%}")
    print("\nFor HD sets 'blocked' is the recall; for GD sets it is the false-block rate.")

    from sklearn.metrics import roc_auc_score
    gd_sets = [n for n in per_document if "GD" in n or n.startswith("gd")]
    hd_sets = [n for n in per_document if n not in gd_sets]
    for g in gd_sets:
        for h in hd_sets:
            neg, pos = per_document[g], per_document[h]
            if neg and pos:
                auc = roc_auc_score([0] * len(neg) + [1] * len(pos), [r["p_hd"] for r in neg + pos])
                print(f"AUROC {h} vs {g}: {auc:.3f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"filter_model": harm_filter.model_path, "tau_safe": harm_filter.harmful_threshold,
                   "summary": summaries, "documents": per_document}, f, indent=1)
    print(f"Saved {os.path.relpath(args.output, BACKEND)}")


if __name__ == "__main__":
    main()
