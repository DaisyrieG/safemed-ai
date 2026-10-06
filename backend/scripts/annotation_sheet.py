"""Blind two-annotator labelling sheet and Cohen's kappa for the filter's training labels."""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANNOTATIONS = os.path.join(BACKEND, "data", "annotations")
SHEET = os.path.join(ANNOTATIONS, "annotation_sheet.csv")
KEY = os.path.join(ANNOTATIONS, "annotation_key.json")
REPORT = os.path.join(ANNOTATIONS, "annotation_agreement.json")
HUMAN_LABELS = os.path.join(ANNOTATIONS, "human_labels.jsonl")

CODES = {"GD": "ground_truth", "MD": "mediocre", "HD": "harmful"}
NAMES = {v: k for k, v in CODES.items()}
KAPPA_THRESHOLD = 0.80
COLUMNS = ["item", "question", "document_title", "document_text", "annotator_A", "annotator_B", "notes"]


def sample(n: int, seed: int) -> None:
    rows = []
    for split in ("train", "val", "test"):
        path = os.path.join(ANNOTATIONS, f"filter_{split}.jsonl")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                rows += [json.loads(line) for line in f if line.strip()]
    if not rows:
        sys.exit("No filter_*.jsonl files. Run backend/scripts/build_filter_training_data.py first.")

    rng = random.Random(seed)
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    per_label = n // len(CODES)
    picked = []
    for label in CODES.values():
        pool = by_label.get(label, [])
        picked += rng.sample(pool, min(per_label, len(pool)))
    rng.shuffle(picked)

    os.makedirs(ANNOTATIONS, exist_ok=True)
    with open(SHEET, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for i, row in enumerate(picked, 1):
            writer.writerow([i, row["query"], row.get("title", ""), row.get("text", ""), "", "", ""])
    with open(KEY, "w", encoding="utf-8") as f:
        json.dump({str(i): {k: row.get(k) for k in ("pubid", "doc_id", "label", "label_source", "query", "title", "text")}
                   for i, row in enumerate(picked, 1)}, f, indent=1, ensure_ascii=False)
    counts = Counter(NAMES[r["label"]] for r in picked)
    print(f"Wrote {len(picked)} pairs to {SHEET} ({dict(counts)}); automatic labels kept in {KEY}")
    print("Annotators fill annotator_A / annotator_B with GD, MD or HD. Don't share the key file with them.")


def _code(value: str, item: str, column: str) -> str:
    value = (value or "").strip().upper()
    if value not in CODES:
        raise ValueError(f"item {item}: {column} is {value!r}; use GD, MD or HD")
    return value


def score(sheet_path: str) -> None:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    with open(KEY, "r", encoding="utf-8") as f:
        key = json.load(f)
    with open(sheet_path, "r", encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))

    a, b, auto, items, skipped = [], [], [], [], []
    for row in sheet:
        item = str(row.get("item", "")).strip()
        if not (row.get("annotator_A") or "").strip() or not (row.get("annotator_B") or "").strip():
            skipped.append(item)
            continue
        a.append(_code(row["annotator_A"], item, "annotator_A"))
        b.append(_code(row["annotator_B"], item, "annotator_B"))
        auto.append(NAMES[key[item]["label"]])
        items.append(item)
    if not items:
        sys.exit("No row has both annotator_A and annotator_B filled in.")

    labels = list(CODES)
    agree = lambda x, y: sum(p == q for p, q in zip(x, y)) / len(x)
    kappa_ab = float(cohen_kappa_score(a, b, labels=labels))
    report = {
        "n_pairs": len(items),
        "skipped_items": skipped,
        "label_counts": {"annotator_A": dict(Counter(a)), "annotator_B": dict(Counter(b)), "automatic": dict(Counter(auto))},
        "kappa_A_vs_B": round(kappa_ab, 4),
        "percent_agreement_A_vs_B": round(agree(a, b), 4),
        "meets_threshold": kappa_ab >= KAPPA_THRESHOLD,
        "threshold": KAPPA_THRESHOLD,
        "kappa_A_vs_automatic": round(float(cohen_kappa_score(a, auto, labels=labels)), 4),
        "kappa_B_vs_automatic": round(float(cohen_kappa_score(b, auto, labels=labels)), 4),
        "accuracy_automatic_vs_agreed": None,
        "confusion_A_rows_vs_B_cols": {"labels": labels, "matrix": confusion_matrix(a, b, labels=labels).tolist()},
        "disagreements": [{"item": i, "A": x, "B": y, "automatic": z} for i, x, y, z in zip(items, a, b, auto) if x != y],
    }
    agreed = [(i, x, z) for i, x, y, z in zip(items, a, b, auto) if x == y]
    if agreed:
        report["accuracy_automatic_vs_agreed"] = round(sum(x == z for _, x, z in agreed) / len(agreed), 4)

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    with open(HUMAN_LABELS, "w", encoding="utf-8") as f:
        for item, code, _ in agreed:
            k = key[item]
            f.write(json.dumps({"pubid": k["pubid"], "query": k["query"], "doc_id": k["doc_id"], "title": k["title"],
                                "text": k["text"], "label": k["label"], "human_label": CODES[code],
                                "label_source": "human_agreed"}, ensure_ascii=False) + "\n")

    print(f"Pairs scored: {len(items)} (skipped {len(skipped)} with an empty column)")
    print(f"Cohen's kappa, annotator A vs B: {kappa_ab:.3f}  "
          f"({'meets' if report['meets_threshold'] else 'below'} the {KAPPA_THRESHOLD} threshold); "
          f"raw agreement {report['percent_agreement_A_vs_B']:.1%}")
    print(f"Kappa vs automatic labels: A {report['kappa_A_vs_automatic']:.3f}, B {report['kappa_B_vs_automatic']:.3f}")
    if report["accuracy_automatic_vs_agreed"] is not None:
        print(f"Automatic labels match the agreed human label on {report['accuracy_automatic_vs_agreed']:.1%} "
              f"of {len(agreed)} agreed pairs")
    print(f"{len(report['disagreements'])} disagreements to discuss: items "
          f"{', '.join(d['item'] for d in report['disagreements']) or 'none'}")
    print(f"Saved {REPORT} and {HUMAN_LABELS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="write the blind annotation sheet")
    s.add_argument("--n", type=int, default=102, help="pairs to sample (split evenly over GD, MD, HD)")
    s.add_argument("--seed", type=int, default=42)
    c = sub.add_parser("score", help="compute Cohen's kappa from the filled sheet")
    c.add_argument("--sheet", default=SHEET, help="the filled sheet, saved as CSV")
    args = parser.parse_args()
    if args.command == "sample":
        sample(args.n, args.seed)
    else:
        score(args.sheet)


if __name__ == "__main__":
    main()
