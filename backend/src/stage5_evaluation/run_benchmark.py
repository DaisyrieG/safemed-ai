"""Stage 5: the experimental run (Chapter 3, Data Gathering Procedure steps 7 to 12)."""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.pipeline import PQAA_CORPUS, SafeMedPipeline
from src.stage1_retriever.build_corpus import SPLITS_PATH, load_pqaa
from src.stage4_generator.llm_client import chat_completion
from src.stage5_evaluation.judge import ClaimJudge
from src.stage5_evaluation.metrics import filter_metrics
from src.stage5_evaluation.statistical_tests import holm_adjust, wilcoxon_one_sample, wilcoxon_paired

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
H2_MARGIN = 0.05
H4_MARGIN = 0.10


class BenchmarkAbort(RuntimeError):
    """Raised when the run cannot produce valid, live results."""


def load_test_queries(split: str, n: Optional[int]) -> Tuple[List[Dict[str, str]], str]:
    """The test (or val) queries of the fixed seed-42 partition, with their reference evidence."""
    if not os.path.isfile(SPLITS_PATH):
        raise BenchmarkAbort(f"{SPLITS_PATH} not found. Run `python -m src.stage1_retriever.build_corpus` first.")
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        pubids = json.load(f)[split]
    pqaa, source = load_pqaa()
    records = []
    for pubid in pubids[: n or len(pubids)]:
        item = pqaa[pubid]
        records.append({
            "pubid": pubid,
            "question": item["QUESTION"].strip(),
            "context": " ".join(item.get("CONTEXTS") or []),
            "long_answer": item["LONG_ANSWER"].strip(),
            "final_decision": str(item.get("final_decision", "")).lower(),
        })
    return records, f"{source} [{split} split, {len(records)} queries]"


def load_pair_labels(path: Optional[str]) -> Dict[Tuple[str, str], str]:
    """Per-(query, document) gold labels: JSONL with pubid, doc_id and human_label / label / llm_label."""
    labels: Dict[Tuple[str, str], str] = {}
    if not path:
        return labels
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                labels[(str(row["pubid"]), str(row["doc_id"]))] = row.get("human_label") or row.get("label") or row.get("llm_label")
    return labels


def reference_evidence(rec: Dict[str, str]) -> str:
    """Chapter 3: the query's PubMedQA context, long answer and reference label."""
    return (f"Context: {rec['context']}\n\nConclusion: {rec['long_answer']}\n\n"
            f"Reference answer: {rec['final_decision']}")


def extract_decision(answer: str) -> str:
    m = re.search(r"decision:\s*(yes|no|maybe)", answer or "", re.IGNORECASE)
    return m.group(1).lower() if m else ""


def preflight(pipe: SafeMedPipeline, judge: ClaimJudge, pair_labels: Dict, allow_same_judge: bool) -> None:
    problems = []
    if pipe.filter is None or pipe.filter.scorer_name != "cross_encoder":
        problems.append("Stage 2 must use the fine-tuned cross-encoder.")
    for name, client, model in (("Stage 4 generator", pipe.generator.client, pipe.generator.model),
                                ("Stage 5 judge", judge.client, judge.model)):
        if client is None:
            problems.append(f"{name} ({model}) has no client.")
            continue
        try:
            chat_completion(client, model=model, messages=[{"role": "user", "content": "ping"}],
                            max_tokens=1, temperature=0.0)
        except Exception as e:
            problems.append(f"{name} ({model}) test call failed: {e}")
    if judge.model == pipe.generator.model and not allow_same_judge:
        problems.append(f"The judge model must differ from the generator (both are {judge.model}); "
                        "set SAFEMED_JUDGE_MODEL or pass --allow-same-judge to record it as a deviation.")
    n_harm = sum(1 for d in pipe.retriever.corpus if d.get("true_label") == "harmful")
    n_harm += sum(1 for v in pair_labels.values() if v == "harmful")
    if n_harm == 0:
        problems.append("No document is labelled 'harmful' in the corpus or --labels, so H1, H3 and SOP1 cannot be computed.")
    if problems:
        raise BenchmarkAbort("Not a live run:\n  - " + "\n  - ".join(problems))


def _ids(docs: List[Dict[str, Any]]) -> List[str]:
    return [str(d.get("id")) for d in docs]


def run_benchmark(args) -> Dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    queries, source = load_test_queries(args.split, args.n)
    pair_labels = load_pair_labels(args.labels)
    pipe = SafeMedPipeline(filter_model_path=args.filter_model, corpus_path=args.corpus)
    judge = ClaimJudge()
    preflight(pipe, judge, pair_labels, args.allow_same_judge)

    def label(pubid: str, doc: Dict[str, Any]) -> str:
        return pair_labels.get((pubid, str(doc.get("id")))) or doc.get("true_label") or "mediocre"

    def frac(pubid: str, docs: List[Dict[str, Any]], cls: str, denom: Optional[int] = None) -> float:
        d = denom if denom is not None else len(docs)
        return sum(1 for x in docs if label(pubid, x) == cls) / d if d else 0.0

    trace_path = os.path.join(args.output_dir, "benchmark_traces.jsonl")
    per_query, candidates_true, candidates_pred, candidates_prob, candidate_query = [], [], [], [], []

    print(f"\nSafeMed AI benchmark: {source}")
    print(f"Corpus: {args.corpus} | tau_safe = {pipe.filter.harmful_threshold} | "
          f"generator = {pipe.generator.model} | judge = {judge.model}\n")

    with open(trace_path, "w", encoding="utf-8") as trace:
        for i, rec in enumerate(queries, 1):
            pubid, q = rec["pubid"], rec["question"]
            res = pipe.run(q, exclude_id=pubid, generate_control_answer=False)
            fr = pipe.last_filter_result
            O1, O2, O3, O4 = fr.classified, fr.surviving, res["top5_documents"], res["answer"]
            control_top5 = pipe.last_control_top5

            if any(str(d.get("id")) == f"pqa_{pubid}" or str(d.get("pmid")) == pubid for d in O1):
                raise BenchmarkAbort(f"Leave-self-out violated at query {i} ({pubid}).")

            claims = judge.extract_claims(O4)
            verdicts = judge.verify_claims(claims, reference_evidence(rec)) if claims else []
            n_unsupported = sum(v["status"] == "UNSUPPORTED" for v in verdicts)
            n_contradicted = sum(v["status"] == "CONTRADICTED" for v in verdicts)

            post_ids = set(_ids(O2))
            gd_pre = [d for d in O1 if label(pubid, d) == "ground_truth"]
            row = {
                "pubid": pubid,
                "hd_density_pre": frac(pubid, O1, "harmful"),
                "hd_density_post": frac(pubid, O2, "harmful"),
                "gt_retention_loss": (sum(str(d.get("id")) not in post_ids for d in gd_pre) / len(gd_pre)) if gd_pre else 0.0,
                "hd_at5_control": frac(pubid, control_top5, "harmful", 5),
                "hd_at5_final": frac(pubid, O3, "harmful", 5),
                "hit_at5_control": int(frac(pubid, control_top5, "ground_truth") > 0),
                "hit_at5_final": int(frac(pubid, O3, "ground_truth") > 0),
                "unsupported_claim_rate": n_unsupported / len(verdicts) if verdicts else 0.0,
                "hallucinated": int(n_unsupported + n_contradicted > 0),
                "answer_correct": int(extract_decision(O4) == rec["final_decision"]),
                "n_claims": len(verdicts),
                "n_blocked": len(fr.blocked),
                "n_reinstated": len(fr.reinstated),
            }
            per_query.append(row)
            for d in O1:
                candidates_true.append("harmful" if label(pubid, d) == "harmful" else "not_harmful")
                candidates_pred.append("harmful" if d["is_blocked"] else "not_harmful")
                candidates_prob.append(d["harmful_probability"])
                candidate_query.append(i - 1)

            trace.write(json.dumps({
                "query_index": i, "pubid": pubid, "query": q, "excluded_source_id": pubid,
                "O1_pre_filter_pool": [{"id": str(d.get("id")), "retrieval_score": d.get("retrieval_score"),
                                        "p_hd": d.get("harmful_probability"), "is_blocked": d.get("is_blocked"),
                                        "gold_label": label(pubid, d)} for d in O1],
                "O2_post_filter_pool": _ids(O2),
                "O2_reinstated": fr.reinstated,
                "O3_final_context": [{"id": str(d.get("id")), "rerank_score": d.get("rerank_score"),
                                      "gold_label": label(pubid, d)} for d in O3],
                "control_top5": [{"id": str(d.get("id")), "gold_label": label(pubid, d)} for d in control_top5],
                "O4_answer": O4,
                "claims": verdicts,
                "metrics": row,
            }) + "\n")

            if i == 1 or i % args.log_interval == 0 or i == len(queries):
                print(f"[{i}/{len(queries)}] {q[:70]}")

    col = lambda k: [r[k] for r in per_query]
    h1 = wilcoxon_paired(col("hd_density_pre"), col("hd_density_post"), alternative="less")
    h2 = wilcoxon_one_sample(col("gt_retention_loss"), reference=H2_MARGIN, alternative="less")
    h3 = wilcoxon_paired(col("hd_at5_control"), col("hd_at5_final"), alternative="less")
    h4 = wilcoxon_one_sample(col("unsupported_claim_rate"), reference=H4_MARGIN, alternative="less")
    for h, p in zip((h1, h2, h3, h4), holm_adjust([h["p_value"] for h in (h1, h2, h3, h4)])):
        h["p_value_holm"] = p

    sop1 = filter_metrics(candidates_true, candidates_pred, candidates_prob)
    by_query = np.array(candidate_query)
    t = np.array([c == "harmful" for c in candidates_true])
    p = np.array([c == "harmful" for c in candidates_pred])
    rng = np.random.default_rng(args.seed)
    boot = {"TPR_Recall": [], "FPR": [], "Precision": [], "F1_score": []}
    for _ in range(args.n_bootstrap):
        mask = np.isin(by_query, rng.integers(0, len(per_query), len(per_query)))
        tp, fp = int((t & p & mask).sum()), int((~t & p & mask).sum())
        fn, tn = int((t & ~p & mask).sum()), int((~t & ~p & mask).sum())
        rec_ = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        boot["TPR_Recall"].append(rec_)
        boot["FPR"].append(fp / (fp + tn) if fp + tn else 0.0)
        boot["Precision"].append(prec)
        boot["F1_score"].append(2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0)
    sop1["ci_95"] = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
                     for k, v in boot.items()}
    sop1["confusion_matrix"] = {"TP": int((t & p).sum()), "FP": int((~t & p).sum()),
                                "FN": int((t & ~p).sum()), "TN": int((~t & ~p).sum())}

    mean = lambda k: round(float(np.mean(col(k))), 4)
    summary = {
        "live_run": True,
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_queries": len(per_query),
        "query_source": source,
        "corpus": args.corpus,
        "pair_labels": args.labels,
        "filter_model": pipe.filter.model_path,
        "tau_safe": pipe.filter.harmful_threshold,
        "generator_model": pipe.generator.model,
        "judge_model": judge.model,
        "judge_same_as_generator": judge.model == pipe.generator.model,
        "leave_self_out": True,
        "claim_reference": "PubMedQA context + LONG_ANSWER + final_decision",
        "means": {k: mean(k) for k in per_query[0] if k != "pubid"},
        "SOP1_filter_detection": sop1,
        "Hypotheses": {
            "H1_harmful_density_pre_vs_post": {**h1, "alternative": "post < pre"},
            "H2_gt_retention_loss_vs_margin": {**h2, "margin": H2_MARGIN, "alternative": "median loss < 0.05"},
            "H3_hd_at_top5_control_vs_final": {**h3, "alternative": "final < control"},
            "H4_unsupported_claim_rate_vs_margin": {**h4, "margin": H4_MARGIN, "alternative": "median rate < 0.10"},
        },
        "trace_file": os.path.relpath(trace_path, BACKEND),
    }
    out_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ WILCOXON SIGNED-RANK (zero_method='pratt') ================")
    print(f"N = {len(per_query)} | alpha = 0.05 | Holm-adjusted p in brackets")
    for name, h in (("H1 Harmful density pre vs post", h1), ("H2 GT retention loss vs 0.05", h2),
                    ("H3 HD@Top-5 control vs final", h3), ("H4 Unsupported claims vs 0.10", h4)):
        n = h.get("n_pairs", h.get("n_samples"))
        print(f"{name:34s} N={n:4d}  W={h['statistic']:>12}  p={h['p_value']:.6g} [{h['p_value_holm']:.4g}]"
              f"  mean shift={h['mean_diff']:+.4f}  median shift={h['median_diff']:+.4f}")
    print(f"SOP1: TPR {sop1['TPR_Recall']:.3f}  FPR {sop1['FPR']:.3f}  Precision {sop1['Precision']:.3f}  "
          f"F1 {sop1['F1_score']:.3f}  AUROC {sop1['AUROC']:.3f}")
    print(f"\nSaved: {os.path.relpath(out_file, BACKEND)} and {summary['trace_file']}\n")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test", choices=["test", "val"], help="query split (default test)")
    p.add_argument("--n", type=int, default=None, help="only the first N queries of the split (default all)")
    p.add_argument("--corpus", default=PQAA_CORPUS, help="retrieval corpus JSON (default the pooled pqa_artificial corpus)")
    p.add_argument("--labels", help="per-pair gold labels JSONL (pubid, doc_id, label) from the annotation step")
    p.add_argument("--filter-model", help="fine-tuned Stage 2 folder (default SAFEMED_FILTER_MODEL or models/safemed_filter)")
    p.add_argument("--output-dir", default=os.path.join(BACKEND, "results"))
    p.add_argument("--allow-same-judge", action="store_true", help="allow judge = generator (recorded as a deviation)")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=25)
    args = p.parse_args(argv)
    try:
        return run_benchmark(args)
    except BenchmarkAbort as e:
        print(f"\nBENCHMARK ABORTED: {e}\n", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
