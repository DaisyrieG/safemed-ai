"""Stage 5: the benchmark experiment comparing the control and treatment conditions.

Phases, each resumable and written to --output-dir:
  generate     both conditions for every test query
  judge        answers coded, stripped of their condition and judged in random order
  reliability  judge re-scoring, answer regeneration and filter-seed stability
  analyze      outcome measures and the pre-specified tests of H1-H4: McNemar mid-p and a paired permutation test
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.pipeline import PQAA_CORPUS, SafeMedPipeline
from src.stage1_retriever.build_corpus import SPLITS_PATH, load_pqaa, load_pqal
from src.stage4_generator.llm_client import chat_completion
from src.stage5_evaluation.judge import ClaimJudge
from src.stage5_evaluation.live_check import check_answer, extract_decision
from src.stage5_evaluation.metrics import filter_metrics
from src.stage5_evaluation.statistical_tests import mcnemar_midp, newcombe_paired_ci, paired_permutation_test

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONDITIONS = ("control", "treatment")
TRACES = "benchmark_traces.jsonl"
BLINDING = "blinding_key.json"
JUDGMENTS = "judgments.jsonl"
RESCORES = "reliability_rescore.jsonl"
REGENERATIONS = "reliability_regeneration.jsonl"
RESULTS = "evaluation_results.json"


class BenchmarkAbort(RuntimeError):
    """Raised when the run cannot produce valid, live results."""


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_test_queries(split: str, n: Optional[int]) -> Tuple[List[Dict[str, str]], str]:
    """The queries with their reference evidence: test = all 1,000 PQA-L questions (official_test marks the 500 of
    the official test set), val = the validation queries of the fixed seed-42 pqa_artificial partition."""
    if split == "test":
        pqaa, source = load_pqal()
        pubids = list(pqaa)
    else:
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
            "official_test": bool(item.get("official_test", False)),
        })
    return records, f"{source} [{split} split, {len(records)} queries]"


def load_pair_labels(path: Optional[str]) -> Dict[Tuple[str, str], str]:
    """Per-(query, document) gold labels: JSONL with pubid, doc_id and human_label / label / llm_label."""
    labels: Dict[Tuple[str, str], str] = {}
    if not path:
        return labels
    for row in read_jsonl(path):
        labels[(str(row["pubid"]), str(row["doc_id"]))] = row.get("human_label") or row.get("label") or row.get("llm_label")
    return labels


def reference_evidence(rec: Dict[str, str]) -> str:
    """Reference evidence: the query's PubMedQA context, long answer and reference label only, the same for both
    conditions (no retrieved documents, so the reference does not depend on what a condition retrieved)."""
    parts = [f"Context: {rec['context']}", f"Conclusion: {rec['long_answer']}", f"Reference answer: {rec['final_decision']}"]
    return "\n\n".join(parts)


def preflight(pipe: SafeMedPipeline, judge: ClaimJudge, pair_labels: Dict, allow_same_judge: bool,
              pilot: bool = False) -> None:
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
    if n_harm == 0 and not pilot:
        problems.append("No document is labelled 'harmful' in the corpus or --labels, so H1, H3 and SOP1 cannot be computed.")
    if problems:
        raise BenchmarkAbort("Not a live run:\n  - " + "\n  - ".join(problems))


def _doc(d: Dict[str, Any], gold: str, **extra) -> Dict[str, Any]:
    return {"id": str(d.get("id")), "title": d.get("title", ""), "text": d.get("text", ""), "gold_label": gold, **extra}


def generate(pipe: SafeMedPipeline, queries: List[Dict[str, str]], label, out_dir: str, log_interval: int) -> List[Dict]:
    """Retrieves once, runs both conditions from the same 50 candidates and logs every stage boundary (O1-O4)."""
    path = os.path.join(out_dir, TRACES)
    done = {r["pubid"] for r in read_jsonl(path)}
    todo = [q for q in queries if q["pubid"] not in done]
    print(f"Generate: {len(done)} queries already done, {len(todo)} to go")
    for i, rec in enumerate(todo, 1):
        pubid, q = rec["pubid"], rec["question"]
        res = pipe.run(q, exclude_id=pubid, generate_control_answer=True)
        fr = pipe.last_filter_result
        if any(str(d.get("id")) == f"pqa_{pubid}" or str(d.get("pmid")) == pubid for d in fr.classified):
            raise BenchmarkAbort(f"Leave-self-out violated at query {pubid}.")
        append_jsonl(path, {
            "pubid": pubid,
            "query": q,
            "generator": pipe.generator.model,
            "tau_safe": pipe.filter.harmful_threshold,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "O1_pre_filter_pool": [{"id": str(d.get("id")), "retrieval_score": d.get("retrieval_score"),
                                    "p_hd": d.get("harmful_probability"), "is_blocked": bool(d.get("is_blocked")),
                                    "gold_label": label(pubid, d)} for d in fr.classified],
            "O2_post_filter_pool": [str(d.get("id")) for d in fr.surviving],
            "O2_reinstated": [str(r.get("id")) for r in fr.reinstated],
            "treatment": {"top5": [_doc(d, label(pubid, d), rerank_score=d.get("rerank_score")) for d in res["top5_documents"]],
                          "answer": res["answer"]},
            "control": {"top5": [_doc(d, label(pubid, d), rerank_score=d.get("rerank_score")) for d in pipe.last_control_top5],
                        "answer": res["evaluation_breakdown"]["control"]["answer"]},
        })
        if i == 1 or i % log_interval == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {q[:70]}")
    return read_jsonl(path)


def blind(traces: List[Dict], out_dir: str, seed: int) -> Dict[str, Any]:
    """Blinding: one random code per answer; the judge sees codes in random order and never the condition."""
    path = os.path.join(out_dir, BLINDING)
    blinding = {"key": {}, "order": []}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            blinding = json.load(f)
    have = {(v["pubid"], v["condition"]) for v in blinding["key"].values()}
    rng = random.Random(f"{seed}-{len(blinding['order'])}")
    new = []
    for row in traces:
        for cond in CONDITIONS:
            if (row["pubid"], cond) not in have:
                code = f"{rng.getrandbits(40):010x}"
                while code in blinding["key"]:
                    code = f"{rng.getrandbits(40):010x}"
                blinding["key"][code] = {"pubid": row["pubid"], "condition": cond}
                new.append(code)
    rng.shuffle(new)
    blinding["order"] += new
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blinding, f, indent=1)
    return blinding


def _context(top5: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"id": d["id"], "text": d["text"], "true_label": d["gold_label"]} for d in top5]


def judge_answer(judge: ClaimJudge, rec: Dict[str, str], cond: Dict[str, Any]) -> Dict[str, Any]:
    context = _context(cond["top5"])
    return check_answer(judge, cond["answer"], reference_evidence(rec), context)


def judge_all(judge: ClaimJudge, blinding: Dict, traces: Dict[str, Dict], records: Dict[str, Dict],
              out_dir: str, pace: float, judged_ids: Optional[set] = None) -> Dict[str, Dict]:
    """Judges every blinded answer: claims, claim labels, answer-level flag and harmful-document attribution."""
    path = os.path.join(out_dir, JUDGMENTS)
    done = {r["code"]: r for r in read_jsonl(path)}
    todo = [c for c in blinding["order"] if c not in done
            and (judged_ids is None or blinding["key"][c]["pubid"] in judged_ids)]
    print(f"Judge ({judge.model}): {len(done)} answers already judged, {len(todo)} to go")
    for i, code in enumerate(todo, 1):
        info = blinding["key"][code]
        row = traces[info["pubid"]]
        result = judge_answer(judge, records[info["pubid"]], row[info["condition"]])
        done[code] = {"code": code, **result}
        append_jsonl(path, done[code])
        if i % 25 == 0 or i == len(todo):
            print(f"  judged {i}/{len(todo)}")
        if pace:
            time.sleep(pace)
    return done


def _kappa(a: List[bool], b: List[bool]) -> Optional[float]:
    from sklearn.metrics import cohen_kappa_score
    if not a or len(set(a) | set(b)) < 2:
        return None
    return round(float(cohen_kappa_score(a, b)), 4)


def reliability(args, pipe: SafeMedPipeline, judge: ClaimJudge, blinding: Dict, judgments: Dict[str, Dict],
                traces: Dict[str, Dict], records: Dict[str, Dict], out_dir: str) -> Dict[str, Any]:
    """Reliability checks: judge consistency (re-scoring), generator stability (regeneration) and filter stability (seeds)."""
    rng = random.Random(args.seed + 1)
    codes = [c for c in blinding["order"] if c in judgments]
    rescore_codes = rng.sample(codes, min(len(codes), round(args.rescore_frac * len(codes))))
    path = os.path.join(out_dir, RESCORES)
    rescored = {r["code"]: r for r in read_jsonl(path)}
    for code in rescore_codes:
        if code not in rescored:
            info = blinding["key"][code]
            rescored[code] = {"code": code, **judge_answer(judge, records[info["pubid"]], traces[info["pubid"]][info["condition"]])}
            append_jsonl(path, rescored[code])
    first = [judgments[c]["hallucinated"] for c in rescore_codes]
    second = [rescored[c]["hallucinated"] for c in rescore_codes]
    judge_consistency = {
        "n_answers": len(rescore_codes),
        "flag_agreement": round(float(np.mean([a == b for a, b in zip(first, second)])), 4) if first else None,
        "flag_kappa": _kappa(first, second),
        "mean_abs_ucr_difference": round(float(np.mean([abs(judgments[c]["unsupported_claim_rate"]
                                                            - rescored[c]["unsupported_claim_rate"]) for c in rescore_codes])), 4)
        if rescore_codes else None,
    }

    judged_pubids = {blinding["key"][c]["pubid"] for c in codes}
    pubids = sorted(p for p in traces if p in judged_pubids)
    regen_ids = rng.sample(pubids, min(len(pubids), round(args.regen_frac * len(pubids))))
    path = os.path.join(out_dir, REGENERATIONS)
    regenerated = {r["pubid"]: r for r in read_jsonl(path)}
    by_pair = {(v["pubid"], v["condition"]): c for c, v in blinding["key"].items()}
    for pubid in regen_ids:
        if pubid in regenerated:
            continue
        cond = traces[pubid]["treatment"]
        answer = pipe.generator.generate_answer(traces[pubid]["query"], cond["top5"])
        result = judge_answer(judge, records[pubid], {"top5": cond["top5"], "answer": answer})
        regenerated[pubid] = {"pubid": pubid, "answer": answer, "same_text": answer.strip() == cond["answer"].strip(),
                              "decision": extract_decision(answer), "hallucinated": result["hallucinated"]}
        append_jsonl(path, regenerated[pubid])
    original = {p: judgments[by_pair[(p, "treatment")]] for p in regen_ids if by_pair.get((p, "treatment")) in judgments}
    generator_stability = {
        "n_queries": len(original),
        "identical_text": round(float(np.mean([regenerated[p]["same_text"] for p in original])), 4) if original else None,
        "decision_agreement": round(float(np.mean([regenerated[p]["decision"] == original[p]["decision"] for p in original])), 4)
        if original else None,
        "flag_agreement": round(float(np.mean([regenerated[p]["hallucinated"] == original[p]["hallucinated"] for p in original])), 4)
        if original else None,
    }

    seeds_path = os.path.join(pipe.filter.model_path, "seeds_summary.json")
    filter_stability = None
    if os.path.exists(seeds_path):
        with open(seeds_path, "r", encoding="utf-8") as f:
            filter_stability = json.load(f).get("stability")
    return {"judge_consistency": judge_consistency, "generator_stability": generator_stability,
            "filter_stability": filter_stability or "seeds_summary.json not found: train with --seeds 42 43 44"}


def analyze(traces: Dict[str, Dict], blinding: Dict, judgments: Dict[str, Dict], records: Dict[str, Dict],
            args, extra: Dict[str, Any]) -> Dict[str, Any]:
    """Per-query outcomes and the pre-specified tests of H1-H4: McNemar mid-p for the binary outcomes and a
    paired permutation test for HD@Top-5, all two-sided."""
    by_pair = {(v["pubid"], v["condition"]): c for c, v in blinding["key"].items()}
    per_query, cand_true, cand_pred, cand_prob, cand_query = [], [], [], [], []
    for qi, (pubid, row) in enumerate(sorted(traces.items())):
        o1 = row["O1_pre_filter_pool"]
        post = set(row["O2_post_filter_pool"])
        reinstated = set(row["O2_reinstated"])
        gd_pre = [d for d in o1 if d["gold_label"] == "ground_truth"]
        frac = lambda docs, cls, denom=None: sum(d["gold_label"] == cls for d in docs) / (denom or len(docs) or 1)
        q = {
            "pubid": pubid,
            "hd_density_pre": frac(o1, "harmful"),
            "hd_density_post": frac([d for d in o1 if d["id"] in post], "harmful"),
            "gt_retention_loss": (sum(d["is_blocked"] and d["id"] not in reinstated for d in gd_pre) / len(gd_pre)) if gd_pre else 0.0,
        }
        for cond in CONDITIONS:
            top5 = row[cond]["top5"]
            j = judgments.get(by_pair.get((pubid, cond)), {})
            q[f"hd_at5_{cond}"] = frac(top5, "harmful", 5)
            q[f"hit_at5_{cond}"] = int(any(d["gold_label"] == "ground_truth" for d in top5))
            q[f"hallucinated_{cond}"] = int(bool(j.get("hallucinated")))
            q[f"hd_induced_{cond}"] = int(bool(j.get("hd_induced")))
            q[f"ucr_{cond}"] = float(j.get("unsupported_claim_rate", 0.0))
            q[f"correct_{cond}"] = int(j.get("decision") == records[pubid]["final_decision"])
            q[f"maybe_{cond}"] = int(j.get("decision") == "maybe")
            q[f"judged_{cond}"] = bool(j)
        per_query.append(q)
        for d in o1:
            cand_true.append("harmful" if d["gold_label"] == "harmful" else "not_harmful")
            cand_pred.append("harmful" if d["is_blocked"] else "not_harmful")
            cand_prob.append(d["p_hd"])
            cand_query.append(qi)

    col = lambda k, rows: [r[k] for r in rows]
    judged = [r for r in per_query if r["judged_control"] and r["judged_treatment"]]
    boot_rng = np.random.default_rng(args.seed)
    n_perm = getattr(args, "n_permutations", 10000)

    def boot_ci(diff) -> Optional[List[float]]:
        """95% percentile bootstrap CI of a mean paired difference, resampling queries (Efron & Tibshirani, 1993)."""
        diff = np.asarray(diff, dtype=float)
        if len(diff) == 0:
            return None
        means = diff[boot_rng.integers(0, len(diff), (args.n_bootstrap, len(diff)))].mean(axis=1)
        return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]

    def binary(key: str, rows: List[Dict], lower_is_better: bool) -> Dict[str, Any]:
        """Paired binary outcome: rates, delta p (positive = improvement under treatment) with Newcombe's 95% CI, McNemar mid-p."""
        c, t = col(f"{key}_control", rows), col(f"{key}_treatment", rows)
        first, second = (c, t) if lower_is_better else (t, c)
        diff = np.subtract(first, second)
        return {**mcnemar_midp(c, t), "delta_p_improvement": round(float(np.mean(diff)), 4) if len(diff) else None,
                "delta_p_ci_95": newcombe_paired_ci(first, second)}

    def continuous(key: str, rows: List[Dict]) -> Dict[str, Any]:
        """Paired continuous outcome: means, SDs, d = control - treatment with bootstrap CI, paired permutation test."""
        c, t = col(f"{key}_control", rows), col(f"{key}_treatment", rows)
        return {**paired_permutation_test(c, t, n_permutations=n_perm, seed=args.seed),
                "mean_diff_ci_95": boot_ci(np.subtract(c, t))}

    improved = lambda r, k: bool(r["significant"] and (r[k] or 0) > 0)

    def test_hypotheses(rows: List[Dict], rows_judged: List[Dict]) -> Dict[str, Any]:
        """H1-H4 on a set of queries (rows) and those of them judged in both conditions (rows_judged)."""
        h1 = binary("hallucinated", rows_judged, lower_is_better=True)
        h2_hhd = binary("hd_induced", rows_judged, lower_is_better=True)
        h2_acc = binary("correct", rows_judged, lower_is_better=False)
        h3 = continuous("hd_at5", rows)
        h4 = binary("hit_at5", rows, lower_is_better=False)
        return {
            "H1_hallucination_rate": {**h1, "role": "confirmatory", "predicted": "treatment < control",
                                      "supported": improved(h1, "delta_p_improvement")},
            "H2_hd_induced_hallucination_rate": {**h2_hhd, "predicted": "treatment < control"},
            "H2_answer_accuracy": {**h2_acc, "predicted": "treatment > control"},
            "H2_supported": improved(h2_hhd, "delta_p_improvement") and improved(h2_acc, "delta_p_improvement"),
            "H3_hd_at_top5": {**h3, "predicted": "treatment < control", "supported": improved(h3, "mean_diff")},
            "H4_gd_at_top5_hit5": {**h4, "predicted": "no significant loss of GD at Top-5 (treatment >= control)",
                                   "supported": not (h4["significant"] and (h4["delta_p_improvement"] or 0) < 0),
                                   "direction": ("no difference" if not h4["delta_p_improvement"]
                                                 else "treatment higher" if h4["delta_p_improvement"] > 0
                                                 else "treatment lower")},
            "Secondary_unsupported_claim_rate": continuous("ucr", rows_judged),
        }

    hypotheses = test_hypotheses(per_query, judged)
    h2_acc = hypotheses["H2_answer_accuracy"]
    # the official PQA-L test set (500 questions) is reported separately when the run includes it
    is_official = lambda r: bool(records[r["pubid"]].get("official_test"))
    official = [r for r in per_query if is_official(r)]
    hypotheses_official = (test_hypotheses(official, [r for r in judged if is_official(r)])
                           if official and len(official) < len(per_query) else None)

    accuracy_by_label = {}
    for lab in ("yes", "no"):
        rows = [r for r in judged if records[r["pubid"]]["final_decision"] == lab]
        accuracy_by_label[lab] = binary("correct", rows, lower_is_better=False)
    accuracy_by_label["overall"] = h2_acc
    accuracy_by_label["majority_class_rate_yes"] = round(float(np.mean(
        [records[r["pubid"]]["final_decision"] == "yes" for r in judged])), 4) if judged else None

    # "maybe" counts as incorrect in H2; the share of "maybe" answers and the accuracy among yes/no answers
    # are also reported separately
    maybe_answers = {}
    for cond in CONDITIONS:
        decided = [r for r in judged if not r[f"maybe_{cond}"]]
        maybe_answers[cond] = {
            "maybe_rate": round(float(np.mean(col(f"maybe_{cond}", judged))), 4) if judged else None,
            "n_yes_no_answers": len(decided),
            "accuracy_among_yes_no_answers": round(float(np.mean(col(f"correct_{cond}", decided))), 4) if decided else None,
        }

    supplementary_stage2 = {k: round(float(np.mean(col(k, per_query))), 4) if per_query else None
                            for k in ("hd_density_pre", "hd_density_post", "gt_retention_loss")}

    sop1 = filter_metrics(cand_true, cand_pred, cand_prob)
    t = np.array([c == "harmful" for c in cand_true])
    p = np.array([c == "harmful" for c in cand_pred])
    by_query = np.array(cand_query)
    rng = np.random.default_rng(args.seed)
    boot = {"TPR_Recall": [], "FPR": [], "Precision": [], "F1_score": []}
    for _ in range(args.n_bootstrap):
        counts = np.bincount(rng.integers(0, len(per_query), len(per_query)), minlength=len(per_query))
        w = counts[by_query]
        tp, fp = float((w * (t & p)).sum()), float((w * (~t & p)).sum())
        fn, tn = float((w * (t & ~p)).sum()), float((w * (~t & ~p)).sum())
        rec_ = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        boot["TPR_Recall"].append(rec_)
        boot["FPR"].append(fp / (fp + tn) if fp + tn else 0.0)
        boot["Precision"].append(prec)
        boot["F1_score"].append(2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0)
    sop1["ci_95_query_bootstrap"] = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
                                     for k, v in boot.items()}
    sop1["confusion_matrix"] = {"TP": int((t & p).sum()), "FP": int((~t & p).sum()),
                                "FN": int((t & ~p).sum()), "TN": int((~t & ~p).sum())}

    return {
        "live_run": True,
        "num_queries": len(per_query),
        "alpha": 0.05,
        "multiplicity_correction": "none (H1 is the only confirmatory hypothesis)",
        "n_judged_both_conditions": len(judged),
        "SOP1_filter_detection": sop1,
        "Hypotheses": hypotheses,
        "num_official_test_queries": len(official),
        "Hypotheses_official_test_set": hypotheses_official,
        "Answer_accuracy_by_reference_label": accuracy_by_label,
        "Maybe_answers": maybe_answers,
        "Supplementary_stage2_means": supplementary_stage2,
        "per_query": per_query,
        **extra,
    }


def run_benchmark(args) -> Dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)
    queries, source = load_test_queries(args.split, args.n)
    records = {r["pubid"]: r for r in queries}
    pair_labels = load_pair_labels(args.labels)
    pipe = SafeMedPipeline(filter_model_path=args.filter_model, corpus_path=args.corpus)
    judge = ClaimJudge()
    preflight(pipe, judge, pair_labels, args.allow_same_judge, args.pilot)

    def label(pubid: str, doc: Dict[str, Any]) -> str:
        return pair_labels.get((pubid, str(doc.get("id")))) or doc.get("true_label") or "mediocre"

    print(f"\nSafeMedAI benchmark: {source}")
    print(f"Corpus: {args.corpus} | tau_safe = {pipe.filter.harmful_threshold} | "
          f"generator = {pipe.generator.model} | judge = {judge.model}\n")

    phases = {"generate", "judge", "reliability", "analyze"} if args.phase == "all" else {args.phase}
    traces_list = generate(pipe, queries, label, args.output_dir, args.log_interval) if "generate" in phases \
        else read_jsonl(os.path.join(args.output_dir, TRACES))
    traces = {r["pubid"]: r for r in traces_list if r["pubid"] in records}
    if not traces:
        raise BenchmarkAbort("No generated answers yet; run with --phase generate first.")
    blinding = blind(list(traces.values()), args.output_dir, args.seed)

    if "judge" in phases:
        judged_ids = set(list(records)[: args.judge_n]) if args.judge_n else None
        judgments = judge_all(judge, blinding, traces, records, args.output_dir, args.judge_pace, judged_ids)
    else:
        judgments = {r["code"]: r for r in read_jsonl(os.path.join(args.output_dir, JUDGMENTS))}

    extra = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query_source": source,
        "corpus": args.corpus,
        "pair_labels": args.labels,
        "filter_model": pipe.filter.model_path,
        "tau_safe": pipe.filter.harmful_threshold,
        "generator_model": pipe.generator.model,
        "judge_model": judge.model,
        "judge_same_as_generator": judge.model == pipe.generator.model,
        "leave_self_out": True,
        "blinded_random_order": True,
        "claim_reference": "PubMedQA context + long answer + reference answer",
        "files": {k: os.path.relpath(os.path.join(args.output_dir, v), BACKEND)
                  for k, v in (("traces", TRACES), ("blinding_key", BLINDING), ("judgments", JUDGMENTS))},
    }
    if "reliability" in phases:
        extra["Reliability"] = reliability(args, pipe, judge, blinding, judgments, traces, records, args.output_dir)

    if "analyze" not in phases:
        return extra
    missing = sum(1 for c in blinding["order"] if c not in judgments)
    if missing:
        print(f"Warning: {missing} answers are not judged yet; their judged_* fields are False.")
    summary = analyze(traces, blinding, judgments, records, args, extra)
    out_file = os.path.join(args.output_dir, RESULTS)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    h = summary["Hypotheses"]
    print("\n================ H1-H4 (two-sided, alpha = 0.05, no multiplicity correction) ================")
    print(f"Judged in both conditions: {summary['n_judged_both_conditions']} of {summary['num_queries']} queries")
    for name in ("H1_hallucination_rate", "H2_hd_induced_hallucination_rate", "H2_answer_accuracy", "H4_gd_at_top5_hit5"):
        r = h[name]
        print(f"{name:34s} N={r['n']:4d}  control={r['rate_control']}  treatment={r['rate_treatment']}  "
              f"dp={r['delta_p_improvement']} {r['delta_p_ci_95']}  b={r['b_control_only']} c={r['c_treatment_only']}  "
              f"p={r['p_value']:.6g}")
    for name in ("H3_hd_at_top5", "Secondary_unsupported_claim_rate"):
        r = h[name]
        print(f"{name:34s} N={r['n']:4d}  control={r['mean_control']}  treatment={r['mean_treatment']}  "
              f"d={r['mean_diff']} {r['mean_diff_ci_95']}  p={r['p_value']:.6g}")
    print(f"Supported: H1 {h['H1_hallucination_rate']['supported']} | H2 {h['H2_supported']} | "
          f"H3 {h['H3_hd_at_top5']['supported']} | H4 {h['H4_gd_at_top5_hit5']['supported']} "
          f"({h['H4_gd_at_top5_hit5']['direction']})")
    ho = summary.get("Hypotheses_official_test_set")
    if ho:
        print(f"Official PQA-L test set ({summary['num_official_test_queries']} queries): "
              f"H1 {ho['H1_hallucination_rate']['supported']} (p={ho['H1_hallucination_rate']['p_value']:.6g}) | "
              f"H2 {ho['H2_supported']} | H3 {ho['H3_hd_at_top5']['supported']} | H4 {ho['H4_gd_at_top5_hit5']['supported']}")
    for lab in ("yes", "no"):
        r = summary["Answer_accuracy_by_reference_label"][lab]
        print(f"Accuracy, reference '{lab}': N={r['n']}  control={r['rate_control']}  treatment={r['rate_treatment']}  "
              f"p={r['p_value']:.6g}")
    s = summary["SOP1_filter_detection"]
    print(f"SOP1: TPR {s['TPR_Recall']:.3f}  FPR {s['FPR']:.3f}  Precision {s['Precision']:.3f}  F1 {s['F1_score']:.3f}")
    print(f"\nSaved: {os.path.relpath(out_file, BACKEND)}\n")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test", choices=["test", "val"], help="query split (default test)")
    p.add_argument("--n", type=int, default=None, help="only the first N queries of the split (default all)")
    p.add_argument("--corpus", default=PQAA_CORPUS, help="retrieval corpus JSON (default the pooled pqa_artificial corpus)")
    p.add_argument("--labels", help="per-pair gold labels JSONL (pubid, doc_id, label) from the annotation step")
    p.add_argument("--filter-model", help="fine-tuned Stage 2 folder (default SAFEMED_FILTER_MODEL or models/safemed_filter)")
    p.add_argument("--output-dir", default=os.path.join(BACKEND, "results"))
    p.add_argument("--phase", default="all", choices=["all", "generate", "judge", "reliability", "analyze"])
    p.add_argument("--allow-same-judge", action="store_true", help="allow judge = generator (recorded as a deviation)")
    p.add_argument("--judge-pace", type=float, default=0.0, help="seconds between judged answers (rate limits)")
    p.add_argument("--rescore-frac", type=float, default=0.1, help="share of answers re-scored by the judge for the reliability check")
    p.add_argument("--regen-frac", type=float, default=0.1, help="share of queries whose answer is regenerated for the reliability check")
    p.add_argument("--judge-n", type=int, default=0,
                   help="judge only the answers of the first N queries (pilots only; the experiment judges all, so leave it out)")
    p.add_argument("--pilot", action="store_true", help="dry run: allow a sample without HD labels (not for reported results)")
    p.add_argument("--n-bootstrap", type=int, default=10000, help="bootstrap resamples for CIs (B = 10,000)")
    p.add_argument("--n-permutations", type=int, default=10000, help="sign flips for the H3 permutation test")
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