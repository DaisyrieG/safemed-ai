"""Tests for the Stage 5 benchmark phases (blinding, judging, analysis) on hand-made traces."""

import json
import types

from src.stage5_evaluation import run_benchmark as rb
from src.stage5_evaluation.judge import ClaimJudge


def _doc(i, gold):
    return {"id": f"d{i}", "title": f"T{i}", "text": f"text {i}", "gold_label": gold}


def _trace(pubid, control_hd=True):
    o1 = [{"id": "d0", "retrieval_score": 0.9, "p_hd": 0.1, "is_blocked": False, "gold_label": "ground_truth"},
          {"id": "d1", "retrieval_score": 0.8, "p_hd": 0.9, "is_blocked": True, "gold_label": "harmful"},
          {"id": "d2", "retrieval_score": 0.7, "p_hd": 0.2, "is_blocked": False, "gold_label": "mediocre"},
          {"id": "d3", "retrieval_score": 0.6, "p_hd": 0.3, "is_blocked": False, "gold_label": "mediocre"}]
    treatment = [_doc(0, "ground_truth"), _doc(2, "mediocre"), _doc(3, "mediocre")]
    control = [_doc(0, "ground_truth"), _doc(1, "harmful" if control_hd else "mediocre"), _doc(2, "mediocre")]
    return {"pubid": pubid, "query": f"question {pubid}?", "O1_pre_filter_pool": o1,
            "O2_post_filter_pool": ["d0", "d2", "d3"], "O2_reinstated": [],
            "treatment": {"top5": treatment, "answer": "Decision: yes\nA [1]."},
            "control": {"top5": control, "answer": "Decision: no\nB [2]."}}


def _records(pubids):
    return {p: {"pubid": p, "question": f"question {p}?", "context": "ctx", "long_answer": "la", "final_decision": "yes"}
            for p in pubids}


def fake_judge():
    """Every answer gets two claims; claim 2 is CONTRADICTED only for answers whose text contains 'B'."""
    seen = []

    def create(**kw):
        prompt = kw["messages"][0]["content"]
        seen.append(prompt)
        if "atomic factual claims" in prompt:
            content = "Claim one.\nClaim two B." if "B [2]" in prompt else "Claim one.\nClaim two."
        elif '"verdicts"' in prompt:
            bad = "Claim two B." in prompt
            content = json.dumps({"verdicts": [{"index": 1, "status": "SUPPORTED", "reasoning": ""},
                                               {"index": 2, "status": "CONTRADICTED" if bad else "SUPPORTED", "reasoning": ""}]})
        else:
            content = json.dumps({"attributions": [{"index": 1, "supported_by_document": 1}]})
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    return ClaimJudge(client=client, model="fake-judge"), seen


def test_blinding_gives_unique_codes_without_the_condition(tmp_path):
    traces = [_trace("1"), _trace("2")]
    blinding = rb.blind(traces, str(tmp_path), seed=42)
    assert len(blinding["order"]) == 4 == len(set(blinding["order"]))
    assert {v["condition"] for v in blinding["key"].values()} == {"control", "treatment"}
    again = rb.blind(traces + [_trace("3")], str(tmp_path), seed=42)
    assert again["order"][:4] == blinding["order"] and len(again["order"]) == 6


def test_reference_evidence_is_the_pubmedqa_record_only():
    ref = rb.reference_evidence(_records(["1"])["1"])
    assert "Reference answer: yes" in ref and "Retrieved document" not in ref


def test_judge_never_sees_the_condition_and_results_resume(tmp_path):
    traces = {p: _trace(p) for p in ("1", "2")}
    records = _records(traces)
    blinding = rb.blind(list(traces.values()), str(tmp_path), seed=1)
    judge, seen = fake_judge()
    judged = rb.judge_all(judge, blinding, traces, records, str(tmp_path), pace=0)
    assert len(judged) == 4
    assert not any("control" in p.lower() or "treatment" in p.lower() for p in seen)
    calls = len(seen)
    rb.judge_all(judge, blinding, traces, records, str(tmp_path), pace=0)
    assert len(seen) == calls


def test_analysis_tests_the_manuscript_hypotheses(tmp_path):
    traces = {p: _trace(p) for p in ("1", "2", "3")}
    records = _records(traces)
    blinding = rb.blind(list(traces.values()), str(tmp_path), seed=1)
    judge, _ = fake_judge()
    judged = rb.judge_all(judge, blinding, traces, records, str(tmp_path), pace=0)
    args = types.SimpleNamespace(seed=42, n_bootstrap=50, n_permutations=200)
    summary = rb.analyze(traces, blinding, judged, records, args, {})

    assert summary["multiplicity_correction"].startswith("none")
    h = summary["Hypotheses"]
    h1 = h["H1_hallucination_rate"]
    assert h1["rate_control"] == 1.0 and h1["rate_treatment"] == 0.0
    assert h1["b_control_only"] == 3 and h1["c_treatment_only"] == 0 and h1["delta_p_improvement"] == 1.0
    assert h1["p_value"] == 0.125  # mid-p for b = 3, c = 0: 2 * 0.125 - 0.125
    assert h["H2_hd_induced_hallucination_rate"]["rate_control"] == 1.0
    assert h["H2_answer_accuracy"]["rate_treatment"] == 1.0 and h["H2_answer_accuracy"]["rate_control"] == 0.0
    assert h["H3_hd_at_top5"]["mean_control"] == 0.2 and h["H3_hd_at_top5"]["mean_diff"] == 0.2
    assert h["H4_gd_at_top5_hit5"]["rate_control"] == 1.0 == h["H4_gd_at_top5_hit5"]["rate_treatment"]
    assert h["H4_gd_at_top5_hit5"]["p_value"] == 1.0 and h["H4_gd_at_top5_hit5"]["supported"]  # no significant loss
    assert h["H4_gd_at_top5_hit5"]["direction"] == "no difference"
    assert summary["Answer_accuracy_by_reference_label"]["yes"]["n"] == 3
    assert summary["Supplementary_stage2_means"]["hd_density_pre"] == 0.25
    assert summary["Maybe_answers"]["treatment"] == {"maybe_rate": 0.0, "n_yes_no_answers": 3,
                                                     "accuracy_among_yes_no_answers": 1.0}
    q = summary["per_query"][0]
    assert q["hd_density_pre"] == 0.25 and q["hd_density_post"] == 0.0 and q["gt_retention_loss"] == 0.0
    assert summary["SOP1_filter_detection"]["confusion_matrix"]["TP"] == 3


def test_judge_subset_limits_h1_h2_to_queries_judged_in_both_conditions(tmp_path):
    traces = {p: _trace(p) for p in ("1", "2", "3", "4")}
    records = _records(traces)
    blinding = rb.blind(list(traces.values()), str(tmp_path), seed=1)
    judge, _ = fake_judge()
    judged = rb.judge_all(judge, blinding, traces, records, str(tmp_path), pace=0, judged_ids={"1", "2"})
    assert len(judged) == 4 and {blinding["key"][c]["pubid"] for c in judged} == {"1", "2"}
    summary = rb.analyze(traces, blinding, judged, records, types.SimpleNamespace(seed=42, n_bootstrap=20), {})
    assert summary["Hypotheses"]["H1_hallucination_rate"]["n"] == 2
    assert summary["Hypotheses"]["H3_hd_at_top5"]["n"] == 4


def test_mcnemar_midp_is_one_when_b_equals_c():
    from src.stage5_evaluation.statistical_tests import mcnemar_midp
    control = [1, 1, 0, 0, 0]
    treatment = [0, 0, 1, 1, 0]  # b = 2, c = 2
    assert mcnemar_midp(control, treatment)["p_value"] == 1.0


def test_mcnemar_midp_by_hand():
    from src.stage5_evaluation.statistical_tests import mcnemar_midp
    # b = 1, c = 7: exact p = 2 * P(X <= 1) = 0.0703; mid-p subtracts P(X = 1) = 0.0313, giving 0.0391
    control = [1] * 1 + [0] * 7 + [0] * 10
    treatment = [0] * 1 + [1] * 7 + [0] * 10
    assert round(mcnemar_midp(control, treatment)["p_value"], 4) == 0.0391


def test_newcombe_ci_matches_fagerland_2014_example():
    from src.stage5_evaluation.statistical_tests import newcombe_paired_ci
    # Fagerland, Lydersen & Laake (2014), Table II (n11=1, n12=1, n21=7, n22=12): difference 0.286, CI 0.026 to 0.507
    before = [1] * 1 + [1] * 1 + [0] * 7 + [0] * 12
    after = [1] * 1 + [0] * 1 + [1] * 7 + [0] * 12
    lo, hi = newcombe_paired_ci(after, before)
    assert round(lo, 3) == 0.026 and round(hi, 3) == 0.507


def test_test_split_is_all_pqal_with_the_official_500_marked():
    from collections import Counter
    records, _ = rb.load_test_queries("test", None)
    assert len(records) == 1000
    assert Counter(r["final_decision"] for r in records) == {"yes": 552, "no": 338, "maybe": 110}
    official = [r for r in records if r["official_test"]]
    assert len(official) == 500
    assert Counter(r["final_decision"] for r in official) == {"yes": 276, "no": 169, "maybe": 55}


def test_analysis_reports_the_official_test_set_separately(tmp_path):
    traces = {p: _trace(p) for p in ("1", "2", "3")}
    records = _records(traces)
    records["1"]["official_test"] = records["2"]["official_test"] = True
    blinding = rb.blind(list(traces.values()), str(tmp_path), seed=1)
    judge, _ = fake_judge()
    judged = rb.judge_all(judge, blinding, traces, records, str(tmp_path), pace=0)
    args = types.SimpleNamespace(seed=42, n_bootstrap=50, n_permutations=200)
    summary = rb.analyze(traces, blinding, judged, records, args, {})
    assert summary["num_queries"] == 3 and summary["num_official_test_queries"] == 2
    official = summary["Hypotheses_official_test_set"]
    assert official["H1_hallucination_rate"]["n"] == 2 and official["H3_hd_at_top5"]["n"] == 2
    assert summary["Hypotheses"]["H1_hallucination_rate"]["n"] == 3
