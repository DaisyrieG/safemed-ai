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


def test_reference_evidence_adds_retrieved_documents_from_the_context():
    ref = rb.reference_evidence(_records(["1"])["1"], [{"text": "gold text"}])
    assert "Reference answer: yes" in ref and "Retrieved document 1: gold text" in ref


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


def test_analysis_reports_table4_tests_without_holm(tmp_path):
    traces = {p: _trace(p) for p in ("1", "2", "3")}
    records = _records(traces)
    blinding = rb.blind(list(traces.values()), str(tmp_path), seed=1)
    judge, _ = fake_judge()
    judged = rb.judge_all(judge, blinding, traces, records, str(tmp_path), pace=0)
    args = types.SimpleNamespace(seed=42, n_bootstrap=50)
    summary = rb.analyze(traces, blinding, judged, records, args, {})

    assert summary["multiplicity_correction"].startswith("none")
    for h in summary["Hypotheses"].values():
        assert "p_value_holm" not in h and "hodges_lehmann" not in h and h["n"] == 3
    q = summary["per_query"][0]
    assert q["hd_density_pre"] == 0.25 and q["hd_density_post"] == 0.0
    assert q["gt_retention_loss"] == 0.0
    assert q["hd_at5_control"] == 0.2 and q["hd_at5_treatment"] == 0.0
    d = summary["Descriptive"]
    assert d["control"]["hallucination_rate"] == 1.0 and d["treatment"]["hallucination_rate"] == 0.0
    assert d["control"]["hd_induced_hallucination_rate"] == 1.0
    assert d["treatment"]["answer_accuracy"] == 1.0 and d["control"]["answer_accuracy"] == 0.0
    assert summary["SOP1_filter_detection"]["confusion_matrix"]["TP"] == 3


def test_judge_subset_limits_h4_to_judged_queries(tmp_path):
    traces = {p: _trace(p) for p in ("1", "2", "3", "4")}
    records = _records(traces)
    blinding = rb.blind(list(traces.values()), str(tmp_path), seed=1)
    judge, _ = fake_judge()
    judged = rb.judge_all(judge, blinding, traces, records, str(tmp_path), pace=0, judged_ids={"1", "2"})
    assert len(judged) == 4 and {blinding["key"][c]["pubid"] for c in judged} == {"1", "2"}
    summary = rb.analyze(traces, blinding, judged, records, types.SimpleNamespace(seed=42, n_bootstrap=20), {})
    assert summary["Hypotheses"]["H4_unsupported_claim_rate"]["n"] == 2
    assert summary["Hypotheses"]["H1_harmful_density_reduction"]["n"] == 4
