"""Unit tests for the annotation labeller and the reliability check."""

import json

import pytest

from tests.conftest import fake_llm_client
from src.annotation.llm_labeler import LabelerUnavailableError, label_pairs
from src.annotation.reliability_check import cohens_kappa, confusion_matrix_report


def test_cohens_kappa_perfect_agreement():
    labels = ["ground_truth", "harmful", "mediocre", "harmful"]
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)


def test_confusion_matrix_report():
    human = ["ground_truth", "harmful", "mediocre", "harmful"]
    llm = ["ground_truth", "harmful", "mediocre", "mediocre"]
    report = confusion_matrix_report(llm, human)
    assert {"cohens_kappa", "confusion_matrix", "class_breakdown"} <= set(report)


def _answers(evidence="", same=(False, False, False), shared=(False, False, False), rationale="r"):
    exposure, outcome, direction = same
    condition, exp, out = shared
    return json.dumps({"q_condition": "c", "q_exposure": "e", "q_outcome": "o",
                       "d_condition": "c", "d_exposure": "e", "d_outcome": "o", "evidence": evidence,
                       "evidence_same_exposure": exposure, "evidence_same_outcome": outcome, "same_direction": direction,
                       "shares_condition": condition, "shares_exposure": exp, "shares_outcome": out,
                       "rationale": rationale})


def test_labeller_sends_rubric_and_reference_one_pair_per_request():
    calls = []
    pairs = [{"query": "Does aspirin help?", "reference_answer": "yes", "text": "Aspirin never helps."},
             {"query": "Does aspirin help?", "reference_answer": "yes", "text": "Statins and lipids."}]
    out = label_pairs(pairs, client=fake_llm_client(calls, reply=_answers()), batch_size=10)
    assert len(calls) == 2  # one request per pair, whatever batch_size says
    assert "codebook" in calls[0]["messages"][0]["content"]
    assert "Correct answer (reference): yes" in calls[0]["messages"][1]["content"]
    assert "Document: Aspirin never helps." in calls[0]["messages"][1]["content"]
    assert [o["llm_label"] for o in out] == ["mediocre", "mediocre"]


def test_gd_needs_a_real_quote_on_the_same_exposure_outcome_and_direction():
    text = "Background. Aspirin helped patients in a large trial. More."
    pair = [{"query": "Does aspirin help?", "reference_answer": "yes", "text": text}]
    gd = label_pairs(pair, client=fake_llm_client(reply=_answers("Aspirin helped patients in a large trial.",
                                                                 same=(True, True, True))))
    assert gd[0]["llm_label"] == "ground_truth" and gd[0]["evidence_found"] is True
    other_exposure = label_pairs(pair, client=fake_llm_client(reply=_answers("Aspirin helped patients in a large trial.",
                                                                             same=(False, True, True))))
    assert other_exposure[0]["llm_label"] == "harmful"  # a real quote on another exposure is not the reference finding
    opposite = label_pairs(pair, client=fake_llm_client(reply=_answers("Aspirin helped patients in a large trial.",
                                                                       same=(True, True, False))))
    assert opposite[0]["llm_label"] == "harmful"


def test_a_gd_quote_that_is_not_in_the_document_is_not_gd():
    pairs = [{"query": "Is hs-CRP irrelevant?", "reference_answer": "yes",
              "text": "Uterine biopsies and progesterone in healthy women."}]
    invented = "hs-CRP is irrelevant for diagnosing endometriosis."
    shared = label_pairs(pairs, client=fake_llm_client(reply=_answers(invented, (True, True, True), (True, False, False))))
    assert shared[0]["llm_label"] == "harmful" and shared[0]["evidence_found"] is False
    unrelated = label_pairs(pairs, client=fake_llm_client(reply=_answers(invented, (True, True, True))))
    assert unrelated[0]["llm_label"] == "mediocre"


def test_any_shared_condition_exposure_or_outcome_is_hd():
    pair = [{"query": "q", "reference_answer": "yes", "text": "d"}]
    for shared in ((True, False, False), (False, True, False), (False, False, True)):
        out = label_pairs(pair, client=fake_llm_client(reply=_answers(shared=shared)))
        assert out[0]["llm_label"] == "harmful"
        assert out[0]["label_checks"]["shares_condition"] is shared[0]


def test_labeller_asks_again_after_an_unusable_reply():
    calls = []
    replies = ["not json", json.dumps({"label": "harmful"}), _answers(shared=(True, False, False))]
    client = fake_llm_client(calls)
    create = client.chat.completions.create
    client.chat.completions.create = lambda **kw: (create(**kw), _reply(replies[len(calls) - 1]))[1]
    out = label_pairs([{"query": "q", "text": "d"}], client=client, max_retries=2)
    assert len(calls) == 3 and out[0]["llm_label"] == "harmful"


def test_labeller_gives_up_after_retries():
    with pytest.raises(LabelerUnavailableError):
        label_pairs([{"query": "q", "text": "d"}], client=fake_llm_client(reply=json.dumps({"label": "harmful"})))


def _reply(content):
    import types
    msg = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])