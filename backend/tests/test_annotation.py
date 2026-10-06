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


def test_labeller_batches_pairs_and_sends_rubric_and_reference():
    calls = []
    reply = json.dumps({"labels": [{"pair": 1, "label": "harmful", "rationale": "contradicts"},
                                   {"pair": 2, "label": "ground_truth", "rationale": "supports"}]})
    pairs = [{"query": "Does aspirin help?", "reference_answer": "yes", "text": "Aspirin never helps."},
             {"query": "Does aspirin help?", "reference_answer": "yes", "text": "Aspirin helped in a trial."}]
    out = label_pairs(pairs, client=fake_llm_client(calls, reply=reply), batch_size=10)
    assert [o["llm_label"] for o in out] == ["harmful", "ground_truth"]
    assert len(calls) == 1
    assert "codebook" in calls[0]["messages"][0]["content"]
    assert "Correct answer (reference): yes" in calls[0]["messages"][1]["content"]


def test_labeller_rejects_an_invalid_label():
    reply = json.dumps({"labels": [{"pair": 1, "label": "probably fine"}]})
    with pytest.raises(LabelerUnavailableError):
        label_pairs([{"query": "q", "text": "d"}], client=fake_llm_client(reply=reply))


def test_labeller_asks_again_for_a_skipped_pair():
    calls = []
    replies = [json.dumps({"labels": [{"pair": 2, "label": "harmful", "rationale": "contradicts"}]}),
               json.dumps({"labels": [{"pair": 1, "label": "mediocre", "rationale": "unrelated"}]})]
    client = fake_llm_client(calls)
    create = client.chat.completions.create
    client.chat.completions.create = lambda **kw: (create(**kw), _reply(replies[len(calls) - 1]))[1]
    out = label_pairs([{"query": "q", "text": "d"}, {"query": "q", "text": "e"}], client=client)
    assert [o["llm_label"] for o in out] == ["mediocre", "harmful"]
    assert len(calls) == 2
    assert "Document: d" in calls[1]["messages"][1]["content"] and "Document: e" not in calls[1]["messages"][1]["content"]


def test_labeller_gives_up_after_retries():
    reply = json.dumps({"labels": [{"pair": 1, "label": "probably fine"}]})
    with pytest.raises(LabelerUnavailableError):
        label_pairs([{"query": "q", "text": "d"}, {"query": "q", "text": "e"}], client=fake_llm_client(reply=reply))


def _reply(content):
    import types
    msg = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
