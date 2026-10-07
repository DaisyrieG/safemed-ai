"""Tests for the SafeMed AI REST API, over the test pipeline (fixture corpus, oracle filter, fake LLM)."""

import json
import types

import pytest
from fastapi.testclient import TestClient

import src.api.app as api
from src.stage5_evaluation.judge import ClaimJudge


def fake_judge_client():
    """Claims: two lines; verdicts: claim 1 SUPPORTED, claim 2 CONTRADICTED; attribution: none."""
    def create(**kw):
        prompt = kw["messages"][0]["content"]
        if "atomic factual claims" in prompt:
            content = "Aspirin reduced colorectal cancer risk.\nThe effect was seen in all age groups."
        elif '"verdicts"' in prompt:
            content = json.dumps({"verdicts": [
                {"index": 1, "status": "SUPPORTED", "reasoning": "stated"},
                {"index": 2, "status": "CONTRADICTED", "reasoning": "only adults"}]})
        else:
            content = json.dumps({"attributions": [{"index": 1, "supported_by_document": 0}]})
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


@pytest.fixture
def client(make_pipeline, monkeypatch):
    monkeypatch.setattr(api, "_pipeline", make_pipeline())
    monkeypatch.setattr(api, "_judge", ClaimJudge(client=fake_judge_client(), model="fake-judge"))
    return TestClient(api.app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "SafeMed AI" in response.json()["service"]


def test_health_endpoint(client):
    data = client.get("/api/health").json()
    assert data["status"] == "healthy"
    assert "stage_1_retriever" in data
    assert data["stage_2_filter"]["cutoff_k"] == 30


def test_sample_cases_endpoint(client):
    cases = client.get("/api/sample-cases").json()
    assert len(cases) == 6
    assert all(c["query"].endswith("?") and c["case_id"].startswith("pqa-") for c in cases)


def test_query_reports_hallucination_check(client):
    data = client.post("/api/query", json={"query": "Does daily aspirin reduce colorectal cancer risk?"}).json()
    check = data["hallucination_check"]
    assert data["hallucination_error"] is None
    assert data["decision"] == "yes"
    assert check["judge_model"] == "fake-judge"
    for condition in ("proposed", "control"):
        r = check[condition]
        assert r["n_claims"] == 2 and r["n_contradicted"] == 1
        assert r["hallucinated"] is True
        assert r["unsupported_claim_rate"] == 0.0


def test_query_can_skip_hallucination_check(client):
    data = client.post("/api/query", json={"query": "Does daily aspirin reduce colorectal cancer risk?",
                                            "check_hallucination": False}).json()
    assert data["hallucination_check"] is None


def test_clinical_query_execution(client):
    payload = {"query": "Are COVID-19 mRNA vaccines associated with increased risk of myocarditis in young adults?"}
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["safety_scan"]["total_scanned"] == 50
    assert 0 < data["safety_scan"]["blocked_count"] < 50
    blocked = [d for d in data["safety_scan"]["evaluation_log"] if d["is_blocked"]]
    assert all(d["status"] == "BLOCKED: UNSAFE" and d["block_reason"] for d in blocked)

    assert len(data["trusted_sources"]) == 5
    assert all(s["verification_status"] == "verified safe" for s in data["trusted_sources"])
    assert data["clinical_summary"].startswith("Decision:")
    assert data["atomic_claims"] == []


def test_hallucination_check_endpoint(client):
    first = client.post("/api/query", json={"query": "Does daily aspirin reduce colorectal cancer risk?",
                                             "check_hallucination": False}).json()
    payload = {
        "query": first["query"],
        "proposed": {"answer": first["clinical_summary"], "doc_ids": [s["id"] for s in first["trusted_sources"]]},
        "control": {"answer": first["control_answer"] or "",
                    "doc_ids": [d["id"] for d in first["evaluation_breakdown"]["control"]["top5_documents"]]},
    }
    check = client.post("/api/hallucination-check", json=payload).json()
    assert check["reference"] == "sources"
    assert check["proposed"]["hallucinated"] is True
    assert check["control"]["n_claims"] == 2


def test_non_biomedical_question_is_refused(client, monkeypatch):
    monkeypatch.setattr(api, "is_biomedical", lambda pipeline, query: False)
    res = client.post("/api/query", json={"query": "is banana yellow?"})
    assert res.status_code == 422 and res.json()["detail"] == api.NOT_BIOMEDICAL
    preset = client.post("/api/query", json={"query": "is banana yellow?", "case_id": "pqa-24318956", "check_hallucination": False})
    assert preset.status_code == 200
