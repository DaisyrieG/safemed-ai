"""Tests for the SafeMed AI REST API, over the test pipeline (fixture corpus, oracle filter, fake LLM)."""

import pytest
from fastapi.testclient import TestClient

import src.api.app as api


@pytest.fixture
def client(make_pipeline, monkeypatch):
    monkeypatch.setattr(api, "_pipeline", make_pipeline())
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
    assert len(cases) == 3
    assert all(c["query"].endswith("?") for c in cases)


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
    assert len(data["atomic_claims"]) > 0
