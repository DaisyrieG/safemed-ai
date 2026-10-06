"""Full-text PDF fetch (mocked network) and PyMuPDF highlighting."""

import json

import httpx
import pytest

pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("rapidfuzz")

from src.api.fulltext import pmc_fetch  # noqa: E402
from src.api.fulltext.pdf_highlight import highlight_sentences  # noqa: E402


def _make_pdf(path):
    doc = pymupdf.open()
    page = doc.new_page()
    lines = [
        "Background: Digoxin has been associated with lower prostate cancer risk in some",
        "studies. We examined the association in a large cohort of men treated for cardio-",
        "vascular disease. Methods: We followed 47,884 men for incident prostate cancer.",
        "Results: Digoxin use was associated with a 24% lower risk of prostate cancer.",
    ]
    for i, line in enumerate(lines):
        page.insert_text((50, 72 + 14 * i), line, fontsize=10)
    doc.save(str(path))
    return path.read_bytes()


def _mock_transport(pdf_bytes, open_access=True):
    def handler(request: httpx.Request) -> httpx.Response:
        if "europepmc/webservices/rest/search" in str(request.url):
            result = {"pmcid": "PMC123", "isOpenAccess": "Y" if open_access else "N", "license": "cc by",
                      "title": "Digoxin and prostate cancer", "fullTextUrlList": {"fullTextUrl": []}}
            return httpx.Response(200, json={"resultList": {"result": [result]}})
        if "oa.fcgi" in str(request.url):
            return httpx.Response(200, text="<OA><records/></OA>")
        if "pdf=render" in str(request.url):
            return httpx.Response(200, content=pdf_bytes)
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def test_highlight_handles_line_breaks_and_hyphenation(tmp_path):
    src = tmp_path / "paper.pdf"
    _make_pdf(src)
    results = highlight_sentences(str(src), [
        "We examined the association in a large cohort of men treated for cardiovascular disease.",
        "Digoxin use was associated with a 24% lower risk of prostate cancer.",
        "Aspirin prevents colorectal cancer in Lynch syndrome carriers.",
    ], str(tmp_path / "out.pdf"))
    assert [r.found for r in results] == [True, True, False]
    assert results[0].method in {"tokens", "fuzzy"}
    assert results[1].page == 1
    out = pymupdf.open(str(tmp_path / "out.pdf"))
    assert len(list(out[0].annots())) == 2


def test_fetch_downloads_open_access_pdf_and_caches(tmp_path):
    pdf_bytes = _make_pdf(tmp_path / "src.pdf")
    with httpx.Client(transport=_mock_transport(pdf_bytes)) as client:
        info = pmc_fetch.get_fulltext_pdf("pqa_24318956", cache_dir=str(tmp_path / "cache"), client=client)
    assert info.pmcid == "PMC123" and info.has_pdf
    meta = json.loads((tmp_path / "cache" / "24318956" / "meta.json").read_text())
    assert meta["is_open_access"] is True
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
        again = pmc_fetch.get_fulltext_pdf("24318956", cache_dir=str(tmp_path / "cache"), client=client)
    assert again.has_pdf


def test_fetch_without_open_access_has_no_pdf(tmp_path):
    with httpx.Client(transport=_mock_transport(b"", open_access=False)) as client:
        info = pmc_fetch.get_fulltext_pdf("24318956", cache_dir=str(tmp_path), client=client)
    assert not info.has_pdf and info.error is None


def test_highlighted_pdf_endpoint(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from src.api.app import app

    pdf_bytes = _make_pdf(tmp_path / "src.pdf")
    monkeypatch.setattr(pmc_fetch, "DEFAULT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(pmc_fetch, "_client", lambda timeout=30.0: httpx.Client(transport=_mock_transport(pdf_bytes)))
    client = TestClient(app)

    status = client.get("/api/documents/24318956/fulltext").json()
    assert status["has_pdf"] is True
    response = client.post("/api/documents/24318956/highlighted-pdf",
                           json={"sentences": ["Digoxin use was associated with a 24% lower risk of prostate cancer."]})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["x-highlights-found"] == "1"
    assert client.get("/api/documents/not-a-pmid/fulltext").status_code == 400
