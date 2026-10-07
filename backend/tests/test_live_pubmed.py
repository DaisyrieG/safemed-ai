"""Live PubMed retriever, tested offline with canned E-utilities responses."""

import json
import threading

import numpy as np
import pytest

from src.stage1_retriever.live_pubmed import LivePubMedError, LivePubMedRetriever, keyword_query, parse_pubmed_xml

XML = """<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID>111</PMID><Article><Journal><Title>Heart J</Title>
<JournalIssue><PubDate><Year>2014</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>Digoxin and <i>prostate</i> cancer</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">Digoxin may matter.</AbstractText>
<AbstractText Label="RESULTS">Lower risk in users.</AbstractText></Abstract>
<PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList></Article></MedlineCitation>
<PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/x</ArticleId></ArticleIdList></PubmedData></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>222</PMID><Article><ArticleTitle>No abstract here</ArticleTitle></Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>333</PMID><Article><ArticleTitle>Vitamin C and colds</ArticleTitle>
<Abstract><AbstractText>Vitamin C did not shorten colds.</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


class FakeEncoder:
    def encode(self, texts, **_):
        return np.array([[1.0, 0.0] if ("igoxin" in t or "prostate" in t) else [0.0, 1.0] for t in texts])


def fake_retriever(responses):
    r = LivePubMedRetriever.__new__(LivePubMedRetriever)
    r.top_k, r.model, r.api_key, r.email, r.tool, r.timeout = 50, FakeEncoder(), None, None, "test", 1
    r.min_interval, r._lock, r._last_call, r._by_id = 0.0, threading.Lock(), 0.0, {}
    r.calls = []

    def _get(endpoint, params):
        r.calls.append((endpoint, params))
        return responses[endpoint](params)
    r._get = _get
    return r


def test_parse_keeps_articles_with_abstracts_and_metadata():
    docs = parse_pubmed_xml(XML)
    assert [d["pmid"] for d in docs] == ["111", "333"]
    d = docs[0]
    assert d["id"] == "pubmed_111" and d["title"] == "Digoxin and prostate cancer"
    assert d["text"] == "BACKGROUND: Digoxin may matter. RESULTS: Lower risk in users."
    assert d["journal"] == "Heart J" and d["year"] == "2014" and d["doi"] == "10.1/x"
    assert d["url"].endswith("/111/")


def test_retrieve_ranks_by_similarity_excludes_self_and_caches():
    r = fake_retriever({
        "esearch.fcgi": lambda p: json.dumps({"esearchresult": {"idlist": ["333", "111"]}}),
        "efetch.fcgi": lambda p: XML,
    })
    out = r.retrieve("Is digoxin associated with prostate cancer?", top_k=2)
    assert [d["pmid"] for d in out] == ["111", "333"] and out[0]["retrieval_score"] == 1.0
    assert {d["id"] for d in r.corpus} == {"pubmed_111", "pubmed_333"}
    assert r.calls[0][1]["sort"] == "relevance"
    assert [d["pmid"] for d in r.retrieve("digoxin prostate", top_k=2, exclude_id="111")] == ["333"]


def test_retrieve_without_abstracts_raises():
    r = fake_retriever({
        "esearch.fcgi": lambda p: json.dumps({"esearchresult": {"idlist": []}}),
        "efetch.fcgi": lambda p: "<PubmedArticleSet/>",
    })
    with pytest.raises(LivePubMedError):
        r.retrieve("nonsense question", top_k=5)


def test_keyword_query_drops_stopwords():
    assert keyword_query("Does digoxin use change the risk of prostate cancer?") == "digoxin AND use AND change AND risk AND prostate AND cancer"
