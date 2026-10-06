"""Test doubles: an oracle filter scorer and a fake LLM client; the real retriever and reranker run."""

import json
import os
import types

import numpy as np
import pytest

from src.stage2_filter.cross_encoder_filter import normalize_document_text

FIXTURE_CORPUS = os.path.join(os.path.dirname(__file__), "fixtures", "sample_corpus.json")


def load_fixture_corpus():
    with open(FIXTURE_CORPUS, "r", encoding="utf-8") as f:
        return json.load(f)


def oracle_scorer(corpus):
    """Columns GD, MD, HD from the fixture's labels."""
    labels = {normalize_document_text(d["text"]): d.get("true_label") for d in corpus}
    rows = {"harmful": [0.1, 0.1, 0.8], "ground_truth": [0.7, 0.2, 0.1]}

    def score(query, texts):
        return np.array([rows.get(labels.get(t), [0.2, 0.7, 0.1]) for t in texts])
    return score


def fake_llm_client(calls=None, reply="Decision: yes\nDaily aspirin reduced colorectal cancer risk [1]."):
    def create(**kw):
        if calls is not None:
            calls.append(kw)
        msg = types.SimpleNamespace(content=reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


@pytest.fixture(scope="session")
def fixture_corpus():
    return load_fixture_corpus()


@pytest.fixture(scope="session")
def retriever(fixture_corpus):
    from src.stage1_retriever.retriever import BiEncoderRetriever
    r = BiEncoderRetriever(top_k=50)
    r.build_index(fixture_corpus)
    return r


@pytest.fixture(scope="session")
def reranker():
    from src.stage3_reranker.reranker import Reranker
    return Reranker(top_k=5)


@pytest.fixture
def make_pipeline(retriever, reranker, fixture_corpus):
    """Builds a SafeMedPipeline over the fixture corpus with the oracle filter and a fake LLM."""
    from src.pipeline import SafeMedPipeline
    from src.stage2_filter.cross_encoder_filter import HarmfulDocumentFilter
    from src.stage4_generator.generator import AnswerGenerator

    def build(tau=0.5, use_filter=True):
        return SafeMedPipeline(
            use_filter=use_filter,
            retriever=retriever,
            harm_filter=HarmfulDocumentFilter(scorer=oracle_scorer(fixture_corpus), harmful_threshold=tau),
            reranker=reranker,
            generator=AnswerGenerator(client=fake_llm_client(), model="gpt-4o-mini"),
        )
    return build
