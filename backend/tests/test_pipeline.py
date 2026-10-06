"""Tests for each pipeline stage and the Chapter 3 mechanics."""

import json

import pytest

from tests.conftest import fake_llm_client, oracle_scorer
from src.pipeline import POOL_SIZE, mark_source_abstract, resolve_corpus_path, DEMO_CORPUS
from src.stage1_retriever.retriever import BiEncoderRetriever
from src.stage2_filter.cross_encoder_filter import FilterUnavailableError, HarmfulDocumentFilter
from src.stage2_filter.fallback_guard import apply_safety_fallback_guard
from src.stage4_generator.generator import AnswerGenerator, GeneratorUnavailableError

QUERY = "Are COVID-19 mRNA vaccines associated with increased risk of myocarditis in young adults?"


def test_stage_1_returns_top_50_with_scores(retriever):
    candidates = retriever.retrieve(QUERY)
    assert len(candidates) == 50
    assert "text" in candidates[0] and "retrieval_score" in candidates[0]
    scores = [c["retrieval_score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_leave_self_out_excludes_source_document(retriever):
    top_id = retriever.retrieve(QUERY, top_k=10)[0]["id"]
    excluded = retriever.retrieve(QUERY, top_k=10, exclude_id=top_id)
    assert len(excluded) == 10
    assert all(d["id"] != top_id for d in excluded)


def test_leave_self_out_matches_pmid_and_pqa_prefix():
    r = BiEncoderRetriever(top_k=5)
    r.build_index([
        {"id": "pqa_111", "pmid": "111", "text": "aspirin colorectal cancer lynch syndrome"},
        {"id": "pqa_222", "pmid": "222", "text": "aspirin colorectal cancer"},
        {"id": "pqa_333", "pmid": "333", "text": "unrelated cardiology text"},
    ])
    for exclude in ("111", "pqa_111", 111):
        ids = [d["id"] for d in r.retrieve("aspirin colorectal cancer lynch syndrome", exclude_id=exclude)]
        assert "pqa_111" not in ids
        assert len(ids) == 2


def _const_filter(p_hd, tau=0.5):
    import numpy as np
    return HarmfulDocumentFilter(scorer=lambda q, texts: np.array([[0.3, 0.7 - p_hd, p_hd]] * len(texts)),
                                 harmful_threshold=tau)


def test_tau_safe_blocks_at_and_above_threshold():
    blocked = _const_filter(0.5).classify("q", [{"id": "d1", "text": "x"}])[0]
    assert blocked["is_blocked"] is True and blocked["block_reason"]
    passed = _const_filter(0.4999).classify("q", [{"id": "d1", "text": "x"}])[0]
    assert passed["is_blocked"] is False and passed["block_reason"] is None


def test_filter_blocks_harmful_and_keeps_top_30_in_retriever_order(retriever, fixture_corpus):
    f = HarmfulDocumentFilter(scorer=oracle_scorer(fixture_corpus), harmful_threshold=0.5)
    candidates = retriever.retrieve(QUERY)
    result = f.filter(QUERY, candidates)
    assert len(result.classified) == 50
    assert result.blocked and all(d.get("true_label") == "harmful" for d in result.blocked)
    assert len(result.surviving) <= POOL_SIZE
    assert not any(d["is_blocked"] for d in result.surviving)
    order = [d["id"] for d in candidates if not any(b["id"] == d["id"] for b in result.blocked)][:POOL_SIZE]
    assert [d["id"] for d in result.surviving] == order


def test_filter_refuses_to_start_without_trained_weights(tmp_path):
    with pytest.raises(FilterUnavailableError):
        HarmfulDocumentFilter(model_path=str(tmp_path / "missing"))


def test_safety_fallback_guard_reinstates_lowest_risk_until_five():
    surviving = [{"id": "s1", "harmful_probability": 0.1}, {"id": "s2", "harmful_probability": 0.2}]
    blocked = [
        {"id": "b_high", "harmful_probability": 0.95},
        {"id": "b_low", "harmful_probability": 0.55},
        {"id": "b_mid", "harmful_probability": 0.70},
        {"id": "b_mid2", "harmful_probability": 0.80},
    ]
    S, events = apply_safety_fallback_guard(surviving, blocked, min_docs=5)
    assert len(S) == 5
    assert [e["id"] for e in events] == ["b_low", "b_mid", "b_mid2"]
    assert all(d["reinstated"] for d in S[2:])
    assert "b_high" not in {d["id"] for d in S}


def test_safety_fallback_guard_noop_when_enough_survive():
    surviving = [{"id": f"s{i}", "harmful_probability": 0.1} for i in range(6)]
    S, events = apply_safety_fallback_guard(surviving, [{"id": "b", "harmful_probability": 0.6}])
    assert len(S) == 6
    assert events == []


def test_stage_3_reranks_to_top_5(reranker):
    candidates = [{"id": f"doc-{i}", "text": f"Cardiology evidence {i}"} for i in range(20)]
    top5 = reranker.rerank("cardiology query", candidates)
    assert len(top5) == 5
    scores = [d["rerank_score"] for d in top5]
    assert scores == sorted(scores, reverse=True)


def test_generator_raises_without_a_client():
    gen = AnswerGenerator(api_key="")
    gen.client = None
    with pytest.raises(GeneratorUnavailableError):
        gen.generate_answer("q", [{"title": "t", "text": "x"}])


def test_generator_calls_gpt4o_mini_at_temperature_zero_with_top5():
    calls = []
    gen = AnswerGenerator(client=fake_llm_client(calls), model="gpt-4o-mini")
    answer = gen.generate_answer("q", [{"title": f"T{i}", "text": f"text {i}"} for i in range(7)])
    assert answer.startswith("Decision: yes")
    assert calls[0]["model"] == "gpt-4o-mini"
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["messages"][1]["content"].count("Source [") == 5


def test_claims_exclude_the_decision_line():
    gen = AnswerGenerator(client=fake_llm_client(reply="Decision: yes\nAspirin lowers risk.\n- Effect lasts 10 years."))
    assert gen.decompose_into_claims("...") == ["Aspirin lowers risk.", "Effect lasts 10 years."]


def test_generator_reads_local_llm_settings_from_env(monkeypatch):
    monkeypatch.setenv("SAFEMED_LLM_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("SAFEMED_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = []
    gen = AnswerGenerator(client=fake_llm_client(calls))
    assert gen.model == "qwen2.5:7b"
    assert gen.base_url == "http://localhost:11434/v1"
    assert gen.api_key == "local"
    gen.generate_answer("q", [{"title": "t", "text": "x"}])
    assert calls[0]["model"] == "qwen2.5:7b"


def test_pipeline_end_to_end(make_pipeline):
    pipe = make_pipeline()
    result = pipe.run(QUERY)
    assert result["safety_scan"]["total_scanned"] == 50
    assert result["safety_scan"]["blocked_count"] > 0
    assert result["safety_scan"]["scorer"] == "injected"
    assert len(result["trusted_sources"]) == 5
    assert all(s["verification_status"] == "verified safe" for s in result["trusted_sources"])
    assert result["clinical_summary"].startswith("Decision:")
    assert result["evaluation_breakdown"]["proposed"]["hd_count"] == 0


def test_control_reranks_the_top_30_by_retriever_similarity(make_pipeline, retriever):
    pipe = make_pipeline()
    pipe.run(QUERY)
    top30_ids = {d["id"] for d in retriever.retrieve(QUERY)[:POOL_SIZE]}
    assert {d["id"] for d in pipe.last_control_top5} <= top30_ids


def test_pipeline_records_reinstatement_in_filter_result(make_pipeline):
    pipe = make_pipeline(tau=0.0)
    result = pipe.run(QUERY)
    fr = pipe.last_filter_result
    assert len(fr.surviving) == 5 and len(fr.reinstated) == 5
    assert result["safety_scan"]["reinstated_count"] == 5
    assert result["safety_scan"]["final_pool_size"] == 5
    statuses = [e["status"] for e in result["safety_scan"]["evaluation_log"]]
    assert statuses.count("REINSTATED: FALLBACK GUARD") == 5


def test_pipeline_returns_attribution(make_pipeline):
    result = make_pipeline().run("Does daily aspirin reduce colorectal cancer in Lynch syndrome?")
    assert result["attribution"]["method"] in {"embedding", "lexical"}
    assert len(result["attribution"]["sources"]) == len(result["trusted_sources"])


def test_attribution_parses_citations_and_highlights_supporting_sentence():
    from src.stage4_generator.attribution import attribute_answer, parse_citations

    assert parse_citations("Aspirin helps [1, 3] and [2-3].", 5) == [1, 3, 2]
    assert parse_citations("Out of range [9].", 5) == []

    sources = [
        {"text": "Daily aspirin reduced colorectal cancer incidence in Lynch syndrome carriers. Follow-up was ten years."},
        {"text": "Cartilage wear was studied in bovine samples."},
    ]
    answer = "Decision: yes. Daily aspirin reduced colorectal cancer incidence in Lynch syndrome [1]."
    result = attribute_answer(answer, sources, encoder=None)

    assert result["method"] == "lexical"
    claim = result["answer_sentences"][-1]
    assert claim["cited"] == [1]
    assert claim["supported_by"][0]["source"] == 1
    assert claim["supported_by"][0]["sentence"] == 0
    assert result["sources"][0]["sentences"][0]["answer_sentences"] == [claim["index"]]
    assert result["sources"][1]["sentences"][0]["answer_sentences"] == []


def test_source_abstract_becomes_ground_truth_for_its_own_question():
    docs = [
        {"id": "pqa_1", "title": "Is digoxin use associated with prostate cancer?"},
        {"id": "pqa_2", "title": "Another question?"},
        {"id": "hd_1_1", "title": "Is digoxin use associated with prostate cancer?", "true_label": "harmful"},
    ]
    marked = mark_source_abstract("is digoxin use associated with prostate cancer", docs)
    assert [d.get("true_label") for d in marked] == ["ground_truth", None, "harmful"]
    assert "true_label" not in docs[0]


def test_synthetic_harmful_document_is_harmful_only_for_its_target_question():
    hd = {"id": "hd_1_1", "true_label": "harmful", "target_question": "Is digoxin associated with prostate cancer?"}
    assert mark_source_abstract("Is digoxin associated with prostate cancer?", [hd])[0]["true_label"] == "harmful"
    assert mark_source_abstract("Do aromatase inhibitors raise cardiac risk?", [hd])[0]["true_label"] == "mediocre"


def test_resolve_corpus_path():
    assert resolve_corpus_path("demo") == DEMO_CORPUS
    with pytest.raises(FileNotFoundError):
        resolve_corpus_path("does-not-exist.json")


def _tiny_base_model(path, id2label=None):
    """A 1-layer BERT with a small vocabulary, saved locally so no download is needed."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "digoxin", "prostate", "cancer", "risk", "lower",
             "cure", "miracle", "detox", "all", "patients", "study", "cohort", "is", "associated", "with", "the",
             "and", "of", "a", "in", "no", "effect", "vitamin", "d", "surgery", "july", "outcomes"]
    path.mkdir(parents=True, exist_ok=True)
    (path / "vocab.txt").write_text("\n".join(words))
    BertTokenizerFast(vocab_file=str(path / "vocab.txt")).save_pretrained(str(path))
    config = BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=300, num_labels=len(id2label or {0: "x"}),
                        id2label=id2label or {0: "relevance"})
    BertForSequenceClassification(config).save_pretrained(str(path))
    return str(path)


def _write_pairs(path, n=6):
    rows = []
    for _ in range(n):
        rows.append({"query": "is digoxin associated with prostate cancer risk", "text": "cohort study digoxin lower prostate cancer risk", "label": "ground_truth"})
        rows.append({"query": "is digoxin associated with prostate cancer risk", "text": "miracle detox cure all patients", "label": "harmful"})
        rows.append({"query": "is digoxin associated with prostate cancer risk", "text": "vitamin d and july surgery outcomes", "label": "mediocre"})
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


def test_train_filter_writes_model_with_harmful_label_and_tau(tmp_path):
    pytest.importorskip("torch")
    import numpy as np
    from src.stage2_filter.train_filter import train, choose_tau, HARMFUL_ID

    base = _tiny_base_model(tmp_path / "base")
    out = tmp_path / "trained"
    report = train(_write_pairs(tmp_path / "train.jsonl"), _write_pairs(tmp_path / "val.jsonl", 2), str(out),
                   test_data_path=_write_pairs(tmp_path / "test.jsonl", 2), base_model=base, epochs=1, batch_size=4)
    assert 0.05 <= report["tau_safe"] <= 0.95
    config = json.loads((out / "filter_config.json").read_text())
    assert config["id2label"][str(HARMFUL_ID)] == "harmful"

    f = HarmfulDocumentFilter(model_path=str(out))
    assert f.scorer_name == "cross_encoder"
    assert f.label_index == {"ground_truth": 0, "mediocre": 1, "harmful": 2}
    assert f.harmful_threshold == config["tau_safe"]
    scored = f.classify("is digoxin associated with prostate cancer risk", [{"id": "d1", "text": "miracle detox cure"}])
    assert 0.0 <= scored[0]["harmful_probability"] <= 1.0

    p = np.array([0.9, 0.8, 0.6, 0.2, 0.1])
    y = np.array([HARMFUL_ID, HARMFUL_ID, 0, 0, 1])
    assert choose_tau(p, y, max_gd_block=0.0)["tau_safe"] > 0.6


def test_filter_reads_harmful_index_from_id2label(tmp_path):
    pytest.importorskip("torch")
    reordered = _tiny_base_model(tmp_path / "reordered", {0: "ground_truth", 1: "harmful", 2: "mediocre"})
    assert HarmfulDocumentFilter(model_path=reordered, harmful_threshold=0.5).label_index["harmful"] == 1

    generic = _tiny_base_model(tmp_path / "generic", {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    with pytest.raises(FilterUnavailableError):
        HarmfulDocumentFilter(model_path=generic, harmful_threshold=0.5)
