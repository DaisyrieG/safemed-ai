Pooled PubMedQA pqa_artificial retrieval corpus (not committed; ~211k abstracts).

Build it from backend/ after placing ori_pqaa.json in backend/data/raw/:

    python -m src.stage1_retriever.build_corpus   # pqaa_corpus.json + pqaa_splits.json (1,000 / 250 / 500 queries, seed 42)
    python -m src.stage1_retriever.build_index    # pqaa_corpus.faiss (all-MiniLM-L6-v2, exact cosine)
