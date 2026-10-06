# SafeMed AI: A Pre-Reranking Cross-Encoder Filter for Reducing Harmful Document Exposure in Biomedical RAG

> **Group 7 Thesis Codebase & Prototype**
> Polytechnic University of the Philippines — College of Computer and Information Sciences
> Bachelor of Science in Computer Science (June 2026)
> **Authors**: Erika P. Biso, Althea May T. Busilacan, Daisyrie I. Gente, Christelle Joy A. Samson

---

## Repository structure

`backend/src/` mirrors the five stages of the Chapter 3 system architecture.

```
Thesis/
├── backend/
│   ├── src/
│   │   ├── pipeline.py           # Orchestrator: runs Stages 1–4 for the control and treatment conditions
│   │   ├── stage1_retriever/     # Stage 1  all-MiniLM-L6-v2 bi-encoder, top-50, leave-self-out
│   │   │   ├── retriever.py
│   │   │   ├── build_corpus.py   #   pools pqa_artificial into one corpus + 1,000/250/500 query split
│   │   │   └── build_index.py    #   embeds the corpus into a FAISS index
│   │   ├── stage2_filter/        # Stage 2  fine-tuned 3-class cross-encoder, tau_safe, top-30
│   │   │   ├── cross_encoder_filter.py
│   │   │   ├── fallback_guard.py #   Safety Fallback Guard (keeps |S| >= 5)
│   │   │   └── train_filter.py   #   fine-tuning + tau_safe selection on validation
│   │   ├── stage3_reranker/      # Stage 3  ms-marco-MiniLM-L-6-v2 reranker, top-5
│   │   ├── stage4_generator/     # Stage 4  LLM answer (GPT-4o-mini by default), answer-to-source attribution
│   │   ├── stage5_evaluation/    # Stage 5  benchmark runner, claim judge, metrics, Wilcoxon tests
│   │   ├── annotation/           # Document Annotation Guide (rubric.md), LLM labeller, Cohen's kappa
│   │   └── api/                  # FastAPI app for the web demo (+ open-access full-text PDFs)
│   ├── scripts/                  # one-off data scripts (demo corpus, synthetic harmful docs, filter training pairs, annotation sheet)
│   ├── data/
│   │   ├── raw/                  # ori_pqaa.json (pqa_artificial; not committed)
│   │   ├── processed/            # pooled corpus, FAISS index, query splits (built locally; not committed)
│   │   ├── demo/                 # web-demo corpus: 1,000 PQA-L abstracts + labelled synthetic harmful docs
│   │   └── annotations/          # labelled (query, document) pairs for filter training
│   ├── models/                   # trained filter weights (not committed)
│   ├── results/                  # benchmark output (evaluation_results.json, traces)
│   └── tests/
├── frontend/                     # Web app (Vite + React)
├── third_party/pubmedqa/         # PubMedQA (MIT): PQA-L data and the official evaluation script
└── documents/                    # Thesis manuscript, defense handbook, UML diagrams
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

The LLM is configured in `backend/.env` (see `backend/.env.example`): `OPENAI_API_KEY` for GPT-4o-mini,
or `SAFEMED_LLM_BASE_URL` / `SAFEMED_LLM_MODEL` for a local OpenAI-compatible server such as LM Studio.

## Running the web demo

```bash
cd backend
python src/api/app.py             # http://127.0.0.1:8000  (docs: /docs)

cd frontend
npm install
npm run dev                       # http://localhost:8443
```

The demo searches `backend/data/demo/` and needs the trained filter in `backend/models/safemed_filter/`.
Every stage loads its real model or stops with an error; there are no heuristic fallbacks.

## Reproducing the experiment (all commands from `backend/`)

```bash
# 1. Corpus and query partition from pqa_artificial (put ori_pqaa.json in data/raw/ first)
python -m src.stage1_retriever.build_corpus
python -m src.stage1_retriever.build_index

# 2. Filter: labelled pairs -> fine-tuning -> tau_safe on validation
python -m src.stage2_filter.train_filter --train data/annotations/filter_train.jsonl \
    --val data/annotations/filter_val.jsonl --test data/annotations/filter_test.jsonl \
    --output-dir models/safemed_filter --epochs 8

# 3. Benchmark on the 500 test queries (needs per-pair gold labels for the test candidates)
python -m src.stage5_evaluation.run_benchmark --labels data/annotations/test_pair_labels.jsonl
```

## Tests

```bash
cd backend
python -m pytest tests -q
```
