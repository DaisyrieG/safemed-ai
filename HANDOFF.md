# Handoff: SafeMed AI thesis sprint

Last updated: Oct 6, 2026, 7:20 AM. **Tool defense: Oct 8, 2026. Everything ready by Oct 8, 5:00 AM.**

If you are a new Claude session: read this file and `README.md` first, then ask which step we are on.

---

## Where things stand

### Done
- **Codebase** reorganised by Chapter 3 stage: `backend/src/stage1_retriever` … `stage5_evaluation`, `annotation/`, `api/`, `pipeline.py`. All tests pass (`cd backend && python -m pytest tests -q`: 43 passed, 1 skipped).
- **No heuristic fallbacks.** Every stage loads its real model or raises an error.
- **pqa_artificial corpus**: 211,269 abstracts in `backend/data/processed/pqaa_corpus.json`, plus the query split (1,000 train / 250 val / 500 test, seed 42) in `pqaa_splits.json`. Both are local only; rebuild with `python -m src.stage1_retriever.build_corpus` (6 s).
- **Stage 2 filter** trained on the demo data (PQA-L + 604 synthetic HD written by Qwen), in `backend/models/safemed_filter/` (local only, **back it up**). Held-out test: macro-F1 0.963, HD recall 99.0% (95/96), GD blocked 2.1% (1/48), MD blocked 0%, `tau_safe` 0.76.
- **Groq (free tier)** set up in `backend/.env`: judge `openai/gpt-oss-120b`, labeller `qwen/qwen3.8-27b`. Both tested.
- **`backend/scripts/annotate_dataset.py`** written and tested (labels 10 pairs per request).

### In progress
- FAISS index build (`python -m src.stage1_retriever.build_index`, about 84 min on CPU). It's done when `backend/data/processed/pqaa_corpus.faiss` exists.

---

## Checklist

### Phase A: the tool defense (must be done)
- [ ] **A1** FAISS index finished
- [ ] **A2** Back up `backend/models/safemed_filter/` (USB or Drive)
- [ ] **A3** Live demo end to end: LM Studio + Qwen server → `cd backend && python src/api/app.py` → `cd frontend && npm run dev` → run the 3 sample cases. Check: scorer `cross_encoder`, synthetic HD documents BLOCKED (expected: control top-5 has 2 HD, filtered top-5 has 0), the answer starts with "Decision:".
- [ ] **A4** Record a backup video of one full query, and screenshots of the 3 cases
- [ ] **A5** Simplify the frontend to a query box + answer + citations (no benchmark UI)
- [ ] **A6** Adviser approval of the deviations (see below), and ask whether benchmark results are expected on Oct 8
- [ ] **A7** Slides: architecture, filter results, the shortcut-learning story, the experiment plan

### Phase B: the experiment (nice to have by Oct 8; required for the final defense)
- [ ] **B1** `cd backend && python scripts/annotate_dataset.py --limit 20` (test), then without `--limit` (1,250 train/val pairs, about 2 h). Output: `backend/data/annotations/pqaa_llm_labels.jsonl`. **Check the HD count**: real pqa_artificial may have very few harmful documents.
- [ ] **B2** Human check: adapt `scripts/annotation_sheet.py` to read `pqaa_llm_labels.jsonl`, add a correct-answer column, and threshold on κ(LLM vs humans). Plan: 2 teams × 2 people × 150 pairs (= 300), or a 50-pair pilot if short on time. Target κ ≥ 0.80.
- [ ] **B3** Build train/val files from the labels. If HD is rare, add the existing synthetic HD pairs and report it.
- [ ] **B4** Retrain the filter, 3 seeds, into a **separate folder** (e.g. `models/safemed_filter_pqaa`). **Don't overwrite `models/safemed_filter`; the demo uses it.**
- [ ] **B5** Label all 50 candidates of 200 test queries (10,000 pairs). Batch by query: 1,000 Groq requests/day limit.
- [ ] **B6** Batch the judge's claim verification into one request per answer (Groq limit).
- [ ] **B7** `python -m src.stage5_evaluation.run_benchmark --n 200 --labels <test labels> --filter-model models/safemed_filter_pqaa`
- [ ] **B8** Chapter 4: Wilcoxon H1–H4, SOP1 with bootstrap CIs, deviations paragraph

### Phase C: defense morning
- [ ] LM Studio + Qwen loaded, backend and frontend running, one warm-up query done, backup video open

---

## Decisions already made (don't re-ask)
- **No paid API.** OpenAI is not affordable. Azure for Students ($100) **cannot create Azure OpenAI**: the subscription only allows `eastasia, indonesiacentral, centralindia, malaysiawest, indiasouthcentral`, and none of them offer it.
- **Models:** generator = local Qwen2.5-7B-Instruct (LM Studio, `SAFEMED_LLM_*`); judge = Groq `openai/gpt-oss-120b`; labeller = Groq `qwen/qwen3.8-27b`. Judge ≠ generator, as Chapter 3 requires.
- **Deviations from Chapter 3** (need adviser approval and a paragraph in Ch. 3/4): open-weight models instead of GPT-4o-mini/GPT-4o; N = 200 test queries (above the power-analysis minimum of 194); demo filter trained on synthetic HD.
- **Groq free-tier limits** (per model): 1,000 requests/day, 8,000 input tokens/min, 1,000 output tokens/min (qwen). gpt-oss models are reasoning models and need a generous `max_tokens`.
- **Demo labels are per query** (`mark_source_abstract` in `pipeline.py`): a synthetic HD counts as harmful only for its target question.
- **Code style:** no `#` comments; one-line docstrings only.

## Lessons learned (good material for the panel)
1. First filter: 98% scores came from **title leakage**. PubMedQA titles are the questions, so the model copied them; in the live pipeline it blocked 92% of GD documents. Fixed by training and running on the abstract only.
2. A 3-epoch abstract-only model reached 54% (undertrained); 8 epochs gave 97%.
3. About 25% of HD detection came from **"Background:" headings** that only the synthetic documents had. They're now stripped from all text (`normalize_document_text`). Final model: 99% HD recall, 2.1% GD blocked.
4. Checks done: an HD paired with the wrong question → 0/96 blocked (the filter uses the query); shortening GD to HD length → 3/48 blocked (not a length shortcut).

---

## Local-only files (not on GitHub; copy by USB to move laptops)
| Item | Path | If missing |
|---|---|---|
| Trained filter | `backend/models/safemed_filter/` | Retrain: `python -m src.stage2_filter.train_filter --train data/annotations/filter_train.jsonl --val data/annotations/filter_val.jsonl --test data/annotations/filter_test.jsonl --output-dir models/safemed_filter --epochs 8` (~30 min) |
| Settings + Groq key | `backend/.env` | Recreate from `.env.example` + the Groq lines above |
| pqa_artificial raw | `backend/data/raw/ori_pqaa.json` | Download from pubmedqa.github.io |
| Corpus + index | `backend/data/processed/` | `build_corpus` (6 s) + `build_index` (~84 min) |
| Python env | `.venv` | `python -m venv .venv` → `.venv\Scripts\Activate.ps1` → `pip install -r requirements.txt` |

## Common pitfalls on this laptop
- **Always activate `.venv`.** The system Python 3.14 is blocked by Windows Application Control (pyarrow DLL). The prompt must show `(.venv)`.
- **Run `python -m src....` from inside `backend/`**, not the repo root.
- **Don't run two CPU-heavy jobs at once** (training, index build, benchmark). Groq calls are fine alongside anything.
- **Unsaved VS Code tabs** can overwrite newer files on save. Use *Revert File* if a tab looks out of date.
