# SafeMed AI — Backend

See the [root README](../README.md) for the folder layout, setup, the web demo and the experiment commands.

## API endpoints

* `GET /api/health` — which model each stage loaded, tau_safe and the top-30 cutoff.
* `GET /api/sample-cases` — preset PubMedQA questions for the demo.
* `POST /api/query` — runs a question through Stages 1–4 (both conditions) and returns the answer, the
  Stage 2 safety log and the control-vs-treatment comparison.
* `GET /api/evaluation-results` — the latest Stage 5 benchmark output (`results/evaluation_results.json`).
* `GET /api/documents/{pmid}/fulltext`, `POST /api/documents/{pmid}/highlighted-pdf` — open-access PDF with the cited sentences highlighted.
