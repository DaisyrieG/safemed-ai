# Upstream source

Vendored copy of https://github.com/pubmedqa/pubmedqa
at commit `1cbae8e92f72f20c8d3747cbb3bf5bc53554d997` (2023-04-18), copied 2026-10-04.

Original work: Jin, Q., Dhingra, B., Liu, Z., Cohen, W. W., & Lu, X. (2019).
PubMedQA: A Dataset for Biomedical Research Question Answering. EMNLP-IJCNLP 2019.
Licensed under the MIT License (see `LICENSE` in this folder, kept unchanged).

Files are unmodified. The upstream repository does not contain the large
`ori_pqaa.json` (pqa_artificial) or `ori_pqau.json` (pqa_unlabeled) files; per its
README they are downloaded separately from Google Drive. Place them in
`backend/data/raw/` or `experiments/data/raw/` (git-ignored) for the SafeMed AI evaluation.
