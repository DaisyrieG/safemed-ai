"""Zips what the Colab notebook needs: the Stage 2 code and the two training files (no keys, no models)."""

import os
import zipfile

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILES = [
    "src/__init__.py",
    "src/stage2_filter/__init__.py",
    "src/stage2_filter/cross_encoder_filter.py",
    "src/stage2_filter/fallback_guard.py",
    "src/stage2_filter/train_filter.py",
    "data/annotations/pqaa_filter_train.jsonl",
    "data/annotations/pqaa_filter_val.jsonl",
]

out = os.path.join(BACKEND, "colab", "filter_colab_bundle.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in FILES:
        z.write(os.path.join(BACKEND, rel), rel)
print(f"{out} ({os.path.getsize(out) / 1e6:.1f} MB)")
