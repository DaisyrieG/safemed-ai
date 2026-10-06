"""Stage 2: Pre-Reranking Harmful Document Filter."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence
import json
import logging
import os
import re

import numpy as np

logger = logging.getLogger(__name__)

_SECTION_LABEL = re.compile(
    r"(?:^|(?<=[.!?\n]))\s*(background|objectives?|aims?|introduction|purpose|methods?|materials and methods|design|"
    r"setting|participants|patients|interventions?|measurements|results?|findings|"
    r"conclusions?|discussion|significance)\s*:\s*",
    re.IGNORECASE,
)

MAX_LENGTH = 256

Scorer = Callable[[str, Sequence[str]], np.ndarray]


def normalize_document_text(text: str) -> str:
    """Strips structured-abstract section labels and collapses whitespace."""
    return re.sub(r"\s+", " ", _SECTION_LABEL.sub(" ", text or "")).strip()


class FilterUnavailableError(RuntimeError):
    """Raised when the fine-tuned filter weights cannot be loaded."""


@dataclass
class FilterResult:
    """Audit trace of one Stage 2 pass."""
    tau_safe: float
    classified: List[Dict[str, Any]] = field(default_factory=list)
    blocked: List[Dict[str, Any]] = field(default_factory=list)
    surviving: List[Dict[str, Any]] = field(default_factory=list)
    reinstated: List[Dict[str, Any]] = field(default_factory=list)


def resolve_label_index(id2label: Dict[Any, str]) -> Dict[str, int]:
    """Maps each class name to the model's output index; refuses models without a harmful class."""
    aliases = {"ground_truth": {"ground_truth", "ground-truth", "gd"},
               "mediocre": {"mediocre", "md"},
               "harmful": {"harmful", "hd"}}
    index = {}
    for idx, name in id2label.items():
        key = str(name).strip().lower()
        for label, names in aliases.items():
            if key in names:
                index[label] = int(idx)
    if set(index) != set(aliases):
        raise FilterUnavailableError(f"filter model labels {dict(id2label)} are not ground_truth/mediocre/harmful")
    return index


class HarmfulDocumentFilter:
    def __init__(
        self,
        model_path: Optional[str] = None,
        harmful_threshold: Optional[float] = None,
        cutoff_k: int = 30,
        scorer: Optional[Scorer] = None,
        batch_size: int = 16,
    ):
        """model_path: folder written by train_filter.py (weights, tokenizer, filter_config.json)."""
        self.model_path = model_path
        self.cutoff_k = cutoff_k
        self.batch_size = batch_size
        self.config: Dict[str, Any] = {}

        if scorer is not None:
            self._score = scorer
            self.scorer_name = "injected"
            self.label_index = {"ground_truth": 0, "mediocre": 1, "harmful": 2}
        else:
            self._load_cross_encoder(model_path)
            self.scorer_name = "cross_encoder"

        if harmful_threshold is None:
            if "tau_safe" not in self.config:
                raise FilterUnavailableError("no tau_safe given and none in filter_config.json")
            harmful_threshold = float(self.config["tau_safe"])
        if not 0.0 <= harmful_threshold <= 1.0:
            raise ValueError(f"harmful_threshold (tau_safe) must be in [0, 1], got {harmful_threshold}")
        self.harmful_threshold = harmful_threshold

    def _load_cross_encoder(self, model_path: Optional[str]) -> None:
        if not model_path or not os.path.isfile(os.path.join(model_path, "config.json")):
            raise FilterUnavailableError(
                f"no fine-tuned filter at {model_path!r}. Train one with "
                "`python -m src.stage2_filter.train_filter ...` (writes backend/models/safemed_filter) "
                "or set SAFEMED_FILTER_MODEL."
            )
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        config_file = os.path.join(model_path, "filter_config.json")
        if os.path.isfile(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()
        self.label_index = resolve_label_index(self.model.config.id2label)
        self.max_length = int(self.config.get("max_length", MAX_LENGTH))
        self._torch = torch
        logger.info("Loaded filter %s (labels %s)", model_path, self.label_index)

    def _score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        """Softmax probabilities for each (query, text) pair, scored in batches."""
        torch = self._torch
        rows = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            enc = self.tokenizer([query] * len(batch), batch, truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt")
            with torch.no_grad():
                rows.append(torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy())
        return np.concatenate(rows) if rows else np.zeros((0, 3))

    def classify(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scores every candidate and marks it blocked when P(HD) >= tau_safe."""
        texts = [normalize_document_text(c.get("text", "")) for c in candidates]
        probs = self._score(query, texts) if texts else np.zeros((0, 3))
        idx = self.label_index

        classified = []
        for cand, p in zip(candidates, probs):
            label_probs = {label: round(float(p[i]), 4) for label, i in idx.items()}
            p_hd = float(p[idx["harmful"]])
            is_blocked = p_hd >= self.harmful_threshold
            classified.append({
                **cand,
                "label_probabilities": label_probs,
                "harmful_probability": round(p_hd, 4),
                "filter_label": max(label_probs, key=label_probs.get),
                "scorer": self.scorer_name,
                "tau_safe": self.harmful_threshold,
                "is_blocked": is_blocked,
                "reinstated": False,
                "block_reason": (f"P(harmful) = {p_hd:.2f} >= tau_safe = {self.harmful_threshold:.2f}"
                                 if is_blocked else None),
            })

        logger.info("Stage 2 scored %d candidates: %d blocked at tau_safe=%.2f",
                    len(classified), sum(c["is_blocked"] for c in classified), self.harmful_threshold)
        return classified

    def filter(self, query: str, candidates: List[Dict[str, Any]]) -> FilterResult:
        """Full Stage 2 pass: block, keep the top 30 in retriever order, then apply the Safety Fallback Guard."""
        from src.stage2_filter.fallback_guard import apply_safety_fallback_guard

        classified = self.classify(query, candidates)
        blocked = [d for d in classified if d["is_blocked"]]
        surviving = [d for d in classified if not d["is_blocked"]][: self.cutoff_k]
        surviving, reinstated = apply_safety_fallback_guard(surviving, blocked)
        return FilterResult(self.harmful_threshold, classified, blocked, surviving, reinstated)
