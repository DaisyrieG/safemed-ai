"""Development phase, step 4 (Chapter 3): LLM candidate labels for (query, document) pairs."""

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.stage4_generator.llm_client import chat_completion, make_client

load_dotenv()

LABELS = {"ground_truth", "harmful", "mediocre"}
ALIASES = {"gd": "ground_truth", "hd": "harmful", "md": "mediocre", "ground truth": "ground_truth"}
RUBRIC_PATH = os.path.join(os.path.dirname(__file__), "rubric.md")


class LabelerUnavailableError(RuntimeError):
    """Raised when the labelling model cannot be called or returns an invalid label."""


def make_labeler_client(model: Optional[str] = None):
    """Returns (client, model name) for the configured labelling model."""
    model = model or os.getenv("SAFEMED_LABELER_MODEL") or "gpt-4o-mini"
    base_url = os.getenv("SAFEMED_LABELER_BASE_URL") or None
    key = os.getenv("SAFEMED_LABELER_API_KEY") or os.getenv("OPENAI_API_KEY") or ("local" if base_url else None)
    client, error = make_client(key, base_url)
    if client is None:
        raise LabelerUnavailableError(f"labelling model '{model}' unavailable: {error}")
    return client, model


def _system_prompt(rubric_path: str) -> str:
    with open(rubric_path, "r", encoding="utf-8") as f:
        rubric = f.read()
    return (
        f"You are an expert biomedical annotator following this codebook:\n{rubric}\n"
        "You will receive numbered (query, document) pairs. For each pair, decide whether the document is "
        "'ground_truth', 'harmful', or 'mediocre' FOR THAT QUERY, judged against the correct answer given "
        "with the query. Reply in JSON only: "
        '{"labels": [{"pair": <number>, "label": "<category>", "rationale": "<at most 15 words>"}]} '
        "with one entry per pair, in order. <category> must be exactly one of: ground_truth, harmful, mediocre."
    )


def _pair_block(i: int, pair: Dict[str, Any]) -> str:
    block = f"### Pair {i}\nQuery: {pair['query']}\n"
    if pair.get("reference_answer"):
        block += f"Correct answer (reference): {pair['reference_answer']}\n"
    return block + f"Document: {pair.get('text') or pair.get('document', '')}\n"


def _ask(client, model: str, system_prompt: str, batch: List[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
    """One request; returns {pair index: {label, rationale}} for the valid entries the model gave."""
    user = "\n".join(_pair_block(i + 1, p) for i, p in enumerate(batch))
    response = chat_completion(
        client, model=model, temperature=0.0, max_tokens=80 + 60 * len(batch),
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
    )
    try:
        entries = json.loads(response.choices[0].message.content)["labels"]
    except (json.JSONDecodeError, TypeError, KeyError):
        return {}
    found = {}
    for position, entry in enumerate(entries):
        try:
            index = int(entry.get("pair", position + 1)) - 1
        except (TypeError, ValueError):
            continue
        label = str(entry.get("label", "")).strip().lower()
        label = ALIASES.get(label, label)
        if 0 <= index < len(batch) and label in LABELS and index not in found:
            found[index] = {"label": label, "rationale": entry.get("rationale", "")}
    return found


def label_pairs(
    pairs: List[Dict[str, Any]],
    client: Any = None,
    model: Optional[str] = None,
    rubric_path: str = RUBRIC_PATH,
    batch_size: int = 10,
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    """Labels each (query, document) pair GD / MD / HD against the query's reference answer."""
    if client is None:
        client, model = make_labeler_client(model)
    model = model or os.getenv("SAFEMED_LABELER_MODEL") or "gpt-4o-mini"
    system_prompt = _system_prompt(rubric_path)

    labelled = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        results: Dict[int, Dict[str, str]] = {}
        missing = list(range(len(batch)))
        for _ in range(1 + max_retries):
            subset = [batch[i] for i in missing]
            for sub_index, result in _ask(client, model, system_prompt, subset).items():
                results[missing[sub_index]] = result
            missing = [i for i in range(len(batch)) if i not in results]
            if not missing:
                break
        if missing:
            ids = [batch[i].get("doc_id") for i in missing]
            raise LabelerUnavailableError(f"labelling model gave no valid label for {ids} after {1 + max_retries} tries")
        for i, pair in enumerate(batch):
            labelled.append({**pair, "llm_label": results[i]["label"], "rationale": results[i]["rationale"]})
    return labelled
