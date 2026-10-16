"""LLM labels (GD / MD / HD) for (query, document) pairs, with a quote check for GD."""

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.stage4_generator.llm_client import chat_completion, make_client

load_dotenv()

LABELS = {"ground_truth", "harmful", "mediocre"}
ALIASES = {"gd": "ground_truth", "hd": "harmful", "md": "mediocre", "ground truth": "ground_truth"}
RUBRIC_PATH = os.path.join(os.path.dirname(__file__), "rubric_v3.md")


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
        "with the query. For ground_truth, 'evidence' must be one full sentence copied exactly from the DOCUMENT "
        "(not from the reference answer, no '...'); for the other labels, leave 'evidence' empty. Reply in JSON only: "
        '{"labels": [{"pair": <number>, "label": "<category>", "evidence": "<sentence or empty>", '
        '"rationale": "<at most 15 words>"}]} '
        "with one entry per pair, in order. <category> must be exactly one of: ground_truth, harmful, mediocre."
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().strip('"\'').rstrip(".").lower()


def evidence_in_document(evidence: str, document: str) -> bool:
    """True when the quoted sentence really appears in the document (ignoring spacing, case and a final period)."""
    quote = _normalize(evidence)
    return len(quote) >= 20 and quote in _normalize(document)


def _pair_block(i: int, pair: Dict[str, Any]) -> str:
    block = f"### Pair {i}\nQuery: {pair['query']}\n"
    if pair.get("reference_answer"):
        block += f"Correct answer (reference): {pair['reference_answer']}\n"
    return block + f"Document: {pair.get('text') or pair.get('document', '')}\n"


def _ask(client, model: str, system_prompt: str, batch: List[Dict[str, Any]],
         last_bad: Optional[Dict[int, Dict[str, Any]]] = None) -> Dict[int, Dict[str, Any]]:
    """One request; returns {pair index: {label, rationale, evidence, evidence_found}} for the valid entries.
    A ground_truth label whose evidence is not found in the document is left out (and kept in last_bad)."""
    last_bad = {} if last_bad is None else last_bad
    user = "\n".join(_pair_block(i + 1, p) for i, p in enumerate(batch))
    response = chat_completion(
        client, model=model, temperature=0.0, max_tokens=80 + 140 * len(batch),
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
        if not (0 <= index < len(batch) and label in LABELS and index not in found):
            continue
        evidence = str(entry.get("evidence") or "")
        document = batch[index].get("text") or batch[index].get("document", "")
        result = {"label": label, "rationale": entry.get("rationale", ""), "evidence": evidence,
                  "evidence_found": evidence_in_document(evidence, document) if label == "ground_truth" else None}
        if result["evidence_found"] is False:
            last_bad[index] = result  # a GD without a real quote from the document is asked again
            continue
        found[index] = result
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
        results: Dict[int, Dict[str, Any]] = {}
        unquoted: Dict[int, Dict[str, Any]] = {}
        missing = list(range(len(batch)))
        for _ in range(1 + max_retries):
            subset = [batch[i] for i in missing]
            bad: Dict[int, Dict[str, Any]] = {}
            for sub_index, result in _ask(client, model, system_prompt, subset, bad).items():
                results[missing[sub_index]] = result
            unquoted.update({missing[k]: v for k, v in bad.items()})
            missing = [i for i in range(len(batch)) if i not in results]
            if not missing:
                break
        for i in [i for i in missing if i in unquoted]:
            results[i] = unquoted[i]  # still GD without a real quote after all tries: kept, flagged for human review
        missing = [i for i in missing if i not in results]
        if missing:
            ids = [batch[i].get("doc_id") for i in missing]
            raise LabelerUnavailableError(f"labelling model gave no valid label for {ids} after {1 + max_retries} tries")
        for i, pair in enumerate(batch):
            labelled.append({**pair, "llm_label": results[i]["label"], "rationale": results[i]["rationale"],
                             "evidence": results[i]["evidence"], "evidence_found": results[i]["evidence_found"]})
    return labelled