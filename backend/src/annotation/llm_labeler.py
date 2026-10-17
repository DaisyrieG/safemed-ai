"""LLM labels (GD / MD / HD) for (query, document) pairs.

The model answers the codebook's questions for one pair at a time (what the query and the document are about, which
of condition / exposure / outcome they share, and 1 to 3 quoted sentences for GD); the code then applies the codebook's
steps in order, so the label always follows the codebook's rules:
  Step 1  1 to 3 sentences all found in the document, on the same exposure and outcome, same direction  -> GD
  Step 2  otherwise, any shared condition, exposure or outcome (or a quote showing another finding) -> HD
  Step 3  otherwise                                                                                 -> MD
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from dotenv import load_dotenv

from src.stage4_generator.llm_client import chat_completion, make_client

load_dotenv()

LABELS = {"ground_truth", "harmful", "mediocre"}
RUBRIC_PATH = os.path.join(os.path.dirname(__file__), "rubric_v3.md")
SHARED_KEYS = ("shares_condition", "shares_exposure", "shares_outcome")
GD_KEYS = ("evidence_same_exposure", "evidence_same_outcome", "same_direction")
BOOLEAN_KEYS = SHARED_KEYS + GD_KEYS


class LabelerUnavailableError(RuntimeError):
    """Raised when the labelling model cannot be called or returns no valid answer."""


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
        "Answer the codebook's questions for the (query, document) pair. Reply in JSON only:\n"
        '{"q_condition": "...", "q_exposure": "...", "q_outcome": "...", '
        '"d_condition": "...", "d_exposure": "...", "d_outcome": "...", '
        '"evidence": ["<1 to 3 full sentences copied exactly from the DOCUMENT that together report the reference '
        'finding>"] or [], '
        '"evidence_same_exposure": true/false, "evidence_same_outcome": true/false, "same_direction": true/false, '
        '"shares_condition": true/false, "shares_exposure": true/false, "shares_outcome": true/false, '
        '"rationale": "<at most 20 words>"}\n'
        "q_* = the query's main condition, exposure/intervention and outcome; d_* = what the document studies. "
        "'evidence' only if the document reports the same finding as the reference answer for the same exposure and "
        "outcome: a list of 1 to 3 sentences, each copied exactly from the DOCUMENT (not from the reference answer, "
        "no '...'); evidence_same_exposure / evidence_same_outcome = those sentences are about exactly the query's "
        "exposure / outcome (not a related one); 'same_direction' = they point the same way (yes/no) as the reference "
        "answer. "
        "shares_* = the document is about the query's main condition / exposure / outcome (animal, cell or lab studies "
        "and other populations still count)."
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().strip('"\'').rstrip(".").lower()


def evidence_in_document(evidence: Union[str, Sequence[str]], document: str) -> bool:
    """True when 1 to 3 quoted sentences are given and every one really appears in the document (ignoring spacing,
    case and a final period); a single sentence may be passed as a string."""
    quotes = [_normalize(evidence)] if isinstance(evidence, str) else [_normalize(s) for s in evidence]
    text = _normalize(document)
    return 1 <= len(quotes) <= 3 and all(len(q) >= 20 and q in text for q in quotes)


def _user_message(pair: Dict[str, Any]) -> str:
    message = f"Query: {pair['query']}\n"
    if pair.get("reference_answer"):
        message += f"Correct answer (reference): {pair['reference_answer']}\n"
    return message + f"Document: {pair.get('text') or pair.get('document', '')}\n"


def decide_label(answers: Dict[str, Any], document: str) -> Dict[str, Any]:
    """Applies the codebook's three steps to the model's answers; a GD needs a quote that is really in the document."""
    raw = answers.get("evidence")
    sentences = [s for s in ([raw] if isinstance(raw, str) else list(raw or [])) if str(s).strip()]  # list or one
    sentences = [str(s).strip() for s in sentences]
    evidence = " ".join(sentences)
    found = evidence_in_document(sentences, document) if sentences else None
    if found and all(answers.get(k) is True for k in GD_KEYS):
        label = "ground_truth"
    elif any(answers.get(k) is True for k in SHARED_KEYS) or found:
        label = "harmful"  # same topic without the reference finding, or a real quote showing another finding
    else:
        label = "mediocre"
    return {"label": label, "evidence": evidence, "evidence_found": found}


def _ask(client, model: str, system_prompt: str, pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One request for one pair; returns the model's answers, or None when the reply is not usable."""
    response = chat_completion(
        client, model=model, temperature=0.0, max_tokens=500,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": _user_message(pair)}],
    )
    try:
        answers = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    if not isinstance(answers, dict) or not all(isinstance(answers.get(k), bool) for k in BOOLEAN_KEYS):
        return None
    return answers


def label_pairs(
    pairs: List[Dict[str, Any]],
    client: Any = None,
    model: Optional[str] = None,
    rubric_path: str = RUBRIC_PATH,
    batch_size: int = 1,
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    """Labels each (query, document) pair GD / MD / HD against the query's reference answer, one pair per request.
    batch_size is accepted for older callers and ignored (one pair per request labelled best in the pilot)."""
    if client is None:
        client, model = make_labeler_client(model)
    model = model or os.getenv("SAFEMED_LABELER_MODEL") or "gpt-4o-mini"
    system_prompt = _system_prompt(rubric_path)

    labelled = []
    for pair in pairs:
        answers = None
        for _ in range(1 + max_retries):
            answers = _ask(client, model, system_prompt, pair)
            if answers is not None:
                break
        if answers is None:
            raise LabelerUnavailableError(
                f"labelling model gave no valid answer for {pair.get('doc_id')} after {1 + max_retries} tries")
        decision = decide_label(answers, pair.get("text") or pair.get("document", ""))
        labelled.append({**pair, "llm_label": decision["label"], "rationale": str(answers.get("rationale", "")),
                         "evidence": decision["evidence"], "evidence_found": decision["evidence_found"],
                         "label_checks": {k: answers.get(k) for k in BOOLEAN_KEYS}})
    return labelled