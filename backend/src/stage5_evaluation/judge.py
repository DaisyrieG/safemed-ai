"""Stage 5: hallucination judge (Chapter 3, Hallucination Scoring Protocol, parts 1 to 3)."""

import json
import os
import re
from typing import Any, Dict, List, Optional

from src.stage4_generator.llm_client import chat_completion, make_client

DEFAULT_JUDGE_MODEL = "gpt-4o"
LABELS = ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED")

VERDICT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "claim_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": list(LABELS)},
                "reasoning": {"type": "string"},
            },
            "required": ["status", "reasoning"],
            "additionalProperties": False,
        },
    },
}


def _parse_verdict(content: str) -> Dict[str, str]:
    """Parses the judge output; tolerates text around the JSON object."""
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", content or "", re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        m = re.search(r"\b(SUPPORTED|CONTRADICTED|UNSUPPORTED)\b", (content or "").upper())
        return {"status": m.group(1) if m else "UNSUPPORTED", "reasoning": content or ""}


class ClaimJudge:
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, client: Any = None):
        self.model = model or os.getenv("SAFEMED_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL
        self.base_url = base_url or os.getenv("SAFEMED_JUDGE_BASE_URL") or None
        key = (api_key or os.getenv("SAFEMED_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
               or ("local" if self.base_url else None))
        self.client, self.init_error = (client, None) if client is not None else make_client(key, self.base_url)
        self._use_schema = True

    def _ask(self, prompt: str, schema: Optional[Dict] = None) -> str:
        kwargs = dict(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        if schema and self._use_schema:
            try:
                return chat_completion(self.client, response_format=schema, **kwargs).choices[0].message.content
            except Exception as e:
                if "response_format" not in str(e) and "json_schema" not in str(e):
                    raise
                self._use_schema = False
        return chat_completion(self.client, **kwargs).choices[0].message.content

    def extract_claims(self, answer: str) -> List[str]:
        """Atomic factual claims of the answer; the decision line, hedges and statements about the documents are excluded."""
        prompt = (
            "Break the answer below into atomic factual claims, one per line. Exclude the line that starts "
            "with 'Decision:', hedging phrases, and statements about the provided documents themselves "
            "(e.g. 'the documents do not say').\n\n"
            f"Answer:\n{answer}\n\nClaims:"
        )
        lines = (self._ask(prompt) or "").split("\n")
        claims = [re.sub(r"^\s*[-*•\d.)]+\s*", "", line).strip() for line in lines]
        return [c for c in claims if c and not c.lower().startswith("decision:")]

    def verify_claims(self, claims: List[str], reference: str) -> List[Dict[str, str]]:
        """Labels each claim SUPPORTED / UNSUPPORTED / CONTRADICTED against the reference evidence."""
        results = []
        for claim in claims:
            prompt = (
                "You are a strict biomedical NLI (Natural Language Inference) verifier. Given the reference "
                "evidence, label the claim SUPPORTED (entailed by the evidence), CONTRADICTED (conflicts with "
                "the evidence) or UNSUPPORTED (neither).\n\n"
                f"Reference evidence:\n{reference}\n\nClaim:\n{claim}\n\n"
                'Output JSON: {"status": "SUPPORTED"|"CONTRADICTED"|"UNSUPPORTED", "reasoning": "..."}'
            )
            data = _parse_verdict(self._ask(prompt, VERDICT_SCHEMA))
            status = str(data.get("status", "")).upper()
            results.append({"claim": claim,
                            "status": status if status in LABELS else "UNSUPPORTED",
                            "reasoning": data.get("reasoning", "")})
        return results
