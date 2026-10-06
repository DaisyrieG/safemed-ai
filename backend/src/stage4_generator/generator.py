"""Stage 4: Answer generator."""

import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.stage4_generator.llm_client import chat_completion, make_client

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are SafeMed AI, a clinical search assistant. Synthesize a concise, "
    "evidence-based Clinical Response Summary to the user's query.\n"
    "RULES:\n"
    "1. You MUST begin your response with exactly 'Decision: yes', 'Decision: no', or 'Decision: maybe'.\n"
    "2. Your explanation must not exceed 150 words.\n"
    "3. Strictly ground your answer ONLY in the provided documents. Do NOT use internal memory.\n"
    "4. Explicitly state if the provided documents are insufficient to answer the query.\n"
    "5. Cite sources [1], [2], etc., where applicable."
)


class GeneratorUnavailableError(RuntimeError):
    """Raised when the configured LLM cannot be called."""


class AnswerGenerator:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Any = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or os.getenv("SAFEMED_LLM_MODEL") or DEFAULT_MODEL
        self.base_url = base_url or os.getenv("SAFEMED_LLM_BASE_URL") or None
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.base_url and not self.api_key:
            self.api_key = "local"
        self.client = client
        self.init_error: Optional[str] = None
        if self.client is None:
            self.client, self.init_error = make_client(self.api_key, self.base_url)

        print(
            f"[Stage 4] LLM: {self.model} @ {self.base_url or 'OpenAI API'}"
            f" (client {'ready' if self.client is not None else 'NOT ready: ' + str(self.init_error)})"
        )

    def _complete(self, **kwargs) -> str:
        if self.client is None:
            raise GeneratorUnavailableError(
                f"LLM '{self.model}' unavailable ({self.init_error}). Set OPENAI_API_KEY, or point "
                "SAFEMED_LLM_BASE_URL at a local OpenAI-compatible server."
            )
        try:
            response = chat_completion(self.client, model=self.model, temperature=0.0, **kwargs)
        except Exception as e:
            raise GeneratorUnavailableError(f"LLM call to '{self.model}' failed: {e}") from e
        return (response.choices[0].message.content or "").strip()

    def generate_answer(self, query: str, top_documents: List[Dict[str, Any]]) -> str:
        """Answers the query from the top-5 documents only (temperature 0, max 200 tokens)."""
        if not top_documents:
            return "No supporting clinical literature found for this query in the verified database."

        context = "\n\n".join(
            f"Source [{i + 1}] ({doc.get('title', 'Unknown')}):\n{doc.get('text', '')}"
            for i, doc in enumerate(top_documents[:5])
        )
        user_content = f"Question: {query}\n\nRetrieved Documents:\n{context}\n\nClinical Response Summary:"
        return self._complete(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            max_tokens=200,
        )

    def decompose_into_claims(self, answer: str) -> List[str]:
        """Splits the answer into atomic factual statements for display in the web app."""
        prompt = (
            "Break down the following clinical summary into distinct, atomic factual statements (one per line). "
            "Each statement must be a single verifiable clinical fact without conjunctions:\n\n"
            f"{answer}\n\nAtomic Claims:"
        )
        raw = self._complete(messages=[{"role": "user", "content": prompt}])
        claims = [re.sub(r"^\s*[-*•\d.]+\s*", "", line).strip() for line in raw.split("\n")]
        return [c for c in claims if c and not c.lower().startswith("decision:")]
