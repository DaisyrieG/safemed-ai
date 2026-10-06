"""OpenAI-compatible chat client shared by the generator (Stage 4), the judge (Stage 5)"""

import random
import time
from typing import Any, Optional, Tuple

try:
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
except ImportError:
    RETRYABLE_ERRORS = ()


def make_client(api_key: Optional[str], base_url: Optional[str] = None) -> Tuple[Any, Optional[str]]:
    """Returns (client, None), or (None, reason) when no client can be built."""
    if not api_key:
        return None, "OPENAI_API_KEY is not set"
    try:
        from openai import OpenAI
        return (OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)), None
    except Exception as e:
        return None, f"could not create OpenAI client: {e}"


def chat_completion(client, max_attempts: int = 6, max_wait: float = 60.0, **kwargs) -> Any:
    """Chat completion with retries on rate-limit, connection and server errors."""
    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except RETRYABLE_ERRORS:
            if attempt == max_attempts - 1:
                raise
            time.sleep(random.uniform(1.0, min(max_wait, 2 ** (attempt + 1))))
