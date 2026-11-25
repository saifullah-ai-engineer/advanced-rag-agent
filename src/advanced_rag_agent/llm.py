"""Optional LLM generation through OpenRouter free models.

The LLM is strictly optional. When no OPENROUTER_API_KEY is set, get_llm
returns None and the agent answers with the extractive fallback instead.
Only free models are ever used: the default model is a verified free
model, and any model name without ":free" is refused.
"""

import json
import urllib.error
import urllib.request

from advanced_rag_agent.config import RagConfig


class OpenRouterLLM:
    """Minimal OpenRouter chat client using only the standard library."""

    def __init__(self, api_key: str, model: str, base_url: str):
        if ":free" not in model:
            raise ValueError(
                f"Refusing non-free model '{model}'. This project only uses "
                f"OpenRouter free models (names ending in :free)."
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"OpenRouter request failed ({exc.code}): {detail}") from exc
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {str(body)[:300]}")
        return str(choices[0]["message"]["content"]).strip()


def get_llm(config: RagConfig):
    """Return an OpenRouter client when a key is set, else None."""
    if not config.openrouter_api_key:
        return None
    return OpenRouterLLM(
        api_key=config.openrouter_api_key,
        model=config.openrouter_model,
        base_url=config.openrouter_base_url,
    )


def build_answer_prompt(question: str, chunks) -> str:
    """Prompt for grounded generation: answer only from the given chunks."""
    context = "\n\n".join(
        f"[{chunk.citation()}]\n{chunk.text}" for chunk in chunks
    )
    return (
        "Answer the question using ONLY the context below. "
        "Every factual claim in the answer must cite its source as "
        "[source:start-end]. If the context does not contain the answer, "
        "say so plainly instead of guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
