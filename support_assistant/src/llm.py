"""MOCK_LLM toggle: every LLM call in this module goes through here so the
graph nodes stay simple. MOCK_LLM unset or "1" (the default) never
imports/calls a real LLM provider -- no network, no API key. MOCK_LLM=0 is
an optional extension calling Groq's free tier.
"""
from __future__ import annotations

import json
import os

from pydantic import BaseModel, Field, ValidationError


def mock_llm_enabled() -> bool:
    return os.environ.get("MOCK_LLM", "1") != "0"


KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]


def classify_intent_mock(query: str) -> str:
    """Keyword heuristic, no LLM call."""
    lowered = query.lower()
    return "policy_question" if any(kw in lowered for kw in KEYWORDS) else "general_question"


def classify_intent_llm(query: str) -> str:
    """Optional MOCK_LLM=0 extension: ask the LLM to classify instead."""
    prompt = (
        "Classify the following customer question as exactly one word, either "
        "'policy_question' (about Zepto's delivery/returns/membership/tracking/"
        "cancellation/gift cards/support hours policies) or 'general_question' "
        f"(anything else). Question: {query!r}\nAnswer with exactly one word:"
    )
    response = _call_groq(prompt, max_tokens=5).strip().lower()
    return "policy_question" if "policy" in response else "general_question"


def generate_policy_answer_mock(top_chunk_text: str) -> str:
    """Canned template, no LLM call."""
    snippet = top_chunk_text[:200]
    return f"Based on the retrieved context: {snippet}"


class _LLMStructuredAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)


_JSON_OUTPUT_INSTRUCTION = (
    "\n\nRespond with ONLY valid JSON matching exactly this schema, no other "
    'text: {"answer": "<your answer as a string>", "confidence": <float between 0 and 1>}'
)


def generate_structured_policy_answer_llm(prompt: str, max_retries: int = 2) -> dict:
    """Optional MOCK_LLM=0 extension: ask the LLM to answer (grounded in the
    retrieved context via the structured prompt template from src/prompts.py)
    as JSON matching {"answer": str, "confidence": float 0-1}. If the raw
    output fails to parse/validate, retries up to `max_retries` additional
    times with a corrective instruction appended before giving up and
    returning a clearly marked error response. `sources` is filled in by the
    caller from the actual retrieved chunk ids, never by the LLM, so it
    can't be hallucinated.
    """
    current_prompt = prompt + _JSON_OUTPUT_INSTRUCTION
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        raw = _call_groq(current_prompt, max_tokens=250)
        try:
            data = json.loads(raw)
            parsed = _LLMStructuredAnswer.model_validate(data)
            return {"answer": parsed.answer, "confidence": parsed.confidence, "error": None}
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            current_prompt = (
                prompt
                + _JSON_OUTPUT_INSTRUCTION
                + f"\n\nYour previous response was invalid ({e}). Try again -- respond with ONLY the JSON object, nothing else."
            )
    return {
        "answer": f"Error: could not generate a valid response after {max_retries + 1} attempts.",
        "confidence": 0.0,
        "error": str(last_error),
    }


def generate_direct_answer_mock() -> str:
    """Fixed canned string, no LLM call."""
    return "I can only answer questions about Zepto policies right now."


def generate_direct_answer_llm(query: str) -> str:
    """Optional MOCK_LLM=0 extension: ask the LLM directly, no retrieval."""
    prompt = f"You are Zepto's customer support assistant. Answer briefly: {query}"
    return _call_groq(prompt, max_tokens=150).strip()


def _call_groq(prompt: str, max_tokens: int = 200) -> str:
    """Only reached when MOCK_LLM=0 -- imports the groq client lazily so the
    required mock path never needs the package installed or an API key."""
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return completion.choices[0].message.content
