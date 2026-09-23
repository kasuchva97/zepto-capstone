"""Tests the retry-on-validation-failure logic for the optional MOCK_LLM=0
path (Task 4's requirement) without needing a real Groq API key -- by
monkeypatching the low-level _call_groq function."""
import src.llm as llm_module
from src.llm import generate_structured_policy_answer_llm


class TestRetryOnInvalidOutput:
    def test_succeeds_immediately_on_valid_json(self, monkeypatch):
        calls = []

        def fake_call_groq(prompt, max_tokens=250):
            calls.append(prompt)
            return '{"answer": "Delivery takes 10-30 minutes.", "confidence": 0.9}'

        monkeypatch.setattr(llm_module, "_call_groq", fake_call_groq)
        result = generate_structured_policy_answer_llm("some prompt")

        assert result["answer"] == "Delivery takes 10-30 minutes."
        assert result["confidence"] == 0.9
        assert result["error"] is None
        assert len(calls) == 1  # no retry needed

    def test_retries_once_then_succeeds(self, monkeypatch):
        responses = iter(
            [
                "not valid json at all",
                '{"answer": "Refunds take 3-5 business days.", "confidence": 0.85}',
            ]
        )
        calls = []

        def fake_call_groq(prompt, max_tokens=250):
            calls.append(prompt)
            return next(responses)

        monkeypatch.setattr(llm_module, "_call_groq", fake_call_groq)
        result = generate_structured_policy_answer_llm("some prompt", max_retries=2)

        assert result["answer"] == "Refunds take 3-5 business days."
        assert len(calls) == 2
        assert "invalid" in calls[1].lower()  # corrective instruction was appended

    def test_gives_up_after_max_retries_returns_marked_error(self, monkeypatch):
        def always_invalid(prompt, max_tokens=250):
            return "still not json"

        monkeypatch.setattr(llm_module, "_call_groq", always_invalid)
        result = generate_structured_policy_answer_llm("some prompt", max_retries=2)

        assert result["confidence"] == 0.0
        assert "could not generate a valid response" in result["answer"].lower()
        assert result["error"] is not None

    def test_schema_violation_triggers_retry_not_just_malformed_json(self, monkeypatch):
        """Valid JSON but violating the Pydantic schema (confidence out of
        range) must also be treated as a failure and retried."""
        responses = iter(
            [
                '{"answer": "ok", "confidence": 5.0}',  # valid JSON, invalid per schema (>1.0)
                '{"answer": "Fixed answer.", "confidence": 0.7}',
            ]
        )

        def fake_call_groq(prompt, max_tokens=250):
            return next(responses)

        monkeypatch.setattr(llm_module, "_call_groq", fake_call_groq)
        result = generate_structured_policy_answer_llm("some prompt", max_retries=2)

        assert result["answer"] == "Fixed answer."
        assert result["confidence"] == 0.7
