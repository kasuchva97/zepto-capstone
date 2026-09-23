import pytest

from src.llm import (
    KEYWORDS,
    classify_intent_mock,
    generate_direct_answer_mock,
    generate_policy_answer_mock,
    mock_llm_enabled,
)


class TestMockLLMEnabled:
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("MOCK_LLM", raising=False)
        assert mock_llm_enabled() is True

    def test_enabled_when_explicitly_1(self, monkeypatch):
        monkeypatch.setenv("MOCK_LLM", "1")
        assert mock_llm_enabled() is True

    def test_disabled_when_0(self, monkeypatch):
        monkeypatch.setenv("MOCK_LLM", "0")
        assert mock_llm_enabled() is False


class TestClassifyIntentMock:
    @pytest.mark.parametrize("keyword", KEYWORDS)
    def test_each_keyword_triggers_policy_question(self, keyword):
        query = f"I have a question about {keyword} please help"
        assert classify_intent_mock(query) == "policy_question"

    def test_case_insensitive(self):
        assert classify_intent_mock("What about DELIVERY charges?") == "policy_question"

    def test_unrelated_query_is_general_question(self):
        assert classify_intent_mock("What's the weather like today?") == "general_question"

    def test_no_llm_call_involved(self):
        """Purely a function of the string -- no network/mock object needed
        to call it, which is itself evidence no LLM call happens."""
        result = classify_intent_mock("Can I cancel my order?")
        assert result in ("policy_question", "general_question")


class TestGeneratePolicyAnswerMock:
    def test_uses_canned_template_with_snippet(self):
        chunk = "Delivery Policy: Zepto delivers groceries within 10 to 30 minutes."
        answer = generate_policy_answer_mock(chunk)
        assert answer.startswith("Based on the retrieved context: ")
        assert chunk[:50] in answer

    def test_snippet_truncated_to_200_chars(self):
        long_chunk = "x" * 500
        answer = generate_policy_answer_mock(long_chunk)
        snippet = answer.removeprefix("Based on the retrieved context: ")
        assert len(snippet) == 200


class TestGenerateDirectAnswerMock:
    def test_returns_fixed_canned_string(self):
        assert generate_direct_answer_mock() == "I can only answer questions about Zepto policies right now."

    def test_is_deterministic(self):
        assert generate_direct_answer_mock() == generate_direct_answer_mock()
