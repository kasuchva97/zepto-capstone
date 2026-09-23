from src.graph import run_support_graph


class TestGraphRouting:
    def test_policy_question_routes_to_retrieve_and_answer(self, compiled_graph):
        result = run_support_graph(compiled_graph, "How much does delivery cost?")
        assert result["intent"] == "policy_question"
        assert result["answer"].startswith("Based on the retrieved context: ")
        assert len(result["sources"]) == 3
        assert result["confidence"] == 1.0

    def test_general_question_routes_to_direct_answer(self, compiled_graph):
        result = run_support_graph(compiled_graph, "What's the capital of France?")
        assert result["intent"] == "general_question"
        assert result["answer"] == "I can only answer questions about Zepto policies right now."
        assert result["sources"] == []
        assert result["confidence"] == 1.0

    def test_retrieved_sources_are_relevant_to_the_query(self, compiled_graph):
        result = run_support_graph(compiled_graph, "What is the gift card cancellation refund policy?")
        assert result["intent"] == "policy_question"
        assert "doc_07" in result["sources"]  # Gift Cards doc must be among top-3

    def test_no_network_call_needed_in_mock_mode(self, compiled_graph, monkeypatch):
        """Blocking the real-LLM call path entirely and confirming the graph
        still runs end to end proves mock mode never reaches it."""
        import src.llm as llm_module

        def _boom(*args, **kwargs):
            raise AssertionError("_call_groq should never be called in MOCK_LLM mode")

        monkeypatch.setattr(llm_module, "_call_groq", _boom)
        result = run_support_graph(compiled_graph, "Can I cancel my order?")
        assert result["answer"]  # completed without touching _call_groq
