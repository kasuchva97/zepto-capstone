from src.ingest import load_documents, retrieve_top_k


class TestLoadDocuments:
    def test_loads_all_8_documents(self):
        chunks = load_documents()
        assert len(chunks) == 8
        assert [c["id"] for c in chunks] == [f"doc_{i:02d}" for i in range(1, 9)]

    def test_each_chunk_has_nonempty_text(self):
        for c in load_documents():
            assert len(c["text"]) > 20

    def test_source_filenames_match_ids(self):
        for c in load_documents():
            assert c["source"] == f"{c['id']}.txt"


class TestRetrieveTopK:
    def test_delivery_question_retrieves_delivery_doc(self, collection):
        hits = retrieve_top_k(collection, "How much does delivery cost and how fast is it?", k=3)
        assert len(hits) == 3
        assert hits[0]["id"] == "doc_01"  # Delivery Policy

    def test_refund_question_retrieves_returns_doc(self, collection):
        hits = retrieve_top_k(collection, "Can I return a damaged grocery item for a refund?", k=3)
        top_ids = [h["id"] for h in hits]
        assert "doc_02" in top_ids or "doc_06" in top_ids  # Returns&Refunds / Damaged-Missing Items

    def test_membership_question_retrieves_membership_doc(self, collection):
        hits = retrieve_top_k(collection, "What are the different Zepto membership plans?", k=3)
        assert hits[0]["id"] == "doc_03"  # Membership Tiers

    def test_gift_card_question_retrieves_gift_card_doc(self, collection):
        hits = retrieve_top_k(collection, "What gift card amounts can I buy?", k=3)
        assert hits[0]["id"] == "doc_07"  # Gift Cards

    def test_results_ordered_by_similarity_descending(self, collection):
        hits = retrieve_top_k(collection, "customer support availability hours", k=5)
        similarities = [h["similarity"] for h in hits]
        assert similarities == sorted(similarities, reverse=True)

    def test_k_controls_result_count(self, collection):
        assert len(retrieve_top_k(collection, "delivery", k=1)) == 1
        assert len(retrieve_top_k(collection, "delivery", k=5)) == 5
