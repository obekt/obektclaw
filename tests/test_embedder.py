"""Tests for the embedding module.

Tests cover:
- Single text embedding
- Batch text embedding
- Embedding dimension verification
- Edge cases: empty text, very long text, unicode
- Model lazy-loading behavior
"""

import pytest

from obektclaw.memory.embedder import (
    embed,
    embed_batch,
    get_embedding_dimension,
    _get_model,
)


# ============== Dimension Tests ==============


class TestEmbeddingDimension:
    """Tests for embedding dimension."""

    def test_get_embedding_dimension(self):
        """Test that dimension is correct for all-MiniLM-L6-v2."""
        dim = get_embedding_dimension()
        assert dim == 384

    def test_dimension_is_integer(self):
        """Test dimension is integer."""
        dim = get_embedding_dimension()
        assert isinstance(dim, int)

    def test_dimension_is_positive(self):
        """Test dimension is positive."""
        dim = get_embedding_dimension()
        assert dim > 0


# ============== Single Embedding Tests ==============


class TestSingleEmbedding:
    """Tests for single text embedding."""

    def test_embed_returns_list(self):
        """Test that embed returns a list."""
        embedding = embed("test text")
        assert isinstance(embedding, list)

    def test_embed_correct_length(self):
        """Test embedding has correct dimension."""
        embedding = embed("test text")
        assert len(embedding) == get_embedding_dimension()

    def test_embed_values_are_floats(self):
        """Test all values are floats."""
        embedding = embed("test text")
        for val in embedding:
            assert isinstance(val, float)

    def test_embed_values_in_range(self):
        """Test values are in reasonable range."""
        embedding = embed("test text")
        for val in embedding:
            assert -10.0 <= val <= 10.0  # Typical range for normalized embeddings

    def test_embed_different_texts_different_embeddings(self):
        """Test different texts produce different embeddings."""
        emb1 = embed("HTTP client library")
        emb2 = embed("Database connection")

        # Should not be identical
        assert emb1 != emb2

    def test_embed_similar_texts_similar_embeddings(self):
        """Test similar texts produce similar embeddings."""
        emb1 = embed("HTTP client library for Python")
        emb2 = embed("Python HTTP client library")

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a**2 for a in emb1) ** 0.5
        norm2 = sum(a**2 for a in emb2) ** 0.5
        similarity = dot_product / (norm1 * norm2)

        # Similar texts should have high similarity (> 0.7)
        assert similarity > 0.7

    def test_embed_same_text_consistent(self):
        """Test same text produces same embedding."""
        emb1 = embed("same text")
        emb2 = embed("same text")

        # Should be identical
        assert emb1 == emb2


class TestSingleEmbeddingEdgeCases:
    """Tests for edge cases in single embedding."""

    def test_embed_empty_string(self):
        """Test embedding empty string."""
        embedding = embed("")
        assert len(embedding) == get_embedding_dimension()
        # Empty string should still produce a valid embedding

    def test_embed_single_word(self):
        """Test embedding single word."""
        embedding = embed("httpx")
        assert len(embedding) == get_embedding_dimension()

    def test_embed_long_text(self):
        """Test embedding long text."""
        long_text = "This is a very long text that contains many words. " * 100
        embedding = embed(long_text)
        assert len(embedding) == get_embedding_dimension()

    def test_embed_very_long_text(self):
        """Test embedding extremely long text."""
        # Create a text longer than typical model input
        very_long = "word " * 10000  # ~50k characters
        embedding = embed(very_long)
        assert len(embedding) == get_embedding_dimension()

    def test_embed_unicode_text(self):
        """Test embedding unicode text."""
        unicode_text = "日本語のテスト Python テスト"
        embedding = embed(unicode_text)
        assert len(embedding) == get_embedding_dimension()

    def test_embed_mixed_unicode(self):
        """Test embedding mixed language text."""
        mixed = "This is English. 这是中文. これは日本語です."
        embedding = embed(mixed)
        assert len(embedding) == get_embedding_dimension()

    def test_embed_special_chars(self):
        """Test embedding special characters."""
        special = "Code: def foo(): return {'key': 'value'}  # comment"
        embedding = embed(special)
        assert len(embedding) == get_embedding_dimension()

    def test_embed_newlines(self):
        """Test embedding text with newlines."""
        multiline = "Line 1\nLine 2\nLine 3"
        embedding = embed(multiline)
        assert len(embedding) == get_embedding_dimension()

    def test_embed_tabs(self):
        """Test embedding text with tabs."""
        tabbed = "Column1\tColumn2\tColumn3"
        embedding = embed(tabbed)
        assert len(embedding) == get_embedding_dimension()

    def test_embed_numbers(self):
        """Test embedding text with numbers."""
        numeric = "The value is 123.456 and count is 100"
        embedding = embed(numeric)
        assert len(embedding) == get_embedding_dimension()

    def test_embed_code_snippet(self):
        """Test embedding code snippet."""
        code = """
def hello():
    print("Hello, world!")
    return 42
"""
        embedding = embed(code)
        assert len(embedding) == get_embedding_dimension()

    def test_embed_whitespace_only(self):
        """Test embedding whitespace only."""
        whitespace = "   \t\n  "
        embedding = embed(whitespace)
        assert len(embedding) == get_embedding_dimension()


# ============== Batch Embedding Tests ==============


class TestBatchEmbedding:
    """Tests for batch embedding."""

    def test_embed_batch_returns_list(self):
        """Test batch returns list."""
        embeddings = embed_batch(["text1", "text2"])
        assert isinstance(embeddings, list)

    def test_embed_batch_correct_count(self):
        """Test batch has correct number of embeddings."""
        texts = ["one", "two", "three"]
        embeddings = embed_batch(texts)
        assert len(embeddings) == 3

    def test_embed_batch_each_correct_dimension(self):
        """Test each embedding has correct dimension."""
        texts = ["text1", "text2", "text3"]
        embeddings = embed_batch(texts)
        for emb in embeddings:
            assert len(emb) == get_embedding_dimension()

    def test_embed_batch_values_are_floats(self):
        """Test all values in batch are floats."""
        embeddings = embed_batch(["a", "b"])
        for emb in embeddings:
            for val in emb:
                assert isinstance(val, float)

    def test_embed_batch_single_item(self):
        """Test batch with single item."""
        embeddings = embed_batch(["single"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == get_embedding_dimension()

    def test_embed_batch_many_items(self):
        """Test batch with many items."""
        texts = [f"text_{i}" for i in range(50)]
        embeddings = embed_batch(texts)
        assert len(embeddings) == 50

    def test_embed_batch_consistent_with_single(self):
        """Test batch embedding matches single embedding."""
        texts = ["same text"]
        batch_emb = embed_batch(texts)[0]
        single_emb = embed("same text")

        # Should be identical
        assert batch_emb == single_emb

    def test_embed_batch_different_texts(self):
        """Test batch with different texts."""
        texts = ["HTTP client", "Database", "Queue system"]
        embeddings = embed_batch(texts)

        # Each should be different
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]


class TestBatchEmbeddingEdgeCases:
    """Tests for edge cases in batch embedding."""

    def test_embed_batch_empty_list(self):
        """Test batch with empty list."""
        embeddings = embed_batch([])
        assert len(embeddings) == 0

    def test_embed_batch_with_empty_string(self):
        """Test batch containing empty string."""
        embeddings = embed_batch(["real text", "", "more text"])
        assert len(embeddings) == 3
        assert len(embeddings[1]) == get_embedding_dimension()

    def test_embed_batch_all_empty(self):
        """Test batch with all empty strings."""
        embeddings = embed_batch(["", "", ""])
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == get_embedding_dimension()

    def test_embed_batch_with_duplicates(self):
        """Test batch with duplicate texts."""
        embeddings = embed_batch(["same", "same", "different"])

        # Same texts should produce same embeddings
        assert embeddings[0] == embeddings[1]
        assert embeddings[0] != embeddings[2]

    def test_embed_batch_mixed_lengths(self):
        """Test batch with mixed text lengths."""
        texts = [
            "short",
            "medium length text here",
            "very very long text that contains many many words and phrases",
        ]
        embeddings = embed_batch(texts)
        assert len(embeddings) == 3

    def test_embed_batch_unicode(self):
        """Test batch with unicode texts."""
        texts = ["English", "日本語", "中文"]
        embeddings = embed_batch(texts)
        assert len(embeddings) == 3

    def test_embed_batch_with_newlines(self):
        """Test batch with newlines."""
        texts = ["Line 1\nLine 2", "Single line"]
        embeddings = embed_batch(texts)
        assert len(embeddings) == 2


# ============== Model Loading Tests ==============


class TestModelLoading:
    """Tests for model lazy loading."""

    def test_get_model_returns_object(self):
        """Test _get_model returns model."""
        model = _get_model()
        assert model is not None

    def test_get_model_same_instance(self):
        """Test model is cached (singleton)."""
        model1 = _get_model()
        model2 = _get_model()

        # Should be same instance
        assert model1 is model2

    def test_embed_after_model_load(self):
        """Test embed works after explicit model load."""
        _get_model()  # Load model first
        embedding = embed("test")
        assert len(embedding) == get_embedding_dimension()


# ============== Performance Tests ==============


class TestEmbeddingPerformance:
    """Tests for embedding performance characteristics."""

    def test_embed_speed(self):
        """Test single embedding is reasonably fast."""
        import time

        start = time.time()
        embed("test text for speed")
        elapsed = time.time() - start

        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0

    def test_batch_speed_vs_single(self):
        """Test batch is faster than sequential singles."""
        import time

        texts = [f"text {i}" for i in range(10)]

        # Time batch
        start_batch = time.time()
        embed_batch(texts)
        batch_time = time.time() - start_batch

        # Time sequential singles (rough estimate)
        start_single = time.time()
        for text in texts[:3]:  # Only test 3 to save time
            embed(text)
        single_time = (time.time() - start_single) * (10 / 3)  # Extrapolate

        # Batch should be faster or similar (may be similar for small batches)
        # This is a soft assertion - batch optimization varies by model
        assert batch_time < single_time * 2  # At most 2x slower (due to overhead)


# ============== Integration Tests ==============


class TestEmbeddingIntegration:
    """Integration tests for embedding usage."""

    def test_embedding_for_fact_search(self):
        """Test embedding suitable for fact search."""
        fact = "User prefers httpx over requests for HTTP requests"
        embedding = embed(fact)

        # Should produce meaningful embedding
        assert len(embedding) == get_embedding_dimension()
        # Check not all zeros
        assert any(v != 0 for v in embedding)

    def test_embedding_for_skill_search(self):
        """Test embedding suitable for skill search."""
        skill_desc = "Import CSV files into database using pandas"
        embedding = embed(skill_desc)

        assert len(embedding) == get_embedding_dimension()
        assert any(v != 0 for v in embedding)

    def test_embedding_semantic_similarity(self):
        """Test semantic similarity works."""
        # Two semantically similar queries
        q1 = "How to make HTTP requests in Python"
        q2 = "Python HTTP client usage"

        emb1 = embed(q1)
        emb2 = embed(q2)

        # Calculate similarity
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a**2 for a in emb1) ** 0.5
        norm2 = sum(a**2 for a in emb2) ** 0.5
        similarity = dot / (norm1 * norm2)

        # Should be similar (> 0.5)
        assert similarity > 0.5

    def test_embedding_different_semantic(self):
        """Test different semantics produce different embeddings."""
        # Two semantically different queries
        q1 = "HTTP client library"
        q2 = "Database connection pooling"

        emb1 = embed(q1)
        emb2 = embed(q2)

        # Calculate similarity
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a**2 for a in emb1) ** 0.5
        norm2 = sum(a**2 for a in emb2) ** 0.5
        similarity = dot / (norm1 * norm2)

        # Should be less similar (< 0.7)
        assert similarity < 0.7
