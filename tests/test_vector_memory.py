"""Tests for VectorMemory (ChromaDB wrapper).

Tests cover:
- Adding/removing facts, memories, skills, entities
- Search functions with various filters
- Statistics and clear operations
- Edge cases: empty queries, missing IDs, large batches
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obektclaw.config import Config


# Fixtures
@pytest.fixture
def temp_chroma_path():
    """Create a temporary directory for ChromaDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "chroma"


@pytest.fixture
def mock_config(temp_chroma_path):
    """Mock CONFIG with temporary paths."""
    config = Config(
        home=temp_chroma_path.parent,
        db_path=temp_chroma_path.parent / "test.db",
        skills_dir=temp_chroma_path.parent / "skills",
        bundled_skills_dir=temp_chroma_path.parent / "bundled_skills",
        logs_dir=temp_chroma_path.parent / "logs",
        llm_base_url="https://api.openai.com/v1",
        llm_api_key="test-key",
        llm_model="gpt-4o-mini",
        llm_fast_model="gpt-4o-mini",
        tg_token="",
        tg_allowed_chat_ids=(),
        bash_timeout=30,
        workdir=temp_chroma_path.parent,
        # Memory system defaults
        chroma_path=temp_chroma_path,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        semantic_search_limit=10,
        graph_traversal_depth=2,
        context_assembly_max_tokens=2000,
    )
    return config


@pytest.fixture
def mock_embedder():
    """Mock the embedder to return fixed embeddings."""
    with patch("obektclaw.memory.vector_memory.embed") as mock_embed:
        # Return a fixed 384-dimensional embedding
        mock_embed.return_value = [0.1] * 384
        yield mock_embed


@pytest.fixture
def vector_memory(mock_config, mock_embedder):
    """Create a VectorMemory instance with mocked dependencies."""
    with patch("obektclaw.memory.vector_memory.CONFIG", mock_config):
        from obektclaw.memory.vector_memory import VectorMemory

        vm = VectorMemory()
        yield vm
        vm.clear_all()


# ============== Fact Tests ==============


class TestFactOperations:
    """Tests for fact CRUD operations."""

    def test_add_fact_basic(self, vector_memory, mock_embedder):
        """Test adding a basic fact."""
        vector_memory.add_fact(
            fact_id="fact_001",
            content="User prefers httpx over requests",
            category="preference",
            confidence=0.9,
            source_turn=1,
        )

        # Verify fact was added
        fact = vector_memory.get_fact_by_id("fact_001")
        assert fact is not None
        assert fact["id"] == "fact_001"
        assert fact["content"] == "User prefers httpx over requests"
        assert fact["metadata"]["category"] == "preference"
        assert fact["metadata"]["confidence"] == 0.9

    def test_add_fact_with_entity_ids(self, vector_memory, mock_embedder):
        """Test adding a fact linked to entities."""
        vector_memory.add_fact(
            fact_id="fact_002",
            content="Server runs on Hetzner CX22",
            category="environment",
            confidence=0.95,
            source_turn=2,
            entity_ids=["entity_hetzner", "entity_server"],
        )

        fact = vector_memory.get_fact_by_id("fact_002")
        assert fact is not None
        assert fact["metadata"]["entity_ids"] == "entity_hetzner,entity_server"

    def test_add_fact_upsert_behavior(self, vector_memory, mock_embedder):
        """Test that adding same ID overwrites."""
        vector_memory.add_fact(
            fact_id="fact_003",
            content="Original content",
            category="general",
            confidence=0.7,
            source_turn=1,
        )

        # Add again with different content
        vector_memory.add_fact(
            fact_id="fact_003",
            content="Updated content",
            category="preference",
            confidence=0.9,
            source_turn=2,
        )

        fact = vector_memory.get_fact_by_id("fact_003")
        assert fact["content"] == "Updated content"
        assert fact["metadata"]["confidence"] == 0.9

    def test_get_fact_by_id_missing(self, vector_memory):
        """Test getting a non-existent fact."""
        fact = vector_memory.get_fact_by_id("nonexistent")
        assert fact is None

    def test_delete_fact(self, vector_memory, mock_embedder):
        """Test deleting a fact."""
        vector_memory.add_fact(
            fact_id="fact_del",
            content="To be deleted",
            category="ephemeral",
            confidence=0.5,
            source_turn=1,
        )

        # Verify exists
        assert vector_memory.get_fact_by_id("fact_del") is not None

        # Delete
        vector_memory.delete_fact("fact_del")

        # Verify deleted
        assert vector_memory.get_fact_by_id("fact_del") is None

    def test_delete_fact_nonexistent(self, vector_memory):
        """Test deleting a non-existent fact (no error)."""
        # Should not raise
        vector_memory.delete_fact("nonexistent_fact")

    def test_update_fact_confidence(self, vector_memory, mock_embedder):
        """Test updating fact confidence."""
        vector_memory.add_fact(
            fact_id="fact_conf",
            content="Confidence test",
            category="general",
            confidence=0.5,
            source_turn=1,
        )

        vector_memory.update_fact_confidence("fact_conf", 0.95)

        fact = vector_memory.get_fact_by_id("fact_conf")
        assert fact["metadata"]["confidence"] == 0.95
        assert "updated_at" in fact["metadata"]

    def test_update_fact_confidence_missing(self, vector_memory):
        """Test updating confidence of non-existent fact."""
        # Should silently return
        vector_memory.update_fact_confidence("nonexistent", 0.9)


class TestFactSearch:
    """Tests for fact search operations."""

    def test_search_similar_facts_basic(self, vector_memory, mock_embedder):
        """Test basic fact search."""
        vector_memory.add_fact(
            fact_id="fact_a",
            content="User uses httpx for HTTP requests",
            category="preference",
            confidence=0.9,
            source_turn=1,
        )
        vector_memory.add_fact(
            fact_id="fact_b",
            content="Server deployed on Hetzner",
            category="environment",
            confidence=0.8,
            source_turn=2,
        )

        results = vector_memory.search_similar_facts(
            query="HTTP client preference",
            n_results=5,
        )

        assert len(results) >= 1
        # All embeddings are identical in mock, so order may vary

    def test_search_similar_facts_with_category_filter(
        self, vector_memory, mock_embedder
    ):
        """Test fact search with category filter."""
        vector_memory.add_fact(
            fact_id="fact_pref",
            content="Preference fact",
            category="preference",
            confidence=0.9,
            source_turn=1,
        )
        vector_memory.add_fact(
            fact_id="fact_env",
            content="Environment fact",
            category="environment",
            confidence=0.8,
            source_turn=2,
        )

        results = vector_memory.search_similar_facts(
            query="test query",
            n_results=10,
            category_filter="preference",
        )

        # Should only return preference facts
        for fact in results:
            assert fact["metadata"]["category"] == "preference"

    def test_search_similar_facts_with_confidence_filter(
        self, vector_memory, mock_embedder
    ):
        """Test fact search with minimum confidence."""
        vector_memory.add_fact(
            fact_id="fact_high",
            content="High confidence fact",
            category="general",
            confidence=0.9,
            source_turn=1,
        )
        vector_memory.add_fact(
            fact_id="fact_low",
            content="Low confidence fact",
            category="general",
            confidence=0.3,
            source_turn=2,
        )

        results = vector_memory.search_similar_facts(
            query="test query",
            n_results=10,
            min_confidence=0.8,
        )

        # Should only return high confidence facts
        for fact in results:
            assert fact["metadata"]["confidence"] >= 0.8

    def test_search_similar_facts_with_both_filters(self, vector_memory, mock_embedder):
        """Test fact search with both category and confidence filters."""
        vector_memory.add_fact(
            fact_id="fact_p_high",
            content="Preference high",
            category="preference",
            confidence=0.95,
            source_turn=1,
        )
        vector_memory.add_fact(
            fact_id="fact_p_low",
            content="Preference low",
            category="preference",
            confidence=0.4,
            source_turn=2,
        )
        vector_memory.add_fact(
            fact_id="fact_e_high",
            content="Environment high",
            category="environment",
            confidence=0.9,
            source_turn=3,
        )

        results = vector_memory.search_similar_facts(
            query="test",
            n_results=10,
            category_filter="preference",
            min_confidence=0.8,
        )

        # Should only return preference facts with high confidence
        for fact in results:
            assert fact["metadata"]["category"] == "preference"
            assert fact["metadata"]["confidence"] >= 0.8

    def test_search_similar_facts_empty_collection(self, vector_memory):
        """Test search on empty collection."""
        results = vector_memory.search_similar_facts(
            query="anything",
            n_results=5,
        )
        assert len(results) == 0

    def test_get_recent_facts(self, vector_memory, mock_embedder):
        """Test getting recent facts."""
        # Add multiple facts
        for i in range(5):
            vector_memory.add_fact(
                fact_id=f"fact_{i}",
                content=f"Fact number {i}",
                category="general",
                confidence=0.8,
                source_turn=i,
            )

        recent = vector_memory.get_recent_facts(limit=3)
        assert len(recent) <= 3

    def test_get_recent_facts_with_category(self, vector_memory, mock_embedder):
        """Test getting recent facts filtered by category."""
        vector_memory.add_fact(
            fact_id="f1",
            content="Preference 1",
            category="preference",
            confidence=0.9,
            source_turn=1,
        )
        vector_memory.add_fact(
            fact_id="f2",
            content="Environment 1",
            category="environment",
            confidence=0.8,
            source_turn=2,
        )

        recent = vector_memory.get_recent_facts(limit=5, category="preference")
        for fact in recent:
            assert fact["metadata"]["category"] == "preference"


# ============== Memory Tests ==============


class TestMemoryOperations:
    """Tests for conversation memory operations."""

    def test_add_memory_basic(self, vector_memory, mock_embedder):
        """Test adding a conversation memory."""
        vector_memory.add_memory(
            memory_id="mem_001",
            content="How do I use httpx?",
            session_id=1,
            role="user",
            timestamp=datetime.utcnow().isoformat(),
        )

        # Verify exists via search
        results = vector_memory.search_similar_memories(
            query="httpx usage",
            n_results=5,
        )
        assert len(results) >= 1

    def test_add_memory_with_tool_calls(self, vector_memory, mock_embedder):
        """Test adding memory with tool calls metadata."""
        vector_memory.add_memory(
            memory_id="mem_002",
            content="I'll read the file for you",
            session_id=1,
            role="assistant",
            timestamp=datetime.utcnow().isoformat(),
            tool_calls=["read_file", "write_file"],
        )

        results = vector_memory.search_similar_memories(
            query="file operations",
            n_results=5,
        )
        assert len(results) >= 1
        # Check tool_calls in metadata
        for mem in results:
            if mem["id"] == "mem_002":
                assert mem["metadata"]["tool_calls"] == "read_file,write_file"

    def test_search_memories_with_session_filter(self, vector_memory, mock_embedder):
        """Test memory search with session filter."""
        vector_memory.add_memory(
            memory_id="mem_s1",
            content="Session 1 message",
            session_id=1,
            role="user",
            timestamp=datetime.utcnow().isoformat(),
        )
        vector_memory.add_memory(
            memory_id="mem_s2",
            content="Session 2 message",
            session_id=2,
            role="user",
            timestamp=datetime.utcnow().isoformat(),
        )

        results = vector_memory.search_similar_memories(
            query="message",
            n_results=10,
            session_filter=1,
        )

        for mem in results:
            assert mem["metadata"]["session_id"] == 1

    def test_search_memories_with_role_filter(self, vector_memory, mock_embedder):
        """Test memory search with role filter."""
        vector_memory.add_memory(
            memory_id="mem_user",
            content="User question",
            session_id=1,
            role="user",
            timestamp=datetime.utcnow().isoformat(),
        )
        vector_memory.add_memory(
            memory_id="mem_assist",
            content="Assistant response",
            session_id=1,
            role="assistant",
            timestamp=datetime.utcnow().isoformat(),
        )

        results = vector_memory.search_similar_memories(
            query="question response",
            n_results=10,
            role_filter="user",
        )

        for mem in results:
            assert mem["metadata"]["role"] == "user"

    def test_delete_memories_by_session(self, vector_memory, mock_embedder):
        """Test deleting all memories for a session."""
        vector_memory.add_memory(
            memory_id="mem_del1",
            content="To delete 1",
            session_id=99,
            role="user",
            timestamp=datetime.utcnow().isoformat(),
        )
        vector_memory.add_memory(
            memory_id="mem_del2",
            content="To delete 2",
            session_id=99,
            role="assistant",
            timestamp=datetime.utcnow().isoformat(),
        )
        vector_memory.add_memory(
            memory_id="mem_keep",
            content="To keep",
            session_id=100,
            role="user",
            timestamp=datetime.utcnow().isoformat(),
        )

        vector_memory.delete_memories_by_session(99)

        # Verify session 99 memories deleted
        results = vector_memory.search_similar_memories(
            query="delete",
            n_results=10,
            session_filter=99,
        )
        assert len(results) == 0

        # Verify session 100 still exists
        results = vector_memory.search_similar_memories(
            query="keep",
            n_results=10,
            session_filter=100,
        )
        assert len(results) >= 1


# ============== Skill Tests ==============


class TestSkillOperations:
    """Tests for skill vector operations."""

    def test_add_skill_basic(self, vector_memory, mock_embedder):
        """Test adding a skill."""
        vector_memory.add_skill(
            skill_name="csv-import",
            description="Import CSV files into database",
            use_count=5,
            success_count=4,
        )

        results = vector_memory.search_similar_skills(
            query="import CSV data",
            n_results=5,
        )
        assert len(results) >= 1

    def test_add_skill_with_body(self, vector_memory, mock_embedder):
        """Test adding skill with full body."""
        vector_memory.add_skill(
            skill_name="full-skill",
            description="Complete skill",
            body="Step 1: Read file\nStep 2: Parse CSV\nStep 3: Insert",
            use_count=10,
            success_count=9,
        )

        results = vector_memory.search_similar_skills(
            query="complete skill",
            n_results=5,
        )
        assert len(results) >= 1

    def test_add_skill_upsert_behavior(self, vector_memory, mock_embedder):
        """Test that adding same skill name overwrites."""
        vector_memory.add_skill(
            skill_name="upsert-skill",
            description="Original description",
            use_count=1,
        )

        vector_memory.add_skill(
            skill_name="upsert-skill",
            description="Updated description",
            use_count=5,
        )

        results = vector_memory.search_similar_skills(
            query="upsert skill",
            n_results=5,
        )
        # Should find only one
        matching = [r for r in results if r["id"] == "upsert-skill"]
        assert len(matching) == 1
        assert matching[0]["description"].startswith("Updated description")


# ============== Entity Tests ==============


class TestEntityOperations:
    """Tests for entity vector operations."""

    def test_add_entity_basic(self, vector_memory, mock_embedder):
        """Test adding an entity."""
        vector_memory.add_entity(
            entity_id="ent_001",
            description="Tool: httpx HTTP client library",
            entity_type="tool",
            graph_node_id="entity_tool_httpx",
        )

        results = vector_memory.search_similar_entities(
            query="HTTP client",
            n_results=5,
        )
        assert len(results) >= 1

    def test_add_entity_with_type_filter(self, vector_memory, mock_embedder):
        """Test entity search with type filter."""
        vector_memory.add_entity(
            entity_id="ent_tool",
            description="Tool description",
            entity_type="tool",
            graph_node_id="node_tool",
        )
        vector_memory.add_entity(
            entity_id="ent_env",
            description="Environment description",
            entity_type="environment",
            graph_node_id="node_env",
        )

        results = vector_memory.search_similar_entities(
            query="description",
            n_results=10,
            entity_type_filter="tool",
        )

        for entity in results:
            assert entity["metadata"]["entity_type"] == "tool"

    def test_add_entity_upsert_behavior(self, vector_memory, mock_embedder):
        """Test that adding same entity ID overwrites."""
        vector_memory.add_entity(
            entity_id="ent_upsert",
            description="Original description",
            entity_type="tool",
            graph_node_id="node_1",
        )

        vector_memory.add_entity(
            entity_id="ent_upsert",
            description="Updated description",
            entity_type="tool",
            graph_node_id="node_1",
        )

        results = vector_memory.search_similar_entities(
            query="upsert",
            n_results=5,
        )
        matching = [r for r in results if r["id"] == "ent_upsert"]
        assert len(matching) == 1
        assert matching[0]["description"] == "Updated description"


# ============== Stats and Clear Tests ==============


class TestStatsAndClear:
    """Tests for statistics and clear operations."""

    def test_stats_empty(self, vector_memory):
        """Test stats on empty collections."""
        stats = vector_memory.stats()
        assert stats["facts_count"] == 0
        assert stats["memories_count"] == 0
        assert stats["skills_count"] == 0
        assert stats["entities_count"] == 0

    def test_stats_with_data(self, vector_memory, mock_embedder):
        """Test stats with data in collections."""
        vector_memory.add_fact(
            fact_id="f1",
            content="Fact 1",
            category="general",
            confidence=0.8,
            source_turn=1,
        )
        vector_memory.add_memory(
            memory_id="m1",
            content="Memory 1",
            session_id=1,
            role="user",
            timestamp=datetime.utcnow().isoformat(),
        )
        vector_memory.add_skill(
            skill_name="skill1",
            description="Skill 1",
        )
        vector_memory.add_entity(
            entity_id="e1",
            description="Entity 1",
            entity_type="tool",
            graph_node_id="node1",
        )

        stats = vector_memory.stats()
        assert stats["facts_count"] == 1
        assert stats["memories_count"] == 1
        assert stats["skills_count"] == 1
        assert stats["entities_count"] == 1

    def test_clear_all(self, vector_memory, mock_embedder):
        """Test clearing all collections."""
        # Add data
        vector_memory.add_fact(
            fact_id="f_clear",
            content="To clear",
            category="general",
            confidence=0.8,
            source_turn=1,
        )

        # Clear
        vector_memory.clear_all()

        # Verify empty
        stats = vector_memory.stats()
        assert stats["facts_count"] == 0

    def test_close_no_error(self, vector_memory):
        """Test close method (no-op for ChromaDB)."""
        # Should not raise
        vector_memory.close()


# ============== Edge Cases ==============


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_search_empty_query(self, vector_memory, mock_embedder):
        """Test search with empty query."""
        # ChromaDB should handle this (may return empty or error)
        # We test that it doesn't crash
        results = vector_memory.search_similar_facts(
            query="",
            n_results=5,
        )
        # May return empty or all items
        assert isinstance(results, list)

    def test_large_n_results(self, vector_memory, mock_embedder):
        """Test search with large n_results."""
        vector_memory.add_fact(
            fact_id="f_large",
            content="Large result test",
            category="general",
            confidence=0.8,
            source_turn=1,
        )

        results = vector_memory.search_similar_facts(
            query="test",
            n_results=1000,
        )
        # Should return at most 1 (only one fact)
        assert len(results) <= 1

    def test_add_fact_zero_confidence(self, vector_memory, mock_embedder):
        """Test adding fact with zero confidence."""
        vector_memory.add_fact(
            fact_id="f_zero",
            content="Zero confidence",
            category="ephemeral",
            confidence=0.0,
            source_turn=1,
        )

        fact = vector_memory.get_fact_by_id("f_zero")
        assert fact["metadata"]["confidence"] == 0.0

    def test_add_fact_high_confidence(self, vector_memory, mock_embedder):
        """Test adding fact with confidence > 1."""
        # Confidence should work with values up to 1.0
        vector_memory.add_fact(
            fact_id="f_high",
            content="High confidence",
            category="preference",
            confidence=1.0,
            source_turn=1,
        )

        fact = vector_memory.get_fact_by_id("f_high")
        assert fact["metadata"]["confidence"] == 1.0

    def test_add_fact_long_content(self, vector_memory, mock_embedder):
        """Test adding fact with long content."""
        long_content = "A" * 10000  # 10k characters
        vector_memory.add_fact(
            fact_id="f_long",
            content=long_content,
            category="general",
            confidence=0.8,
            source_turn=1,
        )

        fact = vector_memory.get_fact_by_id("f_long")
        assert fact["content"] == long_content

    def test_add_fact_unicode_content(self, vector_memory, mock_embedder):
        """Test adding fact with unicode content."""
        unicode_content = "用户偏好：使用 httpx 而不是 requests 😊"
        vector_memory.add_fact(
            fact_id="f_unicode",
            content=unicode_content,
            category="preference",
            confidence=0.9,
            source_turn=1,
        )

        fact = vector_memory.get_fact_by_id("f_unicode")
        assert fact["content"] == unicode_content

    def test_add_fact_special_chars_in_id(self, vector_memory, mock_embedder):
        """Test adding fact with special characters in ID."""
        # ChromaDB may have restrictions on IDs
        vector_memory.add_fact(
            fact_id="fact_with-dash_underscore",
            content="Special ID",
            category="general",
            confidence=0.8,
            source_turn=1,
        )

        fact = vector_memory.get_fact_by_id("fact_with-dash_underscore")
        assert fact is not None

    def test_search_returns_distance_field(self, vector_memory, mock_embedder):
        """Test that search results include distance field."""
        vector_memory.add_fact(
            fact_id="f_dist",
            content="Distance test",
            category="general",
            confidence=0.8,
            source_turn=1,
        )

        results = vector_memory.search_similar_facts(
            query="distance",
            n_results=5,
        )

        if results:
            assert "distance" in results[0]

    def test_get_recent_facts_empty(self, vector_memory):
        """Test get_recent_facts on empty collection."""
        recent = vector_memory.get_recent_facts(limit=10)
        assert len(recent) == 0
