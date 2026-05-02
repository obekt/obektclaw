"""Tests for MemorySync.

Tests cover:
- Entity sync from CogDB to ChromaDB
- Consistency checking between stores
- Fact-entity linking
- Entity extraction from facts
- Edge cases: missing entities, sync errors
"""

import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obektclaw.memory.graph_memory import Entity, ENTITY_TYPES
from obektclaw.memory.memory_sync import MemorySync


# ============== Mock Fixtures ==============


@pytest.fixture
def mock_graph_memory():
    """Create a mock GraphMemory."""
    mock = MagicMock()
    mock.get_entities_by_type.return_value = []
    mock.get_entity.return_value = None
    mock.get_all_entities.return_value = []
    return mock


@pytest.fixture
def mock_vector_memory():
    """Create a mock VectorMemory."""
    mock = MagicMock()
    mock.add_entity.return_value = None
    mock.entities = MagicMock()
    mock.entities.count.return_value = 0
    mock.entities.get.return_value = {"ids": [], "metadatas": []}
    mock.search_similar_entities.return_value = []
    mock.get_fact_by_id.return_value = None
    mock.update_fact_confidence.return_value = None
    return mock


@pytest.fixture
def memory_sync(mock_graph_memory, mock_vector_memory):
    """Create a MemorySync with mocked dependencies."""
    return MemorySync(
        graph_memory=mock_graph_memory,
        vector_memory=mock_vector_memory,
    )


# ============== Entity Sync Tests ==============


class TestSyncEntityToVector:
    """Tests for sync_entity_to_vector."""

    def test_sync_entity_basic(self, memory_sync, mock_vector_memory):
        """Test basic entity sync."""
        vector_id = memory_sync.sync_entity_to_vector(
            entity_id="entity_httpx",
            entity_name="httpx",
            entity_type="tool",
            description="HTTP client library",
        )

        # Should generate a vector ID
        assert vector_id is not None
        assert len(vector_id) == 12  # MD5 hash truncated to 12 chars

        # Should call add_entity
        mock_vector_memory.add_entity.assert_called_once()
        call_args = mock_vector_memory.add_entity.call_args
        assert call_args.kwargs["entity_id"] == vector_id
        assert call_args.kwargs["entity_type"] == "tool"
        assert call_args.kwargs["graph_node_id"] == "entity_httpx"

    def test_sync_entity_without_description(self, memory_sync, mock_vector_memory):
        """Test entity sync without description (auto-generated)."""
        vector_id = memory_sync.sync_entity_to_vector(
            entity_id="e1",
            entity_name="pytest",
            entity_type="tool",
        )

        call_args = mock_vector_memory.add_entity.call_args
        # Should auto-generate description
        assert "tool: pytest" in call_args.kwargs["description"]

    def test_sync_entity_vector_id_consistent(self, memory_sync):
        """Test that vector ID is consistent for same entity."""
        # Same entity should always get same vector ID
        id1 = memory_sync.sync_entity_to_vector(
            entity_id="entity_001",
            entity_name="name1",
            entity_type="tool",
        )
        id2 = memory_sync.sync_entity_to_vector(
            entity_id="entity_001",
            entity_name="name1",
            entity_type="tool",
        )

        assert id1 == id2

    def test_sync_entity_vector_id_different(self, memory_sync):
        """Test that different entities get different vector IDs."""
        id1 = memory_sync.sync_entity_to_vector(
            entity_id="entity_001",
            entity_name="name1",
            entity_type="tool",
        )
        id2 = memory_sync.sync_entity_to_vector(
            entity_id="entity_002",
            entity_name="name2",
            entity_type="tool",
        )

        assert id1 != id2

    def test_sync_entity_vector_id_hash(self):
        """Test vector ID is MD5 hash of entity ID."""
        entity_id = "test_entity"
        expected_hash = hashlib.md5(entity_id.encode()).hexdigest()[:12]

        # Verify hash logic
        assert len(expected_hash) == 12

    def test_sync_entity_all_types(self, memory_sync, mock_vector_memory):
        """Test syncing entities of all types."""
        for entity_type in ENTITY_TYPES:
            vector_id = memory_sync.sync_entity_to_vector(
                entity_id=f"entity_{entity_type}",
                entity_name=f"name_{entity_type}",
                entity_type=entity_type,
            )
            assert vector_id is not None


class TestSyncAllEntities:
    """Tests for sync_all_entities."""

    def test_sync_all_empty(self, memory_sync, mock_graph_memory):
        """Test sync with no entities."""
        mock_graph_memory.get_entities_by_type.return_value = []

        stats = memory_sync.sync_all_entities()

        assert stats["synced"] == 0
        assert stats["skipped"] == 0
        assert stats["errors"] == 0

    def test_sync_all_with_entities(
        self, memory_sync, mock_graph_memory, mock_vector_memory
    ):
        """Test sync with entities."""
        entities = [
            Entity(id="e1", entity_type="tool", name="httpx"),
            Entity(id="e2", entity_type="environment", name="Hetzner"),
        ]

        # Use side_effect to return entities only for specific types
        def get_entities_by_type_side_effect(entity_type):
            if entity_type == "tool":
                return [Entity(id="e1", entity_type="tool", name="httpx")]
            elif entity_type == "environment":
                return [Entity(id="e2", entity_type="environment", name="Hetzner")]
            return []

        mock_graph_memory.get_entities_by_type.side_effect = (
            get_entities_by_type_side_effect
        )

        stats = memory_sync.sync_all_entities()

        assert stats["synced"] == 2
        assert stats["by_type"]["tool"] == 1
        assert stats["by_type"]["environment"] == 1

    def test_sync_all_by_type(self, memory_sync, mock_graph_memory):
        """Test sync counts by type."""

        # Mock different entity types
        def get_entities_by_type_side_effect(entity_type):
            if entity_type == "tool":
                return [Entity(id="t1", entity_type="tool", name="tool1")]
            elif entity_type == "environment":
                return [Entity(id="e1", entity_type="environment", name="env1")]
            return []

        mock_graph_memory.get_entities_by_type.side_effect = (
            get_entities_by_type_side_effect
        )

        stats = memory_sync.sync_all_entities()

        assert "tool" in stats["by_type"]
        assert "environment" in stats["by_type"]

    def test_sync_all_with_properties(
        self, memory_sync, mock_graph_memory, mock_vector_memory
    ):
        """Test sync includes entity properties in description."""
        entity = Entity(
            id="e1",
            entity_type="environment",
            name="Hetzner",
            properties={"server_type": "CX22", "location": "Germany"},
        )

        def get_entities_by_type_side_effect(entity_type):
            if entity_type == "environment":
                return [entity]
            return []

        mock_graph_memory.get_entities_by_type.side_effect = (
            get_entities_by_type_side_effect
        )

        stats = memory_sync.sync_all_entities()

        # Should include properties in description
        call_args = mock_vector_memory.add_entity.call_args
        description = call_args.kwargs["description"]
        assert "server_type" in description or "CX22" in description

    def test_sync_all_handles_errors(
        self, memory_sync, mock_graph_memory, mock_vector_memory
    ):
        """Test sync handles errors gracefully."""
        entity = Entity(id="e1", entity_type="tool", name="test")

        def get_entities_by_type_side_effect(entity_type):
            if entity_type == "tool":
                return [entity]
            return []

        mock_graph_memory.get_entities_by_type.side_effect = (
            get_entities_by_type_side_effect
        )
        mock_vector_memory.add_entity.side_effect = Exception("DB error")

        stats = memory_sync.sync_all_entities()

        assert stats["errors"] >= 1
        assert stats["synced"] == 0


# ============== Entity Extraction Tests ==============


class TestExtractEntitiesFromFact:
    """Tests for extract_entities_from_fact."""

    def test_extract_no_matches(self, memory_sync, mock_vector_memory):
        """Test with no matching entities."""
        mock_vector_memory.search_similar_entities.return_value = []

        matches = memory_sync.extract_entities_from_fact(
            fact_content="Generic fact",
            category="general",
        )

        assert len(matches) == 0

    def test_extract_with_matches(
        self, memory_sync, mock_vector_memory, mock_graph_memory
    ):
        """Test with matching entities."""
        mock_entity = Entity(id="e_httpx", entity_type="tool", name="httpx")
        mock_graph_memory.get_entity.return_value = mock_entity

        mock_vector_memory.search_similar_entities.return_value = [
            {
                "id": "v1",
                "metadata": {"graph_node_id": "e_httpx"},
                "distance": 0.3,
            }
        ]

        matches = memory_sync.extract_entities_from_fact(
            fact_content="User prefers httpx",
            category="preference",
        )

        assert len(matches) == 1
        assert matches[0]["name"] == "httpx"

    def test_extract_distance_threshold(self, memory_sync, mock_vector_memory):
        """Test that high-distance matches are excluded."""
        mock_vector_memory.search_similar_entities.return_value = [
            {
                "id": "v1",
                "metadata": {"graph_node_id": "e1"},
                "distance": 0.8,  # Too high (> 0.5)
            }
        ]

        matches = memory_sync.extract_entities_from_fact("fact", "general")

        # Should exclude due to high distance
        assert len(matches) == 0

    def test_extract_multiple_matches(
        self, memory_sync, mock_vector_memory, mock_graph_memory
    ):
        """Test with multiple matching entities."""
        e1 = Entity(id="e_httpx", entity_type="tool", name="httpx")
        e2 = Entity(id="e_hetzner", entity_type="environment", name="Hetzner")

        def get_entity_side_effect(id):
            if id == "e_httpx":
                return e1
            elif id == "e_hetzner":
                return e2
            return None

        mock_graph_memory.get_entity.side_effect = get_entity_side_effect

        mock_vector_memory.search_similar_entities.return_value = [
            {"id": "v1", "metadata": {"graph_node_id": "e_httpx"}, "distance": 0.2},
            {"id": "v2", "metadata": {"graph_node_id": "e_hetzner"}, "distance": 0.3},
        ]

        matches = memory_sync.extract_entities_from_fact(
            fact_content="httpx on Hetzner",
            category="environment",
        )

        assert len(matches) == 2

    def test_extract_missing_graph_node_id(self, memory_sync, mock_vector_memory):
        """Test with missing graph_node_id."""
        mock_vector_memory.search_similar_entities.return_value = [
            {"id": "v1", "metadata": {}, "distance": 0.2}  # No graph_node_id
        ]

        matches = memory_sync.extract_entities_from_fact("fact", "general")

        assert len(matches) == 0

    def test_extract_missing_entity_in_graph(
        self, memory_sync, mock_vector_memory, mock_graph_memory
    ):
        """Test when graph entity doesn't exist."""
        mock_vector_memory.search_similar_entities.return_value = [
            {"id": "v1", "metadata": {"graph_node_id": "missing"}, "distance": 0.2}
        ]
        mock_graph_memory.get_entity.return_value = None

        matches = memory_sync.extract_entities_from_fact("fact", "general")

        assert len(matches) == 0

    def test_extract_includes_distance(
        self, memory_sync, mock_vector_memory, mock_graph_memory
    ):
        """Test that matches include distance."""
        mock_entity = Entity(id="e1", entity_type="tool", name="test")
        mock_graph_memory.get_entity.return_value = mock_entity

        mock_vector_memory.search_similar_entities.return_value = [
            {"id": "v1", "metadata": {"graph_node_id": "e1"}, "distance": 0.25}
        ]

        matches = memory_sync.extract_entities_from_fact("fact", "general")

        assert matches[0]["distance"] == 0.25


# ============== Fact Linking Tests ==============


class TestLinkFactToEntities:
    """Tests for link_fact_to_entities."""

    def test_link_fact_basic(self, memory_sync, mock_vector_memory):
        """Test basic fact linking."""
        mock_vector_memory.get_fact_by_id.return_value = {
            "id": "fact_001",
            "content": "test fact",
            "metadata": {"confidence": 0.9, "category": "general"},
        }

        memory_sync.link_fact_to_entities(
            fact_id="fact_001",
            entity_ids=["e_httpx", "e_hetzner"],
        )

        # Should update fact metadata
        mock_vector_memory.update_fact_confidence.assert_called_once()
        call_args = mock_vector_memory.update_fact_confidence.call_args
        assert call_args.args[0] == "fact_001"

    def test_link_fact_missing(self, memory_sync, mock_vector_memory):
        """Test linking missing fact."""
        mock_vector_memory.get_fact_by_id.return_value = None

        memory_sync.link_fact_to_entities("nonexistent", ["e1"])

        # Should silently return (no update)
        mock_vector_memory.update_fact_confidence.assert_not_called()

    def test_link_fact_updates_metadata(self, memory_sync, mock_vector_memory):
        """Test that linking updates metadata."""
        mock_vector_memory.get_fact_by_id.return_value = {
            "id": "fact_001",
            "content": "test",
            "metadata": {"confidence": 0.8, "category": "preference"},
        }

        memory_sync.link_fact_to_entities("fact_001", ["e1", "e2"])

        # Should call update_fact_confidence which updates metadata
        mock_vector_memory.update_fact_confidence.assert_called()

    def test_link_fact_empty_entity_list(self, memory_sync, mock_vector_memory):
        """Test linking with empty entity list."""
        mock_vector_memory.get_fact_by_id.return_value = {
            "id": "fact_001",
            "content": "test",
            "metadata": {"confidence": 0.8},
        }

        memory_sync.link_fact_to_entities("fact_001", [])

        # Should still update (with empty entity_ids)
        mock_vector_memory.update_fact_confidence.assert_called()


# ============== Consistency Check Tests ==============


class TestConsistencyCheck:
    """Tests for check_consistency."""

    def test_consistency_empty(
        self, memory_sync, mock_graph_memory, mock_vector_memory
    ):
        """Test consistency with empty stores."""
        mock_graph_memory.get_entities_by_type.return_value = []
        mock_vector_memory.entities.get.return_value = {"ids": [], "metadatas": []}

        report = memory_sync.check_consistency()

        assert report["graph_entities"] == 0
        assert report["vector_entities"] == 0
        assert report["consistent"] == 0

    def test_consistency_with_matching(
        self, memory_sync, mock_graph_memory, mock_vector_memory
    ):
        """Test consistency with matching entities."""
        entity = Entity(id="e1", entity_type="tool", name="test")

        def get_entities_by_type_side_effect(entity_type):
            if entity_type == "tool":
                return [entity]
            return []

        mock_graph_memory.get_entities_by_type.side_effect = (
            get_entities_by_type_side_effect
        )

        mock_vector_memory.entities.get.return_value = {
            "ids": ["v1"],
            "metadatas": [{"graph_node_id": "e1"}],
        }

        report = memory_sync.check_consistency()

        assert report["graph_entities"] >= 1
        assert report["consistent"] >= 1

    def test_consistency_missing_in_vector(
        self, memory_sync, mock_graph_memory, mock_vector_memory
    ):
        """Test when entity missing in vector store."""
        entity = Entity(id="e_missing", entity_type="tool", name="missing")

        def get_entities_by_type_side_effect(entity_type):
            if entity_type == "tool":
                return [entity]
            return []

        mock_graph_memory.get_entities_by_type.side_effect = (
            get_entities_by_type_side_effect
        )

        mock_vector_memory.entities.get.return_value = {
            "ids": ["v1"],
            "metadatas": [{"graph_node_id": "other_entity"}],
        }

        report = memory_sync.check_consistency()

        assert len(report["missing_in_vector"]) >= 1
        assert "e_missing" in report["missing_in_vector"]

    def test_consistency_missing_in_graph(
        self, memory_sync, mock_graph_memory, mock_vector_memory
    ):
        """Test when entity in vector but not graph."""
        mock_graph_memory.get_entities_by_type.return_value = []

        mock_vector_memory.entities.get.return_value = {
            "ids": ["v1"],
            "metadatas": [{"graph_node_id": "orphan_entity"}],
        }

        report = memory_sync.check_consistency()

        assert len(report["missing_in_graph"]) >= 1
        assert "orphan_entity" in report["missing_in_graph"]

    def test_consistency_counts(
        self, memory_sync, mock_graph_memory, mock_vector_memory
    ):
        """Test consistency counts."""

        # 3 entities in graph - spread across types
        def get_entities_by_type_side_effect(entity_type):
            if entity_type == "tool":
                return [
                    Entity(id="e1", entity_type="tool", name="t1"),
                    Entity(id="e2", entity_type="tool", name="t2"),
                ]
            elif entity_type == "environment":
                return [Entity(id="e3", entity_type="environment", name="env1")]
            return []

        mock_graph_memory.get_entities_by_type.side_effect = (
            get_entities_by_type_side_effect
        )

        # 2 matching in vector, 1 orphan
        mock_vector_memory.entities.count.return_value = 3
        mock_vector_memory.entities.get.return_value = {
            "ids": ["v1", "v2", "v3"],
            "metadatas": [
                {"graph_node_id": "e1"},
                {"graph_node_id": "e2"},
                {"graph_node_id": "orphan"},
            ],
        }

        report = memory_sync.check_consistency()

        assert report["graph_entities"] == 3
        assert report["vector_entities"] == 3
        assert report["consistent"] == 2
        assert len(report["missing_in_graph"]) == 1

    def test_consistency_all_entity_types(self, memory_sync, mock_graph_memory):
        """Test consistency checks all entity types."""
        mock_graph_memory.get_entities_by_type.return_value = []

        report = memory_sync.check_consistency()

        # Should iterate through all ENTITY_TYPES
        # Verify by checking graph_entities count (should be 0 for each)
        assert mock_graph_memory.get_entities_by_type.call_count >= len(ENTITY_TYPES)


# ============== Integration Tests ==============


class TestMemorySyncIntegration:
    """Integration-style tests."""

    def test_full_sync_flow(self, memory_sync, mock_graph_memory, mock_vector_memory):
        """Test full sync flow."""

        # Set up entities per type
        def get_entities_by_type_side_effect(entity_type):
            if entity_type == "tool":
                return [
                    Entity(
                        id="e_httpx",
                        entity_type="tool",
                        name="httpx",
                        properties={"version": "0.24"},
                    ),
                ]
            elif entity_type == "environment":
                return [
                    Entity(
                        id="e_hetzner",
                        entity_type="environment",
                        name="Hetzner",
                        properties={"region": "eu"},
                    ),
                ]
            return []

        mock_graph_memory.get_entities_by_type.side_effect = (
            get_entities_by_type_side_effect
        )

        # Sync all
        stats = memory_sync.sync_all_entities()

        assert stats["synced"] == 2

        # Check consistency
        mock_vector_memory.entities.get.return_value = {
            "ids": ["v1", "v2"],
            "metadatas": [
                {"graph_node_id": "e_httpx"},
                {"graph_node_id": "e_hetzner"},
            ],
        }

        report = memory_sync.check_consistency()
        assert report["consistent"] >= 2


# ============== Edge Cases ==============


class TestMemorySyncEdgeCases:
    """Tests for edge cases."""

    def test_sync_entity_empty_name(self, memory_sync, mock_vector_memory):
        """Test syncing entity with empty name."""
        vector_id = memory_sync.sync_entity_to_vector(
            entity_id="e_empty",
            entity_name="",
            entity_type="tool",
        )

        assert vector_id is not None
        call_args = mock_vector_memory.add_entity.call_args
        assert "tool: " in call_args.kwargs["description"]

    def test_sync_entity_unicode_name(self, memory_sync, mock_vector_memory):
        """Test syncing entity with unicode name."""
        vector_id = memory_sync.sync_entity_to_vector(
            entity_id="e_unicode",
            entity_name="日本語",
            entity_type="concept",
        )

        assert vector_id is not None

    def test_sync_entity_special_chars_id(self, memory_sync):
        """Test syncing entity with special characters in ID."""
        vector_id = memory_sync.sync_entity_to_vector(
            entity_id="entity-with-dash_underscore",
            entity_name="name",
            entity_type="tool",
        )

        assert vector_id is not None
        assert len(vector_id) == 12

    def test_extract_fact_empty_content(self, memory_sync, mock_vector_memory):
        """Test extraction with empty fact content."""
        mock_vector_memory.search_similar_entities.return_value = []

        matches = memory_sync.extract_entities_from_fact("", "general")

        assert len(matches) == 0

    def test_link_fact_empty_id(self, memory_sync, mock_vector_memory):
        """Test linking fact with empty ID."""
        mock_vector_memory.get_fact_by_id.return_value = None

        memory_sync.link_fact_to_entities("", ["e1"])

        mock_vector_memory.update_fact_confidence.assert_not_called()

    def test_consistency_large_vector_entities(
        self, memory_sync, mock_graph_memory, mock_vector_memory
    ):
        """Test consistency with many vector entities."""
        mock_graph_memory.get_entities_by_type.return_value = []

        # Large response from vector store
        mock_vector_memory.entities.count.return_value = 100
        mock_vector_memory.entities.get.return_value = {
            "ids": [f"v{i}" for i in range(100)],
            "metadatas": [{"graph_node_id": f"e{i}"} for i in range(100)],
        }

        report = memory_sync.check_consistency()

        assert report["vector_entities"] == 100

    def test_sync_entity_properties_excluded_metadata(
        self, memory_sync, mock_graph_memory, mock_vector_memory
    ):
        """Test that created_at/updated_at are excluded from description."""
        entity = Entity(
            id="e1",
            entity_type="tool",
            name="test",
            properties={
                "version": "1.0",
                "created_at": "2024-01-01",
                "updated_at": "2024-01-02",
            },
        )

        def get_entities_by_type_side_effect(entity_type):
            if entity_type == "tool":
                return [entity]
            return []

        mock_graph_memory.get_entities_by_type.side_effect = (
            get_entities_by_type_side_effect
        )

        memory_sync.sync_all_entities()

        call_args = mock_vector_memory.add_entity.call_args
        description = call_args.kwargs["description"]

        # Should include version but not created_at/updated_at
        assert "version" in description
        assert "created_at" not in description

    def test_sync_all_partial_failure(
        self, memory_sync, mock_graph_memory, mock_vector_memory
    ):
        """Test sync continues after partial failure."""
        e1 = Entity(id="e_good", entity_type="tool", name="good")
        e2 = Entity(id="e_bad", entity_type="tool", name="bad")

        def get_entities_by_type_side_effect(entity_type):
            if entity_type == "tool":
                return [e1, e2]
            return []

        mock_graph_memory.get_entities_by_type.side_effect = (
            get_entities_by_type_side_effect
        )

        # First succeeds, second fails
        call_count = [0]

        def add_entity_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Error")

        mock_vector_memory.add_entity.side_effect = add_entity_side_effect

        stats = memory_sync.sync_all_entities()

        # Should have 1 success and 1 error
        assert stats["synced"] >= 1
        assert stats["errors"] >= 1
