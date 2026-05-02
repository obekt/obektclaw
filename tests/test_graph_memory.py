"""Tests for GraphMemory (CogDB wrapper).

Tests cover:
- Entity CRUD operations
- Relation CRUD operations
- Graph traversal (connected entities)
- User preferences retrieval
- Edge cases: missing entities, circular relations, soft delete
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obektclaw.memory.graph_memory import (
    Entity,
    Relation,
    GraphMemory,
    ENTITY_TYPES,
    RELATION_TYPES,
)


# ============== Entity/Relation Dataclass Tests ==============


class TestEntityDataclass:
    """Tests for Entity dataclass."""

    def test_entity_creation(self):
        """Test creating an entity."""
        entity = Entity(
            id="entity_001",
            entity_type="tool",
            name="httpx",
            properties={"version": "0.24.0"},
            confidence=0.9,
        )
        assert entity.id == "entity_001"
        assert entity.entity_type == "tool"
        assert entity.name == "httpx"
        assert entity.properties["version"] == "0.24.0"

    def test_entity_to_dict(self):
        """Test entity serialization."""
        entity = Entity(
            id="e1",
            entity_type="tool",
            name="pytest",
            properties={"language": "python"},
            confidence=0.8,
        )
        data = entity.to_dict()
        assert data["id"] == "e1"
        assert data["entity_type"] == "tool"
        assert data["name"] == "pytest"
        assert "created_at" in data
        assert "updated_at" in data

    def test_entity_from_dict(self):
        """Test entity deserialization."""
        data = {
            "id": "e2",
            "entity_type": "environment",
            "name": "Hetzner",
            "properties": {"server": "CX22"},
            "confidence": 0.95,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-02T00:00:00",
        }
        entity = Entity.from_dict(data)
        assert entity.id == "e2"
        assert entity.entity_type == "environment"
        assert entity.created_at.year == 2024

    def test_entity_default_values(self):
        """Test entity default values."""
        entity = Entity(id="e3", entity_type="concept", name="async")
        assert entity.properties == {}
        assert entity.confidence == 1.0
        assert isinstance(entity.created_at, datetime)
        assert isinstance(entity.updated_at, datetime)

    def test_entity_roundtrip(self):
        """Test entity serialization roundtrip."""
        original = Entity(
            id="e4",
            entity_type="project",
            name="obektclaw",
            properties={"language": "python"},
            confidence=0.85,
        )
        data = original.to_dict()
        restored = Entity.from_dict(data)
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.properties == original.properties


class TestRelationDataclass:
    """Tests for Relation dataclass."""

    def test_relation_creation(self):
        """Test creating a relation."""
        relation = Relation(
            id="rel_001",
            source_id="entity_user",
            target_id="entity_httpx",
            relation_type="prefers",
            properties={"strength": "strong"},
            confidence=0.9,
        )
        assert relation.id == "rel_001"
        assert relation.source_id == "entity_user"
        assert relation.target_id == "entity_httpx"
        assert relation.relation_type == "prefers"

    def test_relation_to_dict(self):
        """Test relation serialization."""
        relation = Relation(
            id="r1",
            source_id="s1",
            target_id="t1",
            relation_type="uses",
            confidence=0.8,
        )
        data = relation.to_dict()
        assert data["id"] == "r1"
        assert data["source_id"] == "s1"
        assert data["target_id"] == "t1"
        assert data["relation_type"] == "uses"
        assert "created_at" in data

    def test_relation_from_dict(self):
        """Test relation deserialization."""
        data = {
            "id": "r2",
            "source_id": "user",
            "target_id": "tool",
            "relation_type": "dislikes",
            "properties": {},
            "confidence": 0.7,
            "created_at": "2024-01-01T00:00:00",
        }
        relation = Relation.from_dict(data)
        assert relation.id == "r2"
        assert relation.relation_type == "dislikes"

    def test_relation_roundtrip(self):
        """Test relation serialization roundtrip."""
        original = Relation(
            id="r3",
            source_id="user",
            target_id="server",
            relation_type="owns",
            confidence=0.95,
        )
        data = original.to_dict()
        restored = Relation.from_dict(data)
        assert restored.id == original.id
        assert restored.relation_type == original.relation_type


# ============== Constants Tests ==============


class TestConstants:
    """Tests for type constants."""

    def test_entity_types_defined(self):
        """Test that entity types are defined."""
        assert "tool" in ENTITY_TYPES
        assert "concept" in ENTITY_TYPES
        assert "environment" in ENTITY_TYPES
        assert "person" in ENTITY_TYPES
        assert len(ENTITY_TYPES) >= 5

    def test_relation_types_defined(self):
        """Test that relation types are defined."""
        assert "prefers" in RELATION_TYPES
        assert "uses" in RELATION_TYPES
        assert "dislikes" in RELATION_TYPES
        assert "depends_on" in RELATION_TYPES
        assert len(RELATION_TYPES) >= 5


# ============== Mock Graph Tests ==============


class MockGraph:
    """Mock implementation of cog.torque.Graph."""

    # Relation predicates that should allow multiple edges
    RELATION_PREDICATES = frozenset(
        [
            "prefers",
            "uses",
            "dislikes",
            "depends_on",
            "related_to",
            "owns",
            "works_on",
            "deployed_on",
        ]
    )
    # Metadata predicates that should be unique per subject
    METADATA_PREDICATES = frozenset(
        [
            "is_entity",
            "is_relation",
            "has_type",
            "has_name",
            "has_relation_type",
        ]
    )

    def __init__(self, graph_name="test", cog_home="/tmp", enable_caching=True):
        self._triples = []  # List of (subject, predicate, object) tuples
        self.graph_name = graph_name

    def put(self, subject, predicate, object):
        """Add or update a triple.

        For relation predicates (prefers, uses, etc.), allows multiple edges.
        For metadata predicates (is_entity, has_type, etc.), overwrites existing.
        """
        # Relation predicates can have multiple edges (don't overwrite)
        if predicate in self.RELATION_PREDICATES:
            self._triples.append((subject, predicate, object))
        else:
            # Metadata/property predicates should be unique per subject (overwrite)
            self._triples = [
                t for t in self._triples if not (t[0] == subject and t[1] == predicate)
            ]
            self._triples.append((subject, predicate, object))

    def triples(self):
        """Return all triples."""
        return list(self._triples)

    def v(self, node_id):
        """Return a vertex query object."""
        return MockVertexQuery(self, node_id)


class MockVertexQuery:
    """Mock vertex query for Graph.v()."""

    def __init__(self, graph, node_id):
        self._graph = graph
        self._node_id = node_id

    def out(self, predicate):
        """Query outgoing edges."""
        results = []
        for triple in self._graph.triples():
            if triple[0] == self._node_id and triple[1] == predicate:
                results.append(triple[2])
        return MockQueryResult(results)


class MockQueryResult:
    """Mock query result."""

    def __init__(self, results):
        self._results = results

    def all(self):
        """Return all results."""
        return {"result": self._results}


@pytest.fixture
def mock_graph():
    """Create a mock Graph instance."""
    return MockGraph()


@pytest.fixture
def graph_memory(mock_graph):
    """Create a GraphMemory instance with mock Graph."""
    with patch("obektclaw.memory.graph_memory.Graph", return_value=mock_graph):
        with tempfile.TemporaryDirectory() as tmpdir:
            gm = GraphMemory(Path(tmpdir))
            gm.db = mock_graph  # Replace with our mock
            yield gm


# ============== Entity CRUD Tests ==============


class TestEntityCRUD:
    """Tests for entity CRUD operations."""

    def test_add_entity_basic(self, graph_memory):
        """Test adding a basic entity."""
        entity = Entity(
            id="e_add",
            entity_type="tool",
            name="httpx",
            confidence=0.9,
        )
        graph_memory.add_entity(entity)

        # Verify entity was added
        retrieved = graph_memory.get_entity("e_add")
        assert retrieved is not None
        assert retrieved.id == "e_add"
        assert retrieved.name == "httpx"

    def test_add_entity_with_properties(self, graph_memory):
        """Test adding entity with properties."""
        entity = Entity(
            id="e_props",
            entity_type="environment",
            name="Hetzner",
            properties={"server_type": "CX22", "location": "Germany"},
            confidence=0.95,
        )
        graph_memory.add_entity(entity)

        retrieved = graph_memory.get_entity("e_props")
        assert retrieved.properties["server_type"] == "CX22"
        assert retrieved.properties["location"] == "Germany"

    def test_add_entity_upsert_behavior(self, graph_memory):
        """Test that adding same entity ID overwrites."""
        entity1 = Entity(
            id="e_upsert",
            entity_type="tool",
            name="requests",
            confidence=0.5,
        )
        graph_memory.add_entity(entity1)

        entity2 = Entity(
            id="e_upsert",
            entity_type="tool",
            name="httpx",  # Changed name
            confidence=0.9,
        )
        graph_memory.add_entity(entity2)

        retrieved = graph_memory.get_entity("e_upsert")
        assert retrieved.name == "httpx"  # Should have new name

    def test_get_entity_missing(self, graph_memory):
        """Test getting non-existent entity."""
        entity = graph_memory.get_entity("nonexistent")
        assert entity is None

    def test_get_entities_by_type(self, graph_memory):
        """Test filtering entities by type."""
        # Add entities of different types
        graph_memory.add_entity(Entity(id="e_tool1", entity_type="tool", name="httpx"))
        graph_memory.add_entity(Entity(id="e_tool2", entity_type="tool", name="pytest"))
        graph_memory.add_entity(
            Entity(id="e_env1", entity_type="environment", name="Hetzner")
        )

        tools = graph_memory.get_entities_by_type("tool")
        assert len(tools) == 2
        assert all(e.entity_type == "tool" for e in tools)

        environments = graph_memory.get_entities_by_type("environment")
        assert len(environments) == 1

    def test_get_entities_by_type_empty(self, graph_memory):
        """Test filtering by type with no matches."""
        entities = graph_memory.get_entities_by_type("workflow")
        assert len(entities) == 0

    def test_get_entities_by_name(self, graph_memory):
        """Test finding entities by name."""
        graph_memory.add_entity(Entity(id="e_name1", entity_type="tool", name="httpx"))
        graph_memory.add_entity(
            Entity(id="e_name2", entity_type="tool", name="HTTPX")
        )  # Same name, different case

        matches = graph_memory.get_entities_by_name("httpx")
        # Should match both (case-insensitive)
        assert len(matches) >= 1

    def test_get_entities_by_name_case_insensitive(self, graph_memory):
        """Test name search is case-insensitive."""
        graph_memory.add_entity(Entity(id="e_case", entity_type="tool", name="Python"))

        matches = graph_memory.get_entities_by_name("python")
        assert len(matches) >= 1

    def test_update_entity(self, graph_memory):
        """Test updating an entity."""
        entity = Entity(
            id="e_update",
            entity_type="tool",
            name="requests",
            confidence=0.5,
        )
        graph_memory.add_entity(entity)

        # Update
        entity.name = "httpx"
        entity.confidence = 0.95
        graph_memory.update_entity(entity)

        retrieved = graph_memory.get_entity("e_update")
        assert retrieved.name == "httpx"
        assert retrieved.confidence == 0.95

    def test_update_entity_sets_updated_at(self, graph_memory):
        """Test that update_entity updates the timestamp."""
        entity = Entity(
            id="e_timestamp",
            entity_type="tool",
            name="tool",
        )
        original_updated_at = entity.updated_at
        graph_memory.add_entity(entity)

        # Small delay to ensure timestamp differs
        entity.name = "updated"
        graph_memory.update_entity(entity)

        retrieved = graph_memory.get_entity("e_timestamp")
        assert retrieved.updated_at >= original_updated_at

    def test_delete_entity_soft(self, graph_memory):
        """Test that delete_entity marks as deleted (soft delete)."""
        entity = Entity(id="e_delete", entity_type="tool", name="delete_me")
        graph_memory.add_entity(entity)

        graph_memory.delete_entity("e_delete")

        # Entity still exists but marked as deleted
        retrieved = graph_memory.get_entity("e_delete")
        assert retrieved is not None
        assert retrieved.properties.get("_deleted") == True

    def test_delete_entity_missing(self, graph_memory):
        """Test deleting non-existent entity."""
        # Should not raise
        graph_memory.delete_entity("nonexistent_entity")


# ============== Relation CRUD Tests ==============


class TestRelationCRUD:
    """Tests for relation CRUD operations."""

    def test_add_relation_basic(self, graph_memory):
        """Test adding a basic relation."""
        # First add entities
        graph_memory.add_entity(Entity(id="user", entity_type="person", name="user"))
        graph_memory.add_entity(
            Entity(id="tool_httpx", entity_type="tool", name="httpx")
        )

        relation = Relation(
            id="rel_prefers",
            source_id="user",
            target_id="tool_httpx",
            relation_type="prefers",
            confidence=0.9,
        )
        graph_memory.add_relation(relation)

        # Verify relation was added
        retrieved = graph_memory.get_relation("rel_prefers")
        assert retrieved is not None
        assert retrieved.relation_type == "prefers"

    def test_add_relation_with_properties(self, graph_memory):
        """Test adding relation with properties."""
        graph_memory.add_entity(Entity(id="s1", entity_type="person", name="user"))
        graph_memory.add_entity(Entity(id="t1", entity_type="tool", name="pytest"))

        relation = Relation(
            id="rel_props",
            source_id="s1",
            target_id="t1",
            relation_type="uses",
            properties={"frequency": "daily", "context": "testing"},
            confidence=0.8,
        )
        graph_memory.add_relation(relation)

        retrieved = graph_memory.get_relation("rel_props")
        assert retrieved.properties["frequency"] == "daily"

    def test_get_relation_missing(self, graph_memory):
        """Test getting non-existent relation."""
        relation = graph_memory.get_relation("nonexistent_rel")
        assert relation is None

    def test_get_relations_from(self, graph_memory):
        """Test getting relations from an entity."""
        graph_memory.add_entity(Entity(id="user", entity_type="person", name="user"))
        graph_memory.add_entity(Entity(id="t1", entity_type="tool", name="httpx"))
        graph_memory.add_entity(Entity(id="t2", entity_type="tool", name="pytest"))

        graph_memory.add_relation(
            Relation(id="r1", source_id="user", target_id="t1", relation_type="prefers")
        )
        graph_memory.add_relation(
            Relation(id="r2", source_id="user", target_id="t2", relation_type="uses")
        )

        relations = graph_memory.get_relations_from("user")
        assert len(relations) == 2

    def test_get_relations_from_filtered(self, graph_memory):
        """Test getting relations from entity filtered by type."""
        graph_memory.add_entity(Entity(id="user", entity_type="person", name="user"))
        graph_memory.add_entity(Entity(id="t1", entity_type="tool", name="httpx"))
        graph_memory.add_entity(Entity(id="t2", entity_type="tool", name="requests"))

        graph_memory.add_relation(
            Relation(
                id="r_pref", source_id="user", target_id="t1", relation_type="prefers"
            )
        )
        graph_memory.add_relation(
            Relation(
                id="r_disl", source_id="user", target_id="t2", relation_type="dislikes"
            )
        )

        prefers = graph_memory.get_relations_from("user", relation_type="prefers")
        assert len(prefers) == 1
        assert prefers[0].relation_type == "prefers"

    def test_get_relations_to(self, graph_memory):
        """Test getting relations pointing to an entity."""
        graph_memory.add_entity(Entity(id="u1", entity_type="person", name="user1"))
        graph_memory.add_entity(Entity(id="u2", entity_type="person", name="user2"))
        graph_memory.add_entity(Entity(id="tool", entity_type="tool", name="httpx"))

        graph_memory.add_relation(
            Relation(id="r1", source_id="u1", target_id="tool", relation_type="prefers")
        )
        graph_memory.add_relation(
            Relation(id="r2", source_id="u2", target_id="tool", relation_type="uses")
        )

        relations = graph_memory.get_relations_to("tool")
        assert len(relations) == 2

    def test_get_relations_to_filtered(self, graph_memory):
        """Test getting relations to entity filtered by type."""
        graph_memory.add_entity(Entity(id="user", entity_type="person", name="user"))
        graph_memory.add_entity(Entity(id="tool", entity_type="tool", name="pytest"))

        graph_memory.add_relation(
            Relation(
                id="r1", source_id="user", target_id="tool", relation_type="prefers"
            )
        )
        graph_memory.add_relation(
            Relation(id="r2", source_id="user", target_id="tool", relation_type="uses")
        )

        prefers = graph_memory.get_relations_to("tool", relation_type="prefers")
        assert len(prefers) == 1


# ============== Graph Traversal Tests ==============


class TestGraphTraversal:
    """Tests for graph traversal operations."""

    def test_get_connected_entities_basic(self, graph_memory):
        """Test getting connected entities."""
        # Create chain: user -> prefers -> httpx -> depends_on -> h2
        graph_memory.add_entity(Entity(id="user", entity_type="person", name="user"))
        graph_memory.add_entity(Entity(id="httpx", entity_type="tool", name="httpx"))
        graph_memory.add_entity(Entity(id="h2", entity_type="tool", name="h2"))

        graph_memory.add_relation(
            Relation(
                id="r1", source_id="user", target_id="httpx", relation_type="prefers"
            )
        )
        graph_memory.add_relation(
            Relation(
                id="r2", source_id="httpx", target_id="h2", relation_type="depends_on"
            )
        )

        connected = graph_memory.get_connected_entities("user", max_depth=2)
        assert len(connected) >= 2

    def test_get_connected_entities_depth_limit(self, graph_memory):
        """Test that depth limit is respected."""
        # Create longer chain
        entities = ["e0", "e1", "e2", "e3", "e4"]
        for i, eid in enumerate(entities):
            graph_memory.add_entity(Entity(id=eid, entity_type="tool", name=f"tool{i}"))

        for i in range(len(entities) - 1):
            graph_memory.add_relation(
                Relation(
                    id=f"r{i}",
                    source_id=entities[i],
                    target_id=entities[i + 1],
                    relation_type="related_to",
                )
            )

        # Depth 1 should only get immediate neighbors
        connected_1 = graph_memory.get_connected_entities("e0", max_depth=1)
        # Should get e0 (depth 0) and e1 (depth 1)
        assert len(connected_1) >= 1

        # Depth 3 should get more
        connected_3 = graph_memory.get_connected_entities("e0", max_depth=3)
        assert len(connected_3) >= len(connected_1)

    def test_get_connected_entities_depth_zero(self, graph_memory):
        """Test traversal with depth 0."""
        graph_memory.add_entity(Entity(id="e", entity_type="tool", name="tool"))

        connected = graph_memory.get_connected_entities("e", max_depth=0)
        # Should only include starting entity
        assert len(connected) == 1
        assert connected[0][0].id == "e"
        assert connected[0][1] == 0  # Depth

    def test_get_connected_entities_skips_deleted(self, graph_memory):
        """Test that deleted entities are skipped."""
        graph_memory.add_entity(Entity(id="user", entity_type="person", name="user"))
        graph_memory.add_entity(Entity(id="tool", entity_type="tool", name="tool"))
        graph_memory.add_entity(
            Entity(id="deleted", entity_type="tool", name="deleted")
        )

        # Mark one as deleted
        graph_memory.delete_entity("deleted")

        graph_memory.add_relation(
            Relation(id="r1", source_id="user", target_id="tool", relation_type="uses")
        )
        graph_memory.add_relation(
            Relation(
                id="r2", source_id="user", target_id="deleted", relation_type="uses"
            )
        )

        connected = graph_memory.get_connected_entities("user", max_depth=1)
        # Should skip deleted entity
        for entity, depth in connected:
            assert not entity.properties.get("_deleted")

    def test_get_all_entities(self, graph_memory):
        """Test getting all entities."""
        for i in range(5):
            graph_memory.add_entity(
                Entity(id=f"all_e{i}", entity_type="tool", name=f"tool{i}")
            )

        all_entities = graph_memory.get_all_entities(limit=100)
        assert len(all_entities) >= 5

    def test_get_all_entities_limit(self, graph_memory):
        """Test limit on get_all_entities."""
        for i in range(10):
            graph_memory.add_entity(
                Entity(id=f"limit_e{i}", entity_type="tool", name=f"tool{i}")
            )

        limited = graph_memory.get_all_entities(limit=3)
        assert len(limited) <= 3

    def test_get_all_entities_skips_deleted(self, graph_memory):
        """Test that get_all_entities skips deleted."""
        graph_memory.add_entity(Entity(id="active", entity_type="tool", name="active"))
        graph_memory.add_entity(
            Entity(id="deleted", entity_type="tool", name="deleted")
        )
        graph_memory.delete_entity("deleted")

        all_entities = graph_memory.get_all_entities(limit=100)
        for entity in all_entities:
            assert not entity.properties.get("_deleted")

    def test_get_all_relations(self, graph_memory):
        """Test getting all relations."""
        graph_memory.add_entity(Entity(id="s1", entity_type="tool", name="s1"))
        graph_memory.add_entity(Entity(id="t1", entity_type="tool", name="t1"))
        graph_memory.add_entity(Entity(id="s2", entity_type="tool", name="s2"))
        graph_memory.add_entity(Entity(id="t2", entity_type="tool", name="t2"))

        graph_memory.add_relation(
            Relation(id="r1", source_id="s1", target_id="t1", relation_type="uses")
        )
        graph_memory.add_relation(
            Relation(id="r2", source_id="s2", target_id="t2", relation_type="uses")
        )

        all_relations = graph_memory.get_all_relations(limit=100)
        assert len(all_relations) >= 2


# ============== User Preferences Tests ==============


class TestUserPreferences:
    """Tests for user preference retrieval."""

    def test_get_user_preferences_basic(self, graph_memory):
        """Test getting user preferences."""
        graph_memory.add_entity(Entity(id="user", entity_type="person", name="user"))
        graph_memory.add_entity(
            Entity(id="pref_tool", entity_type="tool", name="httpx")
        )
        graph_memory.add_entity(
            Entity(id="disl_tool", entity_type="tool", name="requests")
        )

        graph_memory.add_relation(
            Relation(
                id="r_pref",
                source_id="user",
                target_id="pref_tool",
                relation_type="prefers",
                confidence=0.9,
            )
        )
        graph_memory.add_relation(
            Relation(
                id="r_disl",
                source_id="user",
                target_id="disl_tool",
                relation_type="dislikes",
                confidence=0.8,
            )
        )

        prefs = graph_memory.get_user_preferences("user")
        assert len(prefs["prefers"]) >= 1
        assert len(prefs["dislikes"]) >= 1
        assert prefs["prefers"][0]["name"] == "httpx"

    def test_get_user_preferences_only_prefers(self, graph_memory):
        """Test user with only preferences."""
        graph_memory.add_entity(Entity(id="user", entity_type="person", name="user"))
        graph_memory.add_entity(Entity(id="pref", entity_type="tool", name="pytest"))

        graph_memory.add_relation(
            Relation(
                id="r", source_id="user", target_id="pref", relation_type="prefers"
            )
        )

        prefs = graph_memory.get_user_preferences("user")
        assert len(prefs["prefers"]) >= 1
        assert len(prefs["dislikes"]) == 0

    def test_get_user_preferences_only_dislikes(self, graph_memory):
        """Test user with only dislikes."""
        graph_memory.add_entity(Entity(id="user", entity_type="person", name="user"))
        graph_memory.add_entity(Entity(id="disl", entity_type="tool", name="requests"))

        graph_memory.add_relation(
            Relation(
                id="r", source_id="user", target_id="disl", relation_type="dislikes"
            )
        )

        prefs = graph_memory.get_user_preferences("user")
        assert len(prefs["prefers"]) == 0
        assert len(prefs["dislikes"]) >= 1

    def test_get_user_preferences_empty(self, graph_memory):
        """Test user with no preferences."""
        graph_memory.add_entity(Entity(id="user", entity_type="person", name="user"))

        prefs = graph_memory.get_user_preferences("user")
        assert prefs["prefers"] == []
        assert prefs["dislikes"] == []

    def test_get_user_preferences_includes_confidence(self, graph_memory):
        """Test that preferences include confidence."""
        graph_memory.add_entity(Entity(id="user", entity_type="person", name="user"))
        graph_memory.add_entity(Entity(id="tool", entity_type="tool", name="httpx"))

        graph_memory.add_relation(
            Relation(
                id="r",
                source_id="user",
                target_id="tool",
                relation_type="prefers",
                confidence=0.95,
            )
        )

        prefs = graph_memory.get_user_preferences("user")
        if prefs["prefers"]:
            assert prefs["prefers"][0]["confidence"] == 0.95


# ============== Clear and Close Tests ==============


class TestClearAndClose:
    """Tests for clear and close operations."""

    def test_clear_soft_delete_all(self, graph_memory):
        """Test that clear marks all entities as deleted."""
        for i in range(3):
            graph_memory.add_entity(
                Entity(id=f"clear_e{i}", entity_type="tool", name=f"tool{i}")
            )

        graph_memory.clear()

        # All entities should be marked deleted
        all_entities = graph_memory.get_all_entities(limit=100)
        for entity in all_entities:
            assert entity.properties.get("_deleted") == True

    def test_close_no_error(self, graph_memory):
        """Test close method (no-op)."""
        # Should not raise
        graph_memory.close()


# ============== Edge Cases ==============


class TestGraphEdgeCases:
    """Tests for edge cases in graph operations."""

    def test_entity_empty_name(self, graph_memory):
        """Test entity with empty name."""
        entity = Entity(id="e_empty", entity_type="tool", name="")
        graph_memory.add_entity(entity)

        retrieved = graph_memory.get_entity("e_empty")
        assert retrieved.name == ""

    def test_entity_special_chars_in_name(self, graph_memory):
        """Test entity with special characters in name."""
        entity = Entity(
            id="e_special",
            entity_type="tool",
            name="httpx-client/v2.0",
            properties={"unicode": "日本語"},
        )
        graph_memory.add_entity(entity)

        retrieved = graph_memory.get_entity("e_special")
        assert "日本語" in retrieved.properties["unicode"]

    def test_relation_between_nonexistent_entities(self, graph_memory):
        """Test adding relation between non-existent entities."""
        # Graph should still store the relation (entities may be added later)
        relation = Relation(
            id="r_noent",
            source_id="missing1",
            target_id="missing2",
            relation_type="related_to",
        )
        graph_memory.add_relation(relation)

        # Relation should exist
        retrieved = graph_memory.get_relation("r_noent")
        assert retrieved is not None

    def test_self_relation(self, graph_memory):
        """Test relation where source equals target."""
        graph_memory.add_entity(Entity(id="self", entity_type="tool", name="self"))

        relation = Relation(
            id="r_self",
            source_id="self",
            target_id="self",
            relation_type="related_to",
        )
        graph_memory.add_relation(relation)

        # Should handle circular reference
        relations = graph_memory.get_relations_from("self")
        assert len(relations) >= 1

    def test_multiple_relations_same_type(self, graph_memory):
        """Test multiple relations of same type between entities."""
        graph_memory.add_entity(Entity(id="s", entity_type="person", name="s"))
        graph_memory.add_entity(Entity(id="t1", entity_type="tool", name="t1"))
        graph_memory.add_entity(Entity(id="t2", entity_type="tool", name="t2"))

        graph_memory.add_relation(
            Relation(id="r1", source_id="s", target_id="t1", relation_type="prefers")
        )
        graph_memory.add_relation(
            Relation(id="r2", source_id="s", target_id="t2", relation_type="prefers")
        )

        prefers = graph_memory.get_relations_from("s", relation_type="prefers")
        assert len(prefers) == 2

    def test_large_properties_dict(self, graph_memory):
        """Test entity with large properties dict."""
        large_props = {f"key_{i}": f"value_{i}" for i in range(100)}
        entity = Entity(
            id="e_large",
            entity_type="tool",
            name="large_props",
            properties=large_props,
        )
        graph_memory.add_entity(entity)

        retrieved = graph_memory.get_entity("e_large")
        assert len(retrieved.properties) == 100

    def test_entity_confidence_boundary(self, graph_memory):
        """Test entity confidence at boundaries."""
        entity_0 = Entity(id="e_0", entity_type="tool", name="zero", confidence=0.0)
        entity_1 = Entity(id="e_1", entity_type="tool", name="one", confidence=1.0)

        graph_memory.add_entity(entity_0)
        graph_memory.add_entity(entity_1)

        retrieved_0 = graph_memory.get_entity("e_0")
        retrieved_1 = graph_memory.get_entity("e_1")

        assert retrieved_0.confidence == 0.0
        assert retrieved_1.confidence == 1.0

    def test_get_all_entities_empty_graph(self, graph_memory):
        """Test get_all_entities on empty graph."""
        entities = graph_memory.get_all_entities(limit=100)
        assert len(entities) == 0

    def test_get_all_relations_empty_graph(self, graph_memory):
        """Test get_all_relations on empty graph."""
        relations = graph_memory.get_all_relations(limit=100)
        assert len(relations) == 0
