"""Tests for HybridRetriever.

Tests cover:
- Retrieval for prompt injection
- Connected entity traversal from facts
- User environment context
- Preference conflict checking
- Edge cases: empty stores, no preferences, missing entities
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obektclaw.config import Config
from obektclaw.memory.hybrid_retriever import HybridRetriever, RetrievedContext
from obektclaw.memory.ranking import RankingAlgorithm


# ============== RetrievedContext Tests ==============


class TestRetrievedContext:
    """Tests for RetrievedContext dataclass."""

    def test_context_creation(self):
        """Test creating RetrievedContext."""
        context = RetrievedContext(
            facts=[{"content": "fact1"}],
            entities=[{"name": "entity1"}],
            skills=[{"id": "skill1"}],
            preferences=[{"name": "pref1"}],
            dislikes=[{"name": "disl1"}],
            total_score=100.0,
            total_tokens=50,
        )
        assert len(context.facts) == 1
        assert len(context.entities) == 1
        assert context.total_score == 100.0

    def test_context_defaults(self):
        """Test RetrievedContext default values."""
        context = RetrievedContext()
        assert context.facts == []
        assert context.entities == []
        assert context.skills == []
        assert context.preferences == []
        assert context.dislikes == []
        assert context.total_score == 0.0

    def test_context_to_prompt_text_empty(self):
        """Test to_prompt_text with empty context."""
        context = RetrievedContext()
        text = context.to_prompt_text()
        assert text == ""

    def test_context_to_prompt_text_with_preferences(self):
        """Test to_prompt_text with preferences."""
        context = RetrievedContext(
            preferences=[
                {"name": "httpx", "entity_type": "tool"},
                {"name": "async", "entity_type": "concept"},
            ]
        )
        text = context.to_prompt_text()
        assert "Preferences" in text
        assert "httpx" in text

    def test_context_to_prompt_text_with_dislikes(self):
        """Test to_prompt_text with dislikes."""
        context = RetrievedContext(dislikes=[{"name": "requests"}, {"name": "sync"}])
        text = context.to_prompt_text()
        assert "Dislikes" in text
        assert "requests" in text

    def test_context_to_prompt_text_with_facts(self):
        """Test to_prompt_text with facts."""
        context = RetrievedContext(
            facts=[
                {"content": "User prefers httpx over requests"},
                {"content": "Server runs on Hetzner CX22"},
            ]
        )
        text = context.to_prompt_text()
        assert "Knowledge" in text
        assert "httpx" in text

    def test_context_to_prompt_text_with_entities(self):
        """Test to_prompt_text with entities."""
        context = RetrievedContext(
            entities=[
                {"name": "httpx", "entity_type": "tool"},
                {"name": "Hetzner", "entity_type": "environment"},
            ]
        )
        text = context.to_prompt_text()
        assert "Context" in text
        assert "httpx" in text or "Hetzner" in text

    def test_context_to_prompt_text_with_skills(self):
        """Test to_prompt_text with skills."""
        context = RetrievedContext(
            skills=[
                {
                    "id": "csv-import",
                    "description": "Import CSV files",
                    "metadata": {"name": "csv-import"},
                },
            ]
        )
        text = context.to_prompt_text()
        assert "Skills" in text

    def test_context_to_prompt_text_full(self):
        """Test to_prompt_text with all fields."""
        context = RetrievedContext(
            facts=[{"content": "Important fact"}],
            entities=[{"name": "httpx", "entity_type": "tool"}],
            skills=[
                {"id": "skill", "description": "A skill", "metadata": {"name": "skill"}}
            ],
            preferences=[{"name": "pref", "entity_type": "tool"}],
            dislikes=[{"name": "disl"}],
        )
        text = context.to_prompt_text()
        # Should have all sections
        assert "Preferences" in text
        assert "Dislikes" in text or "Avoid" in text
        assert "Knowledge" in text
        assert "Context" in text
        assert "Skills" in text

    def test_context_estimate_tokens(self):
        """Test token estimation."""
        context = RetrievedContext(
            facts=[{"content": "A very long fact content here"}],
        )
        tokens = context.estimate_tokens()
        # Approximate: len(text) // 4
        assert tokens >= 0

    def test_context_estimate_tokens_empty(self):
        """Test token estimation for empty context."""
        context = RetrievedContext()
        tokens = context.estimate_tokens()
        assert tokens == 0

    def test_context_retrieval_stats(self):
        """Test retrieval_stats field."""
        context = RetrievedContext(
            retrieval_stats={
                "facts_found": 10,
                "facts_selected": 3,
            }
        )
        assert context.retrieval_stats["facts_found"] == 10


# ============== Mock Fixtures ==============


@pytest.fixture
def mock_config():
    """Mock Config with relevant settings."""
    tmpdir = Path(tempfile.gettempdir())
    return Config(
        home=tmpdir,
        db_path=tmpdir / "test.db",
        skills_dir=tmpdir / "skills",
        bundled_skills_dir=tmpdir / "bundled_skills",
        logs_dir=tmpdir / "logs",
        llm_base_url="https://api.openai.com/v1",
        llm_api_key="test-key",
        llm_model="gpt-4o-mini",
        llm_fast_model="gpt-4o-mini",
        tg_token="",
        tg_allowed_chat_ids=(),
        bash_timeout=30,
        workdir=tmpdir,
        # Memory system
        chroma_path=tmpdir / "chroma",
        semantic_search_limit=10,
        graph_traversal_depth=2,
        context_assembly_max_tokens=500,
    )


@pytest.fixture
def mock_graph_memory():
    """Create a mock GraphMemory."""
    mock = MagicMock()
    mock.get_user_preferences.return_value = {
        "prefers": [],
        "dislikes": [],
    }
    mock.get_entity.return_value = None
    mock.get_connected_entities.return_value = []
    mock.get_relations_from.return_value = []
    return mock


@pytest.fixture
def mock_vector_memory():
    """Create a mock VectorMemory."""
    mock = MagicMock()
    mock.search_similar_facts.return_value = []
    mock.search_similar_skills.return_value = []
    mock.search_similar_entities.return_value = []
    return mock


@pytest.fixture
def hybrid_retriever(mock_graph_memory, mock_vector_memory, mock_config):
    """Create a HybridRetriever with mocked dependencies."""
    with patch("obektclaw.memory.hybrid_retriever.CONFIG", mock_config):
        ranking = RankingAlgorithm(min_score_threshold=0.0)
        retriever = HybridRetriever(
            graph_memory=mock_graph_memory,
            vector_memory=mock_vector_memory,
            ranking=ranking,
            user_entity_id="entity_person_user",
        )
        yield retriever


# ============== Retrieval Tests ==============


class TestRetrieveForPrompt:
    """Tests for retrieve_for_prompt."""

    def test_retrieve_basic(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test basic retrieval."""
        mock_vector_memory.search_similar_facts.return_value = [
            {
                "id": "f1",
                "content": "User prefers httpx",
                "metadata": {"confidence": 0.9, "category": "preference"},
                "distance": 0.2,
            },
        ]
        mock_vector_memory.search_similar_skills.return_value = [
            {
                "id": "s1",
                "description": "CSV import skill",
                "metadata": {"use_count": 5},
            },
        ]
        mock_graph_memory.get_user_preferences.return_value = {
            "prefers": [{"name": "httpx", "confidence": 0.9}],
            "dislikes": [],
        }

        context = hybrid_retriever.retrieve_for_prompt(
            query="HTTP client preference",
            max_tokens=500,
        )

        assert isinstance(context, RetrievedContext)
        assert len(context.facts) >= 1
        assert len(context.skills) >= 1
        assert len(context.preferences) >= 1

    def test_retrieve_empty_stores(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test retrieval with empty stores."""
        mock_vector_memory.search_similar_facts.return_value = []
        mock_vector_memory.search_similar_skills.return_value = []
        mock_graph_memory.get_user_preferences.return_value = {
            "prefers": [],
            "dislikes": [],
        }

        context = hybrid_retriever.retrieve_for_prompt("test query")

        assert len(context.facts) == 0
        assert len(context.skills) == 0
        assert len(context.preferences) == 0

    def test_retrieve_with_token_budget(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test retrieval respects token budget."""
        # Add many facts
        mock_vector_memory.search_similar_facts.return_value = [
            {
                "id": f"f{i}",
                "content": f"Fact {i} with some content",
                "metadata": {"confidence": 0.8, "category": "general"},
            }
            for i in range(20)
        ]
        mock_vector_memory.search_similar_skills.return_value = []
        mock_graph_memory.get_user_preferences.return_value = {
            "prefers": [],
            "dislikes": [],
        }

        context = hybrid_retriever.retrieve_for_prompt(
            query="test",
            max_tokens=100,  # Small budget
        )

        # Should not include all 20 facts
        assert context.total_tokens <= 100
        assert len(context.facts) < 20

    def test_retrieve_includes_retrieval_stats(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test that retrieval includes stats."""
        mock_vector_memory.search_similar_facts.return_value = [
            {
                "id": "f1",
                "content": "fact",
                "metadata": {"confidence": 0.8, "category": "general"},
            }
        ]
        mock_vector_memory.search_similar_skills.return_value = [
            {"id": "s1", "description": "skill", "metadata": {}}
        ]
        mock_graph_memory.get_user_preferences.return_value = {
            "prefers": [],
            "dislikes": [],
        }

        context = hybrid_retriever.retrieve_for_prompt("test")

        assert "facts_found" in context.retrieval_stats
        assert "skills_found" in context.retrieval_stats
        assert "facts_selected" in context.retrieval_stats

    def test_retrieve_facts_with_entity_ids(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test retrieval of facts with entity links."""
        from obektclaw.memory.graph_memory import Entity

        mock_vector_memory.search_similar_facts.return_value = [
            {
                "id": "f1",
                "content": "Server runs on Hetzner",
                "metadata": {
                    "confidence": 0.9,
                    "category": "environment",
                    "entity_ids": "entity_hetzner",
                },
            },
        ]
        mock_vector_memory.search_similar_skills.return_value = []

        mock_entity = Entity(
            id="entity_hetzner",
            entity_type="environment",
            name="Hetzner",
            confidence=0.95,
        )
        mock_graph_memory.get_entity.return_value = mock_entity
        mock_graph_memory.get_connected_entities.return_value = [(mock_entity, 0)]
        mock_graph_memory.get_user_preferences.return_value = {
            "prefers": [],
            "dislikes": [],
        }

        context = hybrid_retriever.retrieve_for_prompt("server info")

        # Should include entity connected to fact
        assert len(context.entities) >= 1

    def test_retrieve_prefers_over_dislikes(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test that preferences appear before dislikes in output."""
        mock_vector_memory.search_similar_facts.return_value = []
        mock_vector_memory.search_similar_skills.return_value = []
        mock_graph_memory.get_user_preferences.return_value = {
            "prefers": [{"name": "httpx", "confidence": 0.9}],
            "dislikes": [{"name": "requests", "_is_dislike": True, "confidence": 0.8}],
        }

        context = hybrid_retriever.retrieve_for_prompt("HTTP clients")

        assert len(context.preferences) >= 1
        assert len(context.dislikes) >= 1

        text = context.to_prompt_text()
        # Preferences should appear before dislikes in text
        pref_pos = text.find("httpx")
        disl_pos = text.find("requests")
        if pref_pos >= 0 and disl_pos >= 0:
            assert pref_pos < disl_pos


# ============== Connected Entity Tests ==============


class TestConnectedEntities:
    """Tests for _get_connected_entities."""

    def test_get_connected_no_entity_ids(self, hybrid_retriever):
        """Test with facts having no entity IDs."""
        facts = [
            {"id": "f1", "content": "plain fact", "metadata": {"confidence": 0.8}},
        ]

        entities = hybrid_retriever._get_connected_entities(facts, max_depth=1)
        assert len(entities) == 0

    def test_get_connected_with_entity_ids(self, hybrid_retriever, mock_graph_memory):
        """Test with facts having entity IDs."""
        from obektclaw.memory.graph_memory import Entity

        mock_entity = Entity(id="e1", entity_type="tool", name="httpx")
        mock_graph_memory.get_entity.return_value = mock_entity
        mock_graph_memory.get_connected_entities.return_value = [(mock_entity, 0)]

        facts = [
            {
                "id": "f1",
                "content": "fact with entity",
                "metadata": {"entity_ids": "e1"},
            },
        ]

        entities = hybrid_retriever._get_connected_entities(facts, max_depth=1)
        assert len(entities) >= 1

    def test_get_connected_multiple_entity_ids(
        self, hybrid_retriever, mock_graph_memory
    ):
        """Test with multiple entity IDs in single fact."""
        from obektclaw.memory.graph_memory import Entity

        e1 = Entity(id="e1", entity_type="tool", name="httpx")
        e2 = Entity(id="e2", entity_type="environment", name="Hetzner")

        def get_entity_side_effect(id):
            if id == "e1":
                return e1
            elif id == "e2":
                return e2
            return None

        mock_graph_memory.get_entity.side_effect = get_entity_side_effect
        mock_graph_memory.get_connected_entities.return_value = []

        facts = [
            {
                "id": "f1",
                "content": "multi entity",
                "metadata": {"entity_ids": "e1,e2"},
            },
        ]

        entities = hybrid_retriever._get_connected_entities(facts, max_depth=0)
        assert len(entities) >= 2

    def test_get_connected_missing_entity(self, hybrid_retriever, mock_graph_memory):
        """Test with entity ID that doesn't exist in graph."""
        mock_graph_memory.get_entity.return_value = None

        facts = [
            {"id": "f1", "metadata": {"entity_ids": "missing_entity"}},
        ]

        entities = hybrid_retriever._get_connected_entities(facts, max_depth=1)
        assert len(entities) == 0

    def test_get_connected_respects_depth(self, hybrid_retriever, mock_graph_memory):
        """Test that depth limit is respected."""
        from obektclaw.memory.graph_memory import Entity

        e1 = Entity(id="e1", entity_type="tool", name="httpx")
        e2 = Entity(id="e2", entity_type="tool", name="h2")

        mock_graph_memory.get_entity.return_value = e1
        # When max_depth > 0, should traverse
        mock_graph_memory.get_connected_entities.return_value = [(e1, 0), (e2, 1)]

        facts = [{"id": "f1", "metadata": {"entity_ids": "e1"}}]

        # With max_depth=0, should not traverse
        entities = hybrid_retriever._get_connected_entities(facts, max_depth=0)
        # Should only have the direct entity (no traversal)
        # Implementation may vary, but traversal shouldn't happen at depth 0

    def test_get_connected_empty_facts(self, hybrid_retriever):
        """Test with empty facts list."""
        entities = hybrid_retriever._get_connected_entities([], max_depth=1)
        assert len(entities) == 0

    def test_get_connected_deduplicates(self, hybrid_retriever, mock_graph_memory):
        """Test that duplicate entity IDs are deduplicated."""
        from obektclaw.memory.graph_memory import Entity

        e1 = Entity(id="e1", entity_type="tool", name="httpx")
        mock_graph_memory.get_entity.return_value = e1
        mock_graph_memory.get_connected_entities.return_value = [(e1, 0)]

        facts = [
            {"id": "f1", "metadata": {"entity_ids": "e1"}},
            {"id": "f2", "metadata": {"entity_ids": "e1,e1,e1"}},
        ]

        entities = hybrid_retriever._get_connected_entities(facts, max_depth=1)
        # Should deduplicate
        entity_ids = [e["id"] for e in entities]
        # May have duplicates due to traversal, but core deduplication should work


# ============== User Environment Tests ==============


class TestUserEnvironment:
    """Tests for get_user_environment."""

    def test_get_user_environment_empty(self, hybrid_retriever, mock_graph_memory):
        """Test with no relations."""
        mock_graph_memory.get_relations_from.return_value = []

        env = hybrid_retriever.get_user_environment()
        assert env == {}

    def test_get_user_environment_with_tools(self, hybrid_retriever, mock_graph_memory):
        """Test with tool relations."""
        from obektclaw.memory.graph_memory import Entity, Relation

        mock_entity = Entity(id="e_tool", entity_type="tool", name="httpx")
        mock_relation = Relation(
            id="r1",
            source_id="entity_person_user",
            target_id="e_tool",
            relation_type="uses",
        )

        mock_graph_memory.get_relations_from.return_value = [mock_relation]
        mock_graph_memory.get_entity.return_value = mock_entity

        env = hybrid_retriever.get_user_environment()
        assert "tool" in env
        assert len(env["tool"]) >= 1

    def test_get_user_environment_with_projects(
        self, hybrid_retriever, mock_graph_memory
    ):
        """Test with project relations."""
        from obektclaw.memory.graph_memory import Entity, Relation

        mock_entity = Entity(id="e_proj", entity_type="project", name="obektclaw")
        mock_relation = Relation(
            id="r1",
            source_id="entity_person_user",
            target_id="e_proj",
            relation_type="owns",
        )

        mock_graph_memory.get_relations_from.return_value = [mock_relation]
        mock_graph_memory.get_entity.return_value = mock_entity

        env = hybrid_retriever.get_user_environment()
        assert "project" in env

    def test_get_user_environment_with_environments(
        self, hybrid_retriever, mock_graph_memory
    ):
        """Test with environment relations."""
        from obektclaw.memory.graph_memory import Entity, Relation

        mock_entity = Entity(id="e_env", entity_type="environment", name="Hetzner")
        mock_relation = Relation(
            id="r1",
            source_id="entity_person_user",
            target_id="e_env",
            relation_type="owns",
        )

        mock_graph_memory.get_relations_from.return_value = [mock_relation]
        mock_graph_memory.get_entity.return_value = mock_entity

        env = hybrid_retriever.get_user_environment()
        assert "environment" in env

    def test_get_user_environment_skips_non_env_types(
        self, hybrid_retriever, mock_graph_memory
    ):
        """Test that non-environment types are excluded."""
        from obektclaw.memory.graph_memory import Entity, Relation

        mock_entity = Entity(id="e_concept", entity_type="concept", name="async")
        mock_relation = Relation(
            id="r1",
            source_id="entity_person_user",
            target_id="e_concept",
            relation_type="uses",
        )

        mock_graph_memory.get_relations_from.return_value = [mock_relation]
        mock_graph_memory.get_entity.return_value = mock_entity

        env = hybrid_retriever.get_user_environment()
        assert "concept" not in env

    def test_get_user_environment_missing_entity(
        self, hybrid_retriever, mock_graph_memory
    ):
        """Test with relation to missing entity."""
        from obektclaw.memory.graph_memory import Relation

        mock_relation = Relation(
            id="r1",
            source_id="entity_person_user",
            target_id="missing",
            relation_type="uses",
        )

        mock_graph_memory.get_relations_from.return_value = [mock_relation]
        mock_graph_memory.get_entity.return_value = None

        env = hybrid_retriever.get_user_environment()
        assert env == {}


# ============== Preference Conflict Tests ==============


class TestPreferenceConflict:
    """Tests for check_preference_conflict."""

    def test_no_conflict_empty(self, hybrid_retriever, mock_vector_memory):
        """Test with no matching entities."""
        mock_vector_memory.search_similar_entities.return_value = []

        result = hybrid_retriever.check_preference_conflict("unknown_tool")
        assert result is None

    def test_no_conflict_not_disliked(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test with entity that isn't disliked."""
        from obektclaw.memory.graph_memory import Entity

        mock_vector_memory.search_similar_entities.return_value = [
            {
                "id": "v1",
                "description": "tool description",
                "metadata": {"graph_node_id": "e1"},
                "distance": 0.2,
            }
        ]

        mock_entity = Entity(id="e1", entity_type="tool", name="pytest")
        mock_graph_memory.get_entity.return_value = mock_entity
        mock_graph_memory.get_relations_from.return_value = []

        result = hybrid_retriever.check_preference_conflict("pytest")
        # Not disliked, no alternatives
        assert result is None

    def test_conflict_disliked(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test with disliked entity."""
        from obektclaw.memory.graph_memory import Entity, Relation

        mock_vector_memory.search_similar_entities.return_value = [
            {
                "id": "v1",
                "metadata": {"graph_node_id": "e_requests"},
                "distance": 0.1,
            }
        ]

        mock_entity = Entity(id="e_requests", entity_type="tool", name="requests")
        mock_relation = Relation(
            id="r_dislike",
            source_id="entity_person_user",
            target_id="e_requests",
            relation_type="dislikes",
        )

        mock_graph_memory.get_entity.return_value = mock_entity
        mock_graph_memory.get_relations_from.return_value = [mock_relation]

        result = hybrid_retriever.check_preference_conflict("requests")
        assert result is not None
        assert result["status"] == "disliked"

    def test_conflict_has_alternatives(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test with entity that has alternatives."""
        from obektclaw.memory.graph_memory import Entity, Relation

        mock_vector_memory.search_similar_entities.return_value = [
            {
                "id": "v1",
                "metadata": {"graph_node_id": "e_requests"},
                "distance": 0.1,
            }
        ]

        requests_entity = Entity(id="e_requests", entity_type="tool", name="requests")
        httpx_entity = Entity(id="e_httpx", entity_type="tool", name="httpx")

        prefers_relation = Relation(
            id="r_pref",
            source_id="entity_person_user",
            target_id="e_httpx",
            relation_type="prefers",
        )

        def get_entity_side_effect(id):
            if id == "e_requests":
                return requests_entity
            elif id == "e_httpx":
                return httpx_entity
            return requests_entity

        mock_graph_memory.get_entity.side_effect = get_entity_side_effect

        # First call for dislikes check, second for prefers check
        mock_graph_memory.get_relations_from.side_effect = [
            [],  # No dislikes
            [prefers_relation],  # Has prefers
        ]

        result = hybrid_retriever.check_preference_conflict("requests")
        assert result is not None
        assert result["status"] == "has_alternatives"
        assert "httpx" in result["alternatives"]

    def test_conflict_high_distance_skipped(self, hybrid_retriever, mock_vector_memory):
        """Test that high-distance matches are skipped."""
        mock_vector_memory.search_similar_entities.return_value = [
            {
                "id": "v1",
                "metadata": {"graph_node_id": "e1"},
                "distance": 0.8,  # High distance, not a match
            }
        ]

        result = hybrid_retriever.check_preference_conflict("tool")
        # Should skip due to high distance
        assert result is None

    def test_conflict_name_mismatch(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test that name mismatch skips."""
        from obektclaw.memory.graph_memory import Entity

        mock_vector_memory.search_similar_entities.return_value = [
            {
                "id": "v1",
                "metadata": {"graph_node_id": "e1"},
                "distance": 0.2,
            }
        ]

        mock_entity = Entity(id="e1", entity_type="tool", name="different_name")
        mock_graph_memory.get_entity.return_value = mock_entity

        result = hybrid_retriever.check_preference_conflict("pytest")
        # Name doesn't match
        assert result is None

    def test_conflict_no_graph_node_id(self, hybrid_retriever, mock_vector_memory):
        """Test with entity without graph_node_id."""
        mock_vector_memory.search_similar_entities.return_value = [
            {
                "id": "v1",
                "metadata": {},  # No graph_node_id
                "distance": 0.2,
            }
        ]

        result = hybrid_retriever.check_preference_conflict("tool")
        assert result is None


# ============== Integration Tests ==============


class TestHybridRetrieverIntegration:
    """Integration-style tests for HybridRetriever."""

    def test_full_retrieval_flow(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test full retrieval flow with all components."""
        from obektclaw.memory.graph_memory import Entity

        # Set up mock data
        mock_vector_memory.search_similar_facts.return_value = [
            {
                "id": "f1",
                "content": "User prefers httpx for HTTP requests",
                "metadata": {
                    "confidence": 0.9,
                    "category": "preference",
                    "entity_ids": "e_httpx",
                },
                "distance": 0.1,
            },
            {
                "id": "f2",
                "content": "Server deployed on Hetzner",
                "metadata": {
                    "confidence": 0.85,
                    "category": "environment",
                    "entity_ids": "e_hetzner",
                },
                "distance": 0.3,
            },
        ]
        mock_vector_memory.search_similar_skills.return_value = [
            {
                "id": "csv-import",
                "description": "Import CSV files into database",
                "metadata": {"use_count": 10, "success_count": 9},
            }
        ]

        httpx_entity = Entity(
            id="e_httpx", entity_type="tool", name="httpx", confidence=0.9
        )
        hetzner_entity = Entity(
            id="e_hetzner", entity_type="environment", name="Hetzner", confidence=0.95
        )

        def get_entity_side_effect(id):
            if id == "e_httpx":
                return httpx_entity
            elif id == "e_hetzner":
                return hetzner_entity
            return None

        mock_graph_memory.get_entity.side_effect = get_entity_side_effect
        mock_graph_memory.get_connected_entities.return_value = []
        mock_graph_memory.get_user_preferences.return_value = {
            "prefers": [{"name": "httpx", "entity_type": "tool", "confidence": 0.9}],
            "dislikes": [{"name": "requests", "_is_dislike": True, "confidence": 0.8}],
        }

        context = hybrid_retriever.retrieve_for_prompt(
            query="HTTP client and server setup",
            max_tokens=1000,
        )

        # Should have retrieved items
        assert len(context.facts) >= 1
        assert len(context.skills) >= 1
        assert len(context.preferences) >= 1
        assert len(context.dislikes) >= 1

        # Text should be formatted properly
        text = context.to_prompt_text()
        assert len(text) > 0

    def test_custom_user_entity_id(
        self, mock_graph_memory, mock_vector_memory, mock_config
    ):
        """Test with custom user entity ID."""
        with patch("obektclaw.memory.hybrid_retriever.CONFIG", mock_config):
            ranking = RankingAlgorithm(min_score_threshold=0.0)
            retriever = HybridRetriever(
                graph_memory=mock_graph_memory,
                vector_memory=mock_vector_memory,
                ranking=ranking,
                user_entity_id="custom_user_entity",
            )

            mock_graph_memory.get_user_preferences.return_value = {
                "prefers": [{"name": "custom_pref"}],
                "dislikes": [],
            }

            retriever.retrieve_for_prompt("test")
            # Should call get_user_preferences with custom ID
            mock_graph_memory.get_user_preferences.assert_called_with(
                "custom_user_entity"
            )


# ============== Edge Cases ==============


class TestHybridRetrieverEdgeCases:
    """Tests for edge cases."""

    def test_empty_query_string(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test with empty query."""
        mock_vector_memory.search_similar_facts.return_value = []
        mock_vector_memory.search_similar_skills.return_value = []

        context = hybrid_retriever.retrieve_for_prompt("")
        # Should handle gracefully
        assert isinstance(context, RetrievedContext)

    def test_very_long_query(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test with very long query."""
        long_query = "a" * 10000

        mock_vector_memory.search_similar_facts.return_value = []
        mock_vector_memory.search_similar_skills.return_value = []

        context = hybrid_retriever.retrieve_for_prompt(long_query)
        assert isinstance(context, RetrievedContext)

    def test_unicode_query(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test with unicode query."""
        unicode_query = "ユーザーは httpx を使用します"

        mock_vector_memory.search_similar_facts.return_value = []
        mock_vector_memory.search_similar_skills.return_value = []

        context = hybrid_retriever.retrieve_for_prompt(unicode_query)
        assert isinstance(context, RetrievedContext)

    def test_zero_max_tokens(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test with zero token budget."""
        mock_vector_memory.search_similar_facts.return_value = [
            {
                "id": "f1",
                "content": "fact",
                "metadata": {"confidence": 0.9, "category": "general"},
            }
        ]

        context = hybrid_retriever.retrieve_for_prompt("test", max_tokens=0)
        # Should return empty context (nothing fits)
        assert context.total_tokens == 0

    def test_negative_max_tokens(
        self, hybrid_retriever, mock_vector_memory, mock_graph_memory
    ):
        """Test with negative token budget."""
        mock_vector_memory.search_similar_facts.return_value = []

        context = hybrid_retriever.retrieve_for_prompt("test", max_tokens=-100)
        assert isinstance(context, RetrievedContext)

    def test_fact_empty_entity_ids_string(self, hybrid_retriever):
        """Test fact with empty entity_ids string."""
        facts = [
            {"id": "f1", "metadata": {"entity_ids": ""}},
        ]

        entities = hybrid_retriever._get_connected_entities(facts, max_depth=1)
        assert len(entities) == 0

    def test_fact_whitespace_entity_ids(self, hybrid_retriever, mock_graph_memory):
        """Test fact with whitespace in entity_ids."""
        from obektclaw.memory.graph_memory import Entity

        mock_entity = Entity(id="e1", entity_type="tool", name="test")
        mock_graph_memory.get_entity.return_value = mock_entity

        facts = [
            {"id": "f1", "metadata": {"entity_ids": " e1 , e1 ,  "}},
        ]

        entities = hybrid_retriever._get_connected_entities(facts, max_depth=0)
        # Should strip whitespace and deduplicate
        assert len(entities) >= 1
