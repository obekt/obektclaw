"""Tests for RankingAlgorithm.

Tests cover:
- Scoring for facts, entities, skills, preferences
- Recency decay computation
- Rank and select with token budget
- Diversity penalty
- Edge cases: zero scores, empty lists, max tokens
"""

import math
from datetime import datetime, timedelta

import pytest

from obektclaw.memory.ranking import (
    RankingAlgorithm,
    ScoredItem,
    CATEGORY_PRIORITY,
    ENTITY_PRIORITY,
)


# ============== ScoredItem Tests ==============


class TestScoredItem:
    """Tests for ScoredItem dataclass."""

    def test_scored_item_creation(self):
        """Test creating a ScoredItem."""
        item = ScoredItem(
            item={"content": "test"},
            score=75.0,
            token_estimate=20,
            source="fact",
        )
        assert item.item == {"content": "test"}
        assert item.score == 75.0
        assert item.token_estimate == 20
        assert item.source == "fact"


# ============== Fact Scoring Tests ==============


class TestFactScoring:
    """Tests for fact scoring."""

    def test_score_fact_basic(self):
        """Test basic fact scoring."""
        ranking = RankingAlgorithm()
        fact = {
            "content": "User prefers httpx",
            "metadata": {
                "category": "preference",
                "confidence": 0.9,
                "created_at": datetime.utcnow().isoformat(),
            },
        }

        scored = ranking.score_fact(fact)
        assert scored.score > 0
        assert scored.source == "fact"
        assert scored.token_estimate > 0

    def test_score_fact_with_distance(self):
        """Test fact scoring with explicit distance."""
        ranking = RankingAlgorithm()
        fact = {
            "content": "Test fact",
            "metadata": {"confidence": 0.8, "category": "general"},
            "distance": 0.3,  # Close match
        }

        scored = ranking.score_fact(fact, query_distance=0.3)
        # Distance 0.3 → similarity score = 40 * (1 - 0.3) = 28
        assert scored.score > 0

    def test_score_fact_high_distance(self):
        """Test fact scoring with high distance (poor match)."""
        ranking = RankingAlgorithm()
        fact = {
            "content": "Test fact",
            "metadata": {"confidence": 0.8, "category": "general"},
            "distance": 0.8,  # Poor match
        }

        scored = ranking.score_fact(fact)
        # Distance 0.8 → similarity score = 40 * (1 - 0.8) = 8
        assert scored.score > 0
        assert scored.score < 50  # Should be lower than good match

    def test_score_fact_zero_distance(self):
        """Test fact scoring with zero distance (perfect match)."""
        ranking = RankingAlgorithm()
        fact = {
            "content": "Perfect match",
            "metadata": {"confidence": 1.0, "category": "preference"},
            "distance": 0.0,
        }

        scored = ranking.score_fact(fact)
        # Distance 0 → similarity score = 40
        assert scored.score >= 40

    def test_score_fact_no_distance_field(self):
        """Test fact scoring without distance field."""
        ranking = RankingAlgorithm()
        fact = {
            "content": "No distance",
            "metadata": {"confidence": 0.8, "category": "general"},
        }

        scored = ranking.score_fact(fact)
        # Should use default similarity score of 20
        assert scored.score >= 20

    def test_score_fact_category_priority(self):
        """Test that different categories affect scoring."""
        ranking = RankingAlgorithm()

        preference_fact = {
            "content": "Preference",
            "metadata": {"confidence": 0.9, "category": "preference"},
        }
        general_fact = {
            "content": "General",
            "metadata": {"confidence": 0.9, "category": "general"},
        }

        pref_scored = ranking.score_fact(preference_fact)
        gen_scored = ranking.score_fact(general_fact)

        # Preference should score higher due to category priority
        assert pref_scored.score > gen_scored.score

    def test_score_fact_confidence_impact(self):
        """Test that confidence affects scoring."""
        ranking = RankingAlgorithm()

        high_conf = {
            "content": "High confidence",
            "metadata": {"confidence": 1.0, "category": "general"},
        }
        low_conf = {
            "content": "Low confidence",
            "metadata": {"confidence": 0.3, "category": "general"},
        }

        high_scored = ranking.score_fact(high_conf)
        low_scored = ranking.score_fact(low_conf)

        # High confidence should score higher
        assert high_scored.score > low_scored.score

    def test_score_fact_entity_connections(self):
        """Test that entity connections boost score."""
        ranking = RankingAlgorithm()

        no_connections = {
            "content": "No connections",
            "metadata": {"confidence": 0.8, "entity_ids": ""},
        }
        with_connections = {
            "content": "With connections",
            "metadata": {"confidence": 0.8, "entity_ids": "e1,e2,e3"},
        }

        no_conn_scored = ranking.score_fact(no_connections)
        with_conn_scored = ranking.score_fact(with_connections)

        # Connected fact should score higher
        assert with_conn_scored.score > no_conn_scored.score

    def test_score_fact_many_connections(self):
        """Test that many connections cap at max score."""
        ranking = RankingAlgorithm()

        fact = {
            "content": "Many connections",
            "metadata": {
                "confidence": 0.8,
                "entity_ids": "e1,e2,e3,e4,e5,e6,e7,e8,e9,e10",
            },
        }

        scored = ranking.score_fact(fact)
        # Connection score capped at 15
        # With 10 entities: min(15, 5 + 10*2) = min(15, 25) = 15
        assert scored.score > 0


class TestRecencyScoring:
    """Tests for recency score computation."""

    def test_recency_fresh_fact(self):
        """Test recency score for very recent fact."""
        ranking = RankingAlgorithm(recency_halflife_days=30.0)
        fact = {
            "content": "Fresh fact",
            "metadata": {
                "confidence": 0.8,
                "category": "general",
                "created_at": datetime.utcnow().isoformat(),
            },
        }

        scored = ranking.score_fact(fact)
        # Fresh fact should have near-maximum recency score (15)
        assert scored.score >= 35  # At least 20 (similarity) + 15 (recency)

    def test_recency_old_fact(self):
        """Test recency score for old fact."""
        ranking = RankingAlgorithm(recency_halflife_days=30.0)
        old_date = datetime.utcnow() - timedelta(days=60)
        fact = {
            "content": "Old fact",
            "metadata": {
                "confidence": 0.8,
                "category": "general",
                "created_at": old_date.isoformat(),
            },
        }

        scored = ranking.score_fact(fact)
        # 60 days with 30-day halflife: recency = 15 * 2^(-60/30) = 15 * 0.25 = 3.75
        # Total score = similarity (20) + confidence (16) + recency (3.75) + category (2) = ~41.75
        # Should still have positive score but recency component is low
        assert scored.score > 0
        # Compare with fresh fact - old fact should score lower due to recency decay
        fresh_fact = {
            "content": "Fresh fact",
            "metadata": {
                "confidence": 0.8,
                "category": "general",
                "created_at": datetime.utcnow().isoformat(),
            },
        }
        fresh_scored = ranking.score_fact(fresh_fact)
        assert scored.score < fresh_scored.score

    def test_recency_halflife_calculation(self):
        """Test exact halflife calculation."""
        ranking = RankingAlgorithm(recency_halflife_days=30.0)
        half_life_date = datetime.utcnow() - timedelta(days=30)

        recency = ranking._compute_recency_score(half_life_date.isoformat())
        # At exactly halflife, score should be half of max (15/2 = 7.5)
        expected = 15 * math.pow(2, -30 / 30)  # = 7.5
        assert abs(recency - expected) < 0.5

    def test_recency_no_timestamp(self):
        """Test recency score without timestamp."""
        ranking = RankingAlgorithm()
        fact = {
            "content": "No timestamp",
            "metadata": {"confidence": 0.8, "category": "general"},
        }

        scored = ranking.score_fact(fact)
        # Should get mid-range recency (7.5)
        assert scored.score > 0

    def test_recency_invalid_timestamp(self):
        """Test recency score with invalid timestamp."""
        ranking = RankingAlgorithm()
        fact = {
            "content": "Invalid timestamp",
            "metadata": {
                "confidence": 0.8,
                "category": "general",
                "created_at": "not-a-date",
            },
        }

        scored = ranking.score_fact(fact)
        # Should gracefully handle and give mid-range
        assert scored.score > 0

    def test_recency_custom_halflife(self):
        """Test custom halflife setting."""
        ranking = RankingAlgorithm(recency_halflife_days=7.0)  # 7-day halflife
        week_old = datetime.utcnow() - timedelta(days=7)

        recency = ranking._compute_recency_score(week_old.isoformat())
        # At halflife, should be 7.5
        expected = 15 * math.pow(2, -7 / 7)
        assert abs(recency - expected) < 0.5


# ============== Entity Scoring Tests ==============


class TestEntityScoring:
    """Tests for entity scoring."""

    def test_score_entity_basic(self):
        """Test basic entity scoring."""
        ranking = RankingAlgorithm()
        entity = {
            "id": "e1",
            "name": "httpx",
            "entity_type": "tool",
            "confidence": 0.9,
        }

        scored = ranking.score_entity(entity)
        assert scored.score > 0
        assert scored.source == "entity"

    def test_score_entity_type_priority(self):
        """Test that entity type affects scoring."""
        ranking = RankingAlgorithm()

        tool_entity = {"name": "httpx", "entity_type": "tool", "confidence": 0.9}
        concept_entity = {"name": "async", "entity_type": "concept", "confidence": 0.9}

        tool_scored = ranking.score_entity(tool_entity)
        concept_scored = ranking.score_entity(concept_entity)

        # Tool has higher priority than concept
        assert ENTITY_PRIORITY["tool"] > ENTITY_PRIORITY["concept"]
        assert tool_scored.score > concept_scored.score

    def test_score_entity_graph_distance(self):
        """Test graph proximity scoring."""
        ranking = RankingAlgorithm()

        close_entity = {"name": "close", "entity_type": "tool", "confidence": 0.9}
        far_entity = {"name": "far", "entity_type": "tool", "confidence": 0.9}

        close_scored = ranking.score_entity(close_entity, graph_distance=0)
        far_scored = ranking.score_entity(far_entity, graph_distance=3)

        # Closer entity should score higher
        assert close_scored.score > far_scored.score

    def test_score_entity_zero_distance(self):
        """Test entity at graph distance 0."""
        ranking = RankingAlgorithm()
        entity = {"name": "direct", "entity_type": "tool", "confidence": 0.9}

        scored = ranking.score_entity(entity, graph_distance=0)
        # Proximity score = max(0, 30 - 0*10) = 30
        assert scored.score >= 70  # type score (32) + proximity (30) + confidence (18)

    def test_score_entity_large_distance(self):
        """Test entity at large graph distance."""
        ranking = RankingAlgorithm()
        entity = {"name": "distant", "entity_type": "tool", "confidence": 0.9}

        scored = ranking.score_entity(entity, graph_distance=10)
        # Proximity score = max(0, 30 - 10*10) = max(0, -70) = 0
        assert scored.score > 0  # Still has type and confidence scores

    def test_score_entity_confidence_impact(self):
        """Test entity confidence scoring."""
        ranking = RankingAlgorithm()

        high_conf = {"name": "high", "entity_type": "tool", "confidence": 1.0}
        low_conf = {"name": "low", "entity_type": "tool", "confidence": 0.3}

        high_scored = ranking.score_entity(high_conf)
        low_scored = ranking.score_entity(low_conf)

        assert high_scored.score > low_scored.score


# ============== Skill Scoring Tests ==============


class TestSkillScoring:
    """Tests for skill scoring."""

    def test_score_skill_basic(self):
        """Test basic skill scoring."""
        ranking = RankingAlgorithm()
        skill = {
            "id": "s1",
            "description": "Import CSV files",
            "metadata": {"use_count": 5, "success_count": 4},
        }

        scored = ranking.score_skill(skill)
        assert scored.score > 0
        assert scored.source == "skill"

    def test_score_skill_with_distance(self):
        """Test skill scoring with distance."""
        ranking = RankingAlgorithm()
        skill = {
            "description": "CSV import",
            "metadata": {"use_count": 5},
            "distance": 0.2,
        }

        scored = ranking.score_skill(skill)
        # Distance 0.2 → similarity = 40 * (1 - 0.2) = 32
        assert scored.score > 0

    def test_score_skill_usage_frequency(self):
        """Test usage frequency boost."""
        ranking = RankingAlgorithm()

        unused_skill = {"description": "New skill", "metadata": {"use_count": 0}}
        popular_skill = {"description": "Popular", "metadata": {"use_count": 100}}

        unused_scored = ranking.score_skill(unused_skill)
        popular_scored = ranking.score_skill(popular_skill)

        # Popular skill should score higher
        assert popular_scored.score > unused_scored.score

    def test_score_skill_success_rate(self):
        """Test success rate scoring."""
        ranking = RankingAlgorithm()

        perfect_skill = {
            "description": "Perfect",
            "metadata": {"use_count": 10, "success_count": 10},
        }
        poor_skill = {
            "description": "Poor",
            "metadata": {"use_count": 10, "success_count": 2},
        }

        perfect_scored = ranking.score_skill(perfect_skill)
        poor_scored = ranking.score_skill(poor_skill)

        # Perfect skill should score higher
        assert perfect_scored.score > poor_scored.score

    def test_score_skill_zero_use_count(self):
        """Test skill with no usage."""
        ranking = RankingAlgorithm()
        skill = {
            "description": "Unused",
            "metadata": {"use_count": 0, "success_count": 0},
        }

        scored = ranking.score_skill(skill)
        # Should still get mid-range success score (10)
        assert scored.score > 0

    def test_score_skill_recency(self):
        """Test skill recency scoring."""
        ranking = RankingAlgorithm()

        new_skill = {
            "description": "New",
            "metadata": {
                "use_count": 5,
                "created_at": datetime.utcnow().isoformat(),
            },
        }
        old_skill = {
            "description": "Old",
            "metadata": {
                "use_count": 5,
                "created_at": (datetime.utcnow() - timedelta(days=60)).isoformat(),
            },
        }

        new_scored = ranking.score_skill(new_skill)
        old_scored = ranking.score_skill(old_skill)

        # New skill should score higher due to recency
        assert new_scored.score > old_scored.score


# ============== Preference Scoring Tests ==============


class TestPreferenceScoring:
    """Tests for preference scoring."""

    def test_score_preference_basic(self):
        """Test basic preference scoring."""
        ranking = RankingAlgorithm()
        preference = {"name": "httpx", "confidence": 0.9}

        scored = ranking.score_preference(preference)
        # Preference base score is 80 + confidence boost
        assert scored.score >= 80
        assert scored.source == "preference"

    def test_score_dislike_basic(self):
        """Test dislike scoring."""
        ranking = RankingAlgorithm()
        dislike = {"name": "requests", "confidence": 0.9}

        scored = ranking.score_preference(dislike, is_dislike=True)
        # Dislike base score is 70 + confidence boost
        assert scored.score >= 70
        assert scored.source == "preference"

    def test_preference_scores_higher_than_dislike(self):
        """Test that preference scores higher than dislike."""
        ranking = RankingAlgorithm()

        pref = {"name": "httpx", "confidence": 0.9}
        dislike = {"name": "requests", "confidence": 0.9}

        pref_scored = ranking.score_preference(pref, is_dislike=False)
        dislike_scored = ranking.score_preference(dislike, is_dislike=True)

        assert pref_scored.score > dislike_scored.score

    def test_preference_confidence_boost(self):
        """Test confidence boost for preferences."""
        ranking = RankingAlgorithm()

        high_conf = {"name": "tool", "confidence": 1.0}
        low_conf = {"name": "tool", "confidence": 0.5}

        high_scored = ranking.score_preference(high_conf)
        low_scored = ranking.score_preference(low_conf)

        # Confidence boost: 10 * confidence
        # High: 80 + 10 = 90, Low: 80 + 5 = 85
        assert high_scored.score > low_scored.score

    def test_preference_always_high_priority(self):
        """Test that preferences always score high."""
        ranking = RankingAlgorithm()

        # Compare preference to high-scoring fact
        preference = {"name": "httpx", "confidence": 0.8}
        fact = {
            "content": "Important fact",
            "metadata": {"confidence": 1.0, "category": "preference"},
            "distance": 0.0,
        }

        pref_scored = ranking.score_preference(preference)
        fact_scored = ranking.score_fact(fact)

        # Preference should still score higher than fact
        assert pref_scored.score > fact_scored.score


# ============== Rank and Select Tests ==============


class TestRankAndSelect:
    """Tests for ranking and selection."""

    def test_rank_and_select_basic(self):
        """Test basic ranking and selection."""
        ranking = RankingAlgorithm(min_score_threshold=0.0)
        items = [
            ScoredItem({"content": "a"}, score=90, token_estimate=10, source="fact"),
            ScoredItem({"content": "b"}, score=80, token_estimate=10, source="fact"),
            ScoredItem({"content": "c"}, score=70, token_estimate=10, source="fact"),
        ]

        selected = ranking.rank_and_select(items, max_tokens=100)
        assert len(selected) == 3  # All fit within budget

    def test_rank_and_select_order(self):
        """Test that selection preserves score order."""
        ranking = RankingAlgorithm(min_score_threshold=0.0)
        items = [
            ScoredItem({"content": "low"}, score=50, token_estimate=10, source="fact"),
            ScoredItem({"content": "high"}, score=90, token_estimate=10, source="fact"),
            ScoredItem({"content": "mid"}, score=70, token_estimate=10, source="fact"),
        ]

        selected = ranking.rank_and_select(items, max_tokens=100)
        # Should be sorted by score descending
        assert selected[0].score == 90
        assert selected[1].score == 70
        assert selected[2].score == 50

    def test_rank_and_select_token_budget(self):
        """Test token budget enforcement."""
        ranking = RankingAlgorithm(min_score_threshold=0.0)
        items = [
            ScoredItem({"content": "a"}, score=90, token_estimate=50, source="fact"),
            ScoredItem({"content": "b"}, score=80, token_estimate=50, source="fact"),
            ScoredItem({"content": "c"}, score=70, token_estimate=50, source="fact"),
        ]

        selected = ranking.rank_and_select(items, max_tokens=100)
        # Should only fit 2 items (100 tokens)
        assert len(selected) == 2
        assert selected[0].score == 90
        assert selected[1].score == 80

    def test_rank_and_select_exact_budget(self):
        """Test selection with exact token budget."""
        ranking = RankingAlgorithm(min_score_threshold=0.0)
        items = [
            ScoredItem({"content": "a"}, score=90, token_estimate=100, source="fact"),
            ScoredItem({"content": "b"}, score=80, token_estimate=1, source="fact"),
        ]

        selected = ranking.rank_and_select(items, max_tokens=100)
        # Should fit exactly one 100-token item
        assert len(selected) == 1
        assert selected[0].score == 90

    def test_rank_and_select_min_threshold(self):
        """Test minimum score threshold."""
        ranking = RankingAlgorithm(min_score_threshold=30.0)
        items = [
            ScoredItem({"content": "high"}, score=90, token_estimate=10, source="fact"),
            ScoredItem({"content": "mid"}, score=50, token_estimate=10, source="fact"),
            ScoredItem({"content": "low"}, score=20, token_estimate=10, source="fact"),
        ]

        selected = ranking.rank_and_select(items, max_tokens=100)
        # Should exclude the 20-score item
        assert len(selected) == 2
        assert all(s.score >= 30 for s in selected)

    def test_rank_and_select_diversity_penalty(self):
        """Test diversity penalty for same-source items."""
        ranking = RankingAlgorithm(min_score_threshold=0.0)
        items = [
            ScoredItem({"content": "a"}, score=90, token_estimate=10, source="fact"),
            ScoredItem({"content": "b"}, score=85, token_estimate=10, source="fact"),
            ScoredItem({"content": "c"}, score=80, token_estimate=10, source="fact"),
            ScoredItem({"content": "d"}, score=75, token_estimate=10, source="entity"),
        ]

        selected = ranking.rank_and_select(items, max_tokens=100, diversity_factor=0.3)
        # With diversity penalty, multiple same-source items get penalized
        # Should still select multiple but penalty applies
        assert len(selected) >= 2

    def test_rank_and_select_mixed_sources(self):
        """Test selection with mixed sources."""
        ranking = RankingAlgorithm(min_score_threshold=0.0)
        items = [
            ScoredItem({"content": "fact"}, score=70, token_estimate=10, source="fact"),
            ScoredItem(
                {"content": "entity"}, score=70, token_estimate=10, source="entity"
            ),
            ScoredItem(
                {"content": "skill"}, score=70, token_estimate=10, source="skill"
            ),
            ScoredItem(
                {"content": "pref"}, score=90, token_estimate=10, source="preference"
            ),
        ]

        selected = ranking.rank_and_select(items, max_tokens=100)
        # Preference should come first
        assert selected[0].source == "preference"

    def test_rank_and_select_empty_list(self):
        """Test selection with empty list."""
        ranking = RankingAlgorithm()
        selected = ranking.rank_and_select([], max_tokens=100)
        assert len(selected) == 0

    def test_rank_and_select_zero_budget(self):
        """Test selection with zero token budget."""
        ranking = RankingAlgorithm(min_score_threshold=0.0)
        items = [
            ScoredItem({"content": "a"}, score=90, token_estimate=10, source="fact"),
        ]

        selected = ranking.rank_and_select(items, max_tokens=0)
        assert len(selected) == 0

    def test_rank_and_select_all_below_threshold(self):
        """Test selection when all items below threshold."""
        ranking = RankingAlgorithm(min_score_threshold=100.0)
        items = [
            ScoredItem({"content": "a"}, score=50, token_estimate=10, source="fact"),
            ScoredItem({"content": "b"}, score=60, token_estimate=10, source="fact"),
        ]

        selected = ranking.rank_and_select(items, max_tokens=100)
        assert len(selected) == 0


# ============== Rank All Tests ==============


class TestRankAll:
    """Tests for full ranking pipeline."""

    def test_rank_all_basic(self):
        """Test basic rank_all operation."""
        ranking = RankingAlgorithm(min_score_threshold=0.0)

        facts = [
            {
                "content": "fact1",
                "metadata": {"confidence": 0.8, "category": "general"},
            },
            {
                "content": "fact2",
                "metadata": {"confidence": 0.9, "category": "preference"},
            },
        ]
        entities = [
            {"name": "entity1", "entity_type": "tool", "confidence": 0.8},
        ]
        skills = [
            {"description": "skill1", "metadata": {"use_count": 5}},
        ]
        preferences = {
            "prefers": [{"name": "httpx", "confidence": 0.9}],
            "dislikes": [{"name": "requests", "confidence": 0.8}],
        }

        result = ranking.rank_all(
            facts=facts,
            entities=entities,
            skills=skills,
            preferences=preferences,
            max_tokens=500,
        )

        assert "facts" in result
        assert "entities" in result
        assert "skills" in result
        assert "preferences" in result
        assert "dislikes" in result
        assert "total_tokens" in result
        assert "total_score" in result

    def test_rank_all_preferences_prioritized(self):
        """Test that preferences are prioritized."""
        ranking = RankingAlgorithm(min_score_threshold=0.0)

        facts = [
            {
                "content": "high scoring fact",
                "metadata": {"confidence": 1.0, "category": "preference"},
                "distance": 0.0,
            },
        ]
        preferences = {
            "prefers": [{"name": "httpx", "confidence": 0.9}],
            "dislikes": [],
        }

        result = ranking.rank_all(
            facts=facts,
            entities=[],
            skills=[],
            preferences=preferences,
            max_tokens=100,
        )

        # Preferences should be selected
        assert len(result["preferences"]) >= 1

    def test_rank_all_token_budget_enforced(self):
        """Test that rank_all enforces token budget."""
        ranking = RankingAlgorithm(min_score_threshold=0.0)

        # Create many high-score items
        facts = [
            {
                "content": f"fact{i}",
                "metadata": {"confidence": 1.0, "category": "preference"},
            }
            for i in range(20)
        ]

        result = ranking.rank_all(
            facts=facts,
            entities=[],
            skills=[],
            preferences={"prefers": [], "dislikes": []},
            max_tokens=50,
        )

        assert result["total_tokens"] <= 50

    def test_rank_all_empty_inputs(self):
        """Test rank_all with empty inputs."""
        ranking = RankingAlgorithm()

        result = ranking.rank_all(
            facts=[],
            entities=[],
            skills=[],
            preferences={"prefers": [], "dislikes": []},
            max_tokens=100,
        )

        assert result["facts"] == []
        assert result["entities"] == []
        assert result["skills"] == []
        assert result["preferences"] == []
        assert result["dislikes"] == []
        assert result["total_tokens"] == 0

    def test_rank_all_dislikes_categorized(self):
        """Test that dislikes are properly categorized."""
        ranking = RankingAlgorithm(min_score_threshold=0.0)

        preferences = {
            "prefers": [{"name": "httpx", "confidence": 0.9}],
            "dislikes": [{"name": "requests", "_is_dislike": True, "confidence": 0.8}],
        }

        result = ranking.rank_all(
            facts=[],
            entities=[],
            skills=[],
            preferences=preferences,
            max_tokens=100,
        )

        # Prefers should go to preferences, dislikes to dislikes
        assert all("_is_dislike" not in p for p in result["preferences"])
        assert all(d.get("_is_dislike") for d in result["dislikes"])


# ============== Category/Entity Priority Tests ==============


class TestPriorityConstants:
    """Tests for priority constant values."""

    def test_category_priority_values(self):
        """Test category priority ordering."""
        # Preference should be highest
        assert CATEGORY_PRIORITY["preference"] > CATEGORY_PRIORITY["environment"]
        assert CATEGORY_PRIORITY["environment"] > CATEGORY_PRIORITY["workflow"]
        assert CATEGORY_PRIORITY["ephemeral"] == 0  # Lowest

    def test_entity_priority_values(self):
        """Test entity priority ordering."""
        # Person and tool should be high
        assert ENTITY_PRIORITY["person"] == 10
        assert ENTITY_PRIORITY["tool"] == 8
        assert ENTITY_PRIORITY["concept"] == 4  # Lower


# ============== Edge Cases ==============


class TestEdgeCases:
    """Tests for edge cases."""

    def test_fact_empty_content(self):
        """Test scoring fact with empty content."""
        ranking = RankingAlgorithm()
        fact = {
            "content": "",
            "metadata": {"confidence": 0.8, "category": "general"},
        }

        scored = ranking.score_fact(fact)
        # Should still compute score
        assert scored.score > 0
        # Token estimate should be minimal
        assert scored.token_estimate >= 5  # Base +5 for formatting

    def test_entity_missing_type(self):
        """Test scoring entity without type."""
        ranking = RankingAlgorithm()
        entity = {"name": "unknown", "confidence": 0.8}

        scored = ranking.score_entity(entity)
        # Should use default type
        assert scored.score > 0

    def test_skill_missing_metadata(self):
        """Test scoring skill without metadata."""
        ranking = RankingAlgorithm()
        skill = {"description": "Basic skill"}

        scored = ranking.score_skill(skill)
        # Should use defaults
        assert scored.score > 0

    def test_preference_missing_confidence(self):
        """Test preference without confidence."""
        ranking = RankingAlgorithm()
        preference = {"name": "tool"}

        scored = ranking.score_preference(preference)
        # Should use default confidence (0.9)
        assert scored.score >= 80

    def test_very_large_token_estimate(self):
        """Test item with huge token estimate."""
        ranking = RankingAlgorithm(min_score_threshold=0.0)
        items = [
            ScoredItem(
                {"content": "big"}, score=90, token_estimate=10000, source="fact"
            ),
            ScoredItem(
                {"content": "small"}, score=80, token_estimate=10, source="fact"
            ),
        ]

        selected = ranking.rank_and_select(items, max_tokens=100)
        # Should skip the large item and take the small one
        assert len(selected) == 1
        assert selected[0].token_estimate == 10

    def test_negative_scores(self):
        """Test handling of negative scores."""
        ranking = RankingAlgorithm(min_score_threshold=0.0)
        items = [
            ScoredItem({"content": "neg"}, score=-10, token_estimate=10, source="fact"),
            ScoredItem({"content": "pos"}, score=50, token_estimate=10, source="fact"),
        ]

        selected = ranking.rank_and_select(items, max_tokens=100)
        # Negative score should be included if threshold is 0
        # (sorted descending, negative comes last)
        assert len(selected) >= 1

    def test_all_same_scores(self):
        """Test selection when all items have same score."""
        ranking = RankingAlgorithm(min_score_threshold=0.0)
        items = [
            ScoredItem({"content": "a"}, score=50, token_estimate=10, source="fact"),
            ScoredItem({"content": "b"}, score=50, token_estimate=10, source="entity"),
            ScoredItem({"content": "c"}, score=50, token_estimate=10, source="skill"),
        ]

        selected = ranking.rank_and_select(items, max_tokens=100)
        # Should select as many as fit
        assert len(selected) == 3
