"""Ranking algorithm for context assembly.

Scores retrieved items by multiple factors and selects top-ranked within token budget.
Fully algorithmic - no LLM needed for ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ScoredItem:
    """An item with its computed relevance score."""

    item: dict
    score: float
    token_estimate: int
    source: str  # "fact", "entity", "skill", "preference"


# Category priority weights (0-10 points)
CATEGORY_PRIORITY = {
    "preference": 10,
    "environment": 8,
    "workflow": 6,
    "tool": 5,
    "project": 4,
    "concept": 3,
    "general": 2,
    "ephemeral": 0,
}

# Entity type priority (0-10 points, scaled to 0-40)
ENTITY_PRIORITY = {
    "person": 10,
    "tool": 8,
    "environment": 7,
    "project": 6,
    "workflow": 5,
    "concept": 4,
    "preference": 3,
}


class RankingAlgorithm:
    """Multi-factor ranking for context selection.

    Scoring factors (100 points max):
    1. Semantic similarity (vector distance) — 0-40 points
    2. Confidence score (extraction quality) — 0-20 points
    3. Recency boost (recent facts more relevant) — 0-15 points
    4. Entity connection strength (graph proximity) — 0-15 points
    5. Category priority (preferences > environment > general) — 0-10 points
    """

    def __init__(
        self,
        recency_halflife_days: float = 30.0,
        min_score_threshold: float = 20.0,
    ):
        """Initialize ranking algorithm.

        Args:
            recency_halflife_days: Days for recency score to halve
            min_score_threshold: Minimum score to include in context
        """
        self.recency_halflife_days = recency_halflife_days
        self.min_score_threshold = min_score_threshold

    def score_fact(
        self, fact: dict, query_distance: Optional[float] = None
    ) -> ScoredItem:
        """Score a fact for relevance.

        Args:
            fact: Fact dict with content, metadata, and optional distance
            query_distance: Semantic distance from query (0-1, lower = more similar)

        Returns:
            ScoredItem with computed score
        """
        metadata = fact.get("metadata", {})

        # 1. Semantic similarity (0-40 points)
        if query_distance is not None:
            similarity_score = 40 * (1 - query_distance)
        elif "distance" in fact:
            similarity_score = 40 * (1 - fact["distance"])
        else:
            similarity_score = 20  # Default mid-range

        # 2. Confidence score (0-20 points)
        confidence = float(metadata.get("confidence", 0.8))
        confidence_score = 20 * confidence

        # 3. Recency boost (0-15 points)
        created_at = metadata.get("created_at")
        recency_score = self._compute_recency_score(created_at)

        # 4. Entity connection strength (0-15 points)
        entity_ids = metadata.get("entity_ids", "")
        connection_score = 0
        if entity_ids:
            num_connections = len(entity_ids.split(",")) if entity_ids else 0
            connection_score = min(15, 5 + num_connections * 2)

        # 5. Category priority (0-10 points)
        category = metadata.get("category", "general")
        category_score = CATEGORY_PRIORITY.get(category, 2)

        total_score = (
            similarity_score
            + confidence_score
            + recency_score
            + connection_score
            + category_score
        )

        # Estimate tokens (~4 chars per token)
        content = fact.get("content", "")
        token_estimate = len(content) // 4 + 5

        return ScoredItem(
            item=fact,
            score=total_score,
            token_estimate=token_estimate,
            source="fact",
        )

    def score_entity(self, entity: dict, graph_distance: int = 0) -> ScoredItem:
        """Score an entity for relevance.

        Args:
            entity: Entity dict with properties
            graph_distance: Distance from user entity in graph

        Returns:
            ScoredItem with computed score
        """
        entity_type = entity.get("entity_type", "concept")
        name = entity.get("name", "")

        # Entity type priority (0-40 points)
        type_score = ENTITY_PRIORITY.get(entity_type, 4) * 4

        # Graph proximity (0-30 points)
        proximity_score = max(0, 30 - graph_distance * 10)

        # Confidence (0-20 points)
        confidence = float(entity.get("confidence", 0.8))
        confidence_score = 20 * confidence

        total_score = type_score + proximity_score + confidence_score

        # Token estimate
        token_estimate = len(name) // 4 + 10

        return ScoredItem(
            item=entity,
            score=total_score,
            token_estimate=token_estimate,
            source="entity",
        )

    def score_skill(
        self, skill: dict, query_distance: Optional[float] = None
    ) -> ScoredItem:
        """Score a skill for relevance.

        Args:
            skill: Skill dict with description and metadata
            query_distance: Semantic distance from query

        Returns:
            ScoredItem with computed score
        """
        metadata = skill.get("metadata", {})

        # Semantic similarity (0-40 points)
        if query_distance is not None:
            similarity_score = 40 * (1 - query_distance)
        elif "distance" in skill:
            similarity_score = 40 * (1 - skill["distance"])
        else:
            similarity_score = 20

        # Usage frequency boost (0-25 points)
        use_count = int(metadata.get("use_count", 0))
        usage_score = min(25, 5 + math.log10(use_count + 1) * 5)

        # Success rate (0-20 points)
        success_count = int(metadata.get("success_count", 0))
        if use_count > 0:
            success_rate = success_count / use_count
            success_score = 20 * success_rate
        else:
            success_score = 10  # New skills get mid-range

        # Recency (0-15 points)
        created_at = metadata.get("created_at")
        recency_score = self._compute_recency_score(created_at)

        total_score = similarity_score + usage_score + success_score + recency_score

        # Token estimate
        description = skill.get("description", "")
        token_estimate = len(description) // 4 + 15

        return ScoredItem(
            item=skill,
            score=total_score,
            token_estimate=token_estimate,
            source="skill",
        )

    def score_preference(
        self, preference: dict, is_dislike: bool = False
    ) -> ScoredItem:
        """Score a user preference.

        Args:
            preference: Preference entity dict
            is_dislike: Whether this is a dislike (negative preference)

        Returns:
            ScoredItem with computed score
        """
        # Preferences always get high priority
        base_score = 80 if not is_dislike else 70

        # Confidence boost
        confidence = float(preference.get("confidence", 0.9))
        confidence_boost = 10 * confidence

        total_score = base_score + confidence_boost

        # Token estimate
        name = preference.get("name", "")
        token_estimate = len(name) // 4 + 10

        return ScoredItem(
            item=preference,
            score=total_score,
            token_estimate=token_estimate,
            source="preference",
        )

    def _compute_recency_score(self, created_at: Optional[str]) -> float:
        """Compute recency score based on timestamp.

        Uses exponential decay with configurable halflife.
        Score = 15 * 2^(-age_days / halflife_days)

        Args:
            created_at: ISO timestamp string

        Returns:
            Recency score (0-15)
        """
        if not created_at:
            return 7.5  # Mid-range for unknown

        try:
            created = datetime.fromisoformat(created_at)
            age_days = (datetime.utcnow() - created).total_seconds() / 86400

            # Exponential decay
            recency_score = 15 * math.pow(2, -age_days / self.recency_halflife_days)
            return max(0, min(15, recency_score))
        except (ValueError, TypeError):
            return 7.5

    def rank_and_select(
        self,
        items: list[ScoredItem],
        max_tokens: int,
        diversity_factor: float = 0.3,
    ) -> list[ScoredItem]:
        """Rank items and select top items within token budget.

        Uses greedy selection with diversity penalty to avoid
        selecting too many similar items.

        Args:
            items: List of scored items
            max_tokens: Maximum token budget
            diversity_factor: Penalty for same-source items (0-1)

        Returns:
            Selected items within budget, ranked by score
        """
        # Sort by score descending
        sorted_items = sorted(items, key=lambda x: x.score, reverse=True)

        selected = []
        tokens_used = 0
        source_counts = {}

        for item in sorted_items:
            # Skip below threshold
            if item.score < self.min_score_threshold:
                continue

            # Apply diversity penalty
            source = item.source
            diversity_penalty = source_counts.get(source, 0) * diversity_factor * 10
            adjusted_score = item.score - diversity_penalty

            # Still worth including after penalty?
            if adjusted_score < self.min_score_threshold:
                continue

            # Check token budget
            if tokens_used + item.token_estimate > max_tokens:
                continue

            selected.append(item)
            tokens_used += item.token_estimate
            source_counts[source] = source_counts.get(source, 0) + 1

        return selected

    def rank_all(
        self,
        facts: list[dict],
        entities: list[dict],
        skills: list[dict],
        preferences: dict,
        max_tokens: int,
    ) -> dict:
        """Score all items and select optimal context.

        Args:
            facts: List of facts from vector search
            entities: List of entities from graph traversal
            skills: List of skills from vector search
            preferences: Dict with 'prefers' and 'dislikes' lists
            max_tokens: Maximum token budget

        Returns:
            Dict with selected items by category and total tokens
        """
        all_items = []

        # Score facts
        for fact in facts:
            scored = self.score_fact(fact)
            all_items.append(scored)

        # Score entities (ordered by graph proximity)
        for i, entity in enumerate(entities):
            scored = self.score_entity(entity, graph_distance=min(i, 3))
            all_items.append(scored)

        # Score skills
        for skill in skills:
            scored = self.score_skill(skill)
            all_items.append(scored)

        # Score preferences
        for pref in preferences.get("prefers", []):
            scored = self.score_preference(pref, is_dislike=False)
            all_items.append(scored)

        for dislike in preferences.get("dislikes", []):
            scored = self.score_preference(dislike, is_dislike=True)
            all_items.append(scored)

        # Rank and select
        selected = self.rank_and_select(all_items, max_tokens)

        # Group by source
        result = {
            "facts": [],
            "entities": [],
            "skills": [],
            "preferences": [],
            "dislikes": [],
            "total_tokens": 0,
            "total_score": 0.0,
        }

        for item in selected:
            if item.source == "fact":
                result["facts"].append(item.item)
            elif item.source == "entity":
                result["entities"].append(item.item)
            elif item.source == "skill":
                result["skills"].append(item.item)
            elif item.source == "preference":
                # Check if it's a dislike based on item properties
                if item.item.get("_is_dislike"):
                    result["dislikes"].append(item.item)
                else:
                    result["preferences"].append(item.item)

            result["total_tokens"] += item.token_estimate
            result["total_score"] += item.score

        return result
