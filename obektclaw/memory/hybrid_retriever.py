"""Automatic context retrieval combining graph traversal and vector search.

Called during system prompt assembly — agent never invokes directly.
Provides transparent context injection without agent awareness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


from obektclaw.memory.graph_memory import GraphMemory
from obektclaw.memory.vector_memory import VectorMemory
from obektclaw.memory.ranking import RankingAlgorithm


@dataclass
class RetrievedContext:
    """Retrieved context for system prompt assembly.

    Automatically assembled from ranked knowledge items.
    The agent receives this in its system prompt without knowing
    the retrieval mechanism.
    """

    facts: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    skills: list[dict] = field(default_factory=list)
    preferences: list[dict] = field(default_factory=list)
    dislikes: list[dict] = field(default_factory=list)

    # Metadata for debugging
    total_score: float = 0.0
    total_tokens: int = 0
    retrieval_stats: dict = field(default_factory=dict)

    def to_prompt_text(self) -> str:
        """Convert to text for system prompt injection.

        The agent sees formatted context without knowing
        how it was retrieved or ranked.
        """
        lines = []

        # Preferences (highest priority)
        if self.preferences:
            lines.append("### Your Preferences")
            for pref in self.preferences:
                name = pref.get("name", "unknown")
                entity_type = pref.get("entity_type", "")
                if entity_type == "tool":
                    lines.append(f"- You prefer using **{name}**")
                else:
                    lines.append(f"- You prefer: {name}")
            lines.append("")

        # Dislikes (also high priority)
        if self.dislikes:
            lines.append("### Your Dislikes (Avoid These)")
            for dislike in self.dislikes:
                name = dislike.get("name", "unknown")
                lines.append(f"- You dislike **{name}** — use alternatives if possible")
            lines.append("")

        # Key facts (ranked by relevance)
        if self.facts:
            lines.append("### Relevant Knowledge")
            for fact in self.facts:
                content = fact.get("content", "")
                lines.append(f"- {content}")
            lines.append("")

        # Related entities (context enrichment)
        if self.entities:
            lines.append("### Context")
            entity_parts = []
            for entity in self.entities[:5]:
                entity_type = entity.get("entity_type", "unknown")
                name = entity.get("name", "unknown")
                entity_parts.append(f"{name} ({entity_type})")
            if entity_parts:
                lines.append("Related: " + ", ".join(entity_parts))
                lines.append("")

        # Relevant skills (ranked by relevance)
        if self.skills:
            lines.append("### Potentially Useful Skills")
            for skill in self.skills:
                name = skill.get("metadata", {}).get("name", skill.get("id", "unknown"))
                desc = skill.get("description", "")[:80]
                lines.append(f"- **{name}**: {desc}...")
            lines.append("")

        return "\n".join(lines)

    def estimate_tokens(self) -> int:
        """Estimate token count for context."""
        text = self.to_prompt_text()
        return len(text) // 4


class HybridRetriever:
    """Automatic context retrieval for system prompt assembly.

    This class is called by Agent._compose_system_prompt() — the agent
    never directly invokes it or knows about memory retrieval.

    Workflow:
    1. Vector search for similar facts/skills
    2. Graph traversal for connected entities
    3. Graph query for user preferences
    4. Rank all items using RankingAlgorithm
    5. Select top items within token budget
    6. Return formatted context for prompt injection
    """

    def __init__(
        self,
        graph_memory: GraphMemory,
        vector_memory: VectorMemory,
        ranking: Optional[RankingAlgorithm] = None,
        user_entity_id: str = "entity_person_user",
    ):
        self.graph = graph_memory
        self.vector = vector_memory
        self.ranking = ranking or RankingAlgorithm()
        self.user_entity_id = user_entity_id

    def retrieve_for_prompt(
        self,
        query: str,
        max_tokens: Optional[int] = None,
    ) -> RetrievedContext:
        """Retrieve context for system prompt injection.

        This is called automatically during prompt composition.
        The agent never calls this directly.

        Args:
            query: Current user message (used for semantic search)
            max_tokens: Token budget (default 2000, 0 means no budget)

        Returns:
            RetrievedContext with ranked, selected items
        """
        # Use provided max_tokens, default only if None (not if 0)
        if max_tokens is None:
            max_tokens = 2000
        search_limit = 10

        # 1. Vector search for facts
        facts = self.vector.search_similar_facts(
            query=query,
            n_results=search_limit,
        )

        # 2. Vector search for skills
        skills = self.vector.search_similar_skills(
            query=query,
            n_results=search_limit,
        )

        # 3. Graph traversal for entities connected to retrieved facts
        entities = self._get_connected_entities(
            facts, max_depth=3
        )

        # 4. Get user preferences from graph (always included if present)
        user_prefs = self.graph.get_user_preferences(self.user_entity_id)

        # 5. Rank and select using ranking algorithm
        ranked_result = self.ranking.rank_all(
            facts=facts,
            entities=entities,
            skills=skills,
            preferences=user_prefs,
            max_tokens=max_tokens,
        )

        # 6. Build context object
        context = RetrievedContext(
            facts=ranked_result["facts"],
            entities=ranked_result["entities"],
            skills=ranked_result["skills"],
            preferences=ranked_result["preferences"],
            dislikes=ranked_result["dislikes"],
            total_score=ranked_result["total_score"],
            total_tokens=ranked_result["total_tokens"],
            retrieval_stats={
                "facts_found": len(facts),
                "skills_found": len(skills),
                "entities_found": len(entities),
                "facts_selected": len(ranked_result["facts"]),
                "skills_selected": len(ranked_result["skills"]),
                "entities_selected": len(ranked_result["entities"]),
            },
        )

        return context

    def _get_connected_entities(
        self,
        facts: list[dict],
        max_depth: int = 2,
    ) -> list[dict]:
        """Get entities connected to retrieved facts.

        Traverses graph from entities mentioned in facts
        to find related context.
        """
        entities = []
        seen_ids = set()

        for fact in facts:
            # Get entity IDs from fact metadata
            entity_ids_str = fact.get("metadata", {}).get("entity_ids", "")
            if not entity_ids_str:
                continue

            for entity_id in entity_ids_str.split(","):
                entity_id = entity_id.strip()
                if entity_id in seen_ids:
                    continue
                seen_ids.add(entity_id)

                # Get entity from graph
                entity = self.graph.get_entity(entity_id)
                if entity:
                    # Convert Entity object to dict for ranking
                    entities.append(entity.to_dict())

                    # Traverse to related entities (graph proximity matters for ranking)
                    if max_depth > 0:
                        connected = self.graph.get_connected_entities(
                            entity_id,
                            max_depth=1,
                        )
                        for related_entity, depth in connected:
                            if related_entity.id in seen_ids:
                                continue
                            seen_ids.add(related_entity.id)
                            entities.append(related_entity.to_dict())

        return entities

    def get_user_environment(self) -> dict:
        """Get user's environment context (servers, tools, projects).

        Used for preference conflict checking.
        """
        env_types = ["environment", "tool", "project"]
        environment = {}

        # Get entities user is connected to
        relations_from_user = self.graph.get_relations_from(
            self.user_entity_id,
        )

        for rel in relations_from_user:
            target_entity = self.graph.get_entity(rel.target_id)
            if not target_entity:
                continue

            entity_type = target_entity.entity_type
            if entity_type in env_types:
                environment.setdefault(entity_type, []).append(
                    {
                        "id": rel.target_id,
                        "name": target_entity.name,
                        "relation": rel.relation_type,
                    }
                )

        return environment

    def check_preference_conflict(self, entity_name: str) -> Optional[dict]:
        """Check if entity conflicts with user preferences.

        Returns preference info if entity is disliked or has alternatives.
        """
        # Search for this entity in vector store
        matching_entities = self.vector.search_similar_entities(
            query=entity_name,
            n_results=5,
        )

        for match in matching_entities:
            graph_node_id = match.get("metadata", {}).get("graph_node_id")
            if not graph_node_id:
                continue

            entity = self.graph.get_entity(graph_node_id)
            if not entity:
                continue

            # Check if name matches
            if entity.name.lower() != entity_name.lower():
                continue

            # Check if user dislikes this entity
            dislikes = self.graph.get_relations_from(
                self.user_entity_id,
                relation_type="dislikes",
            )

            for rel in dislikes:
                if rel.target_id == graph_node_id:
                    return {
                        "entity": entity_name,
                        "status": "disliked",
                        "reason": "User preference indicates dislike",
                    }

            # Check for preferred alternatives
            prefers = self.graph.get_relations_from(
                self.user_entity_id,
                relation_type="prefers",
            )

            alternatives = []
            for rel in prefers:
                alt_entity = self.graph.get_entity(rel.target_id)
                if alt_entity and alt_entity.entity_type == entity.entity_type:
                    alternatives.append(alt_entity.name)

            if alternatives:
                return {
                    "entity": entity_name,
                    "status": "has_alternatives",
                    "alternatives": alternatives,
                }

        return None
