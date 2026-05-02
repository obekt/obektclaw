"""Post-turn extraction for memory persistence.

Runs after every assistant turn. Uses the ExtractionLLMClient (separate context
from main agent) for structured entity/fact extraction.

Steps:
  1. Curate memory     — extract entities, relations, and facts
  2. Create skill      — was this exchange a generalizable how-to worth saving?
  3. Improve skill     — did an existing skill produce something better?
  4. (Recall is at prompt-build time via HybridRetriever, not here.)
  5. User modeling     — refine the 12-layer profile.

All of this happens with a single LLM call that returns a structured JSON
extraction, then applied via the memory APIs.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent, Turn

from .logging_config import get_logger

log = get_logger(__name__)


EXTRACTION_PROMPT = """You are the memory extraction system for a self-improving agent.

You will be shown the most recent user request, the agent's reply, and the
agent's existing user-model snapshot. Your job is to extract any *durable*
learnings — things that should still be true a week from now — and emit them
as a JSON object with this shape:

{
  "entities": [
    {"name": "entity name", "type": "tool|concept|environment|project|person|workflow",
     "confidence": 0.0..1.0, "properties": {"key": "value"}}
  ],
  "relations": [
    {"subject": "entity name", "predicate": "prefers|uses|dislikes|depends_on|related_to|owns|works_on|deployed_on",
     "object": "entity name", "confidence": 0.0..1.0}
  ],
  "facts": [
    {"content": "full sentence fact", "category": "preference|environment|workflow|tool|project|concept|general",
     "confidence": 0.0..1.0}
  ],
  "user_model_updates": [
    {"layer": "one of the 12 layers",
     "value": "concise inference",
     "evidence": "why you think this"}
  ],
  "new_skill": {
    "name": "kebab-case-name or null",
    "description": "one line",
    "body": "markdown body with steps and gotchas"
  } | null,
  "skill_improvement": {
    "name": "existing skill name or null",
    "append": "markdown to append (or null)"
  } | null,
  "notes": "one short sentence on why you made these calls (for the human)"
}

## Entity Extraction Rules:
- Entities are nouns mentioned: tools (httpx), concepts (async), environments (Hetzner), projects, people, workflows
- Extract entities that are relevant to the user's work, not generic programming terms
- Entity types: tool, concept, environment, project, person, workflow
- Include properties for tools (e.g., {"category": "http_client", "feature": "async"})

## Relation Extraction Rules:
- Relations connect entities, especially user preferences
- "subject" should typically be "user" for preference relations
- Relation types: prefers, uses, dislikes, depends_on, related_to, owns, works_on, deployed_on
- Only extract relations when there's clear evidence of a relationship

## What to EXCLUDE from facts (ephemeral / one-off):
- File paths from one-off questions (e.g., "csv_file_path: /tmp/x.csv")
- Counts or statistics that will change (e.g., "python_files_count: 7")
- Temporary state (e.g., "server_is_running", "current_directory")
- Anything that was just verified in the moment rather than stated as a preference

## What to INCLUDE in facts (durable):
- Explicit user preferences ("I always use httpx", "prefer async code")
- Environment/deployment info ("server is on Hetzner CX22", "uses PostgreSQL")
- Project structure that persists ("monorepo with packages/ dir")
- Tool choices the user stated as their standard

## The 12 user-model layers (with descriptions):
1.  technical_level     — beginner/intermediate/expert by domain (e.g., "expert in Python, novice in Rust")
2.  primary_goals       — what they're trying to achieve right now (e.g., "ship MVP for startup")
3.  work_rhythm         — when they're typically active (e.g., "nights and weekends")
4.  comm_style          — verbosity/tone preferences (e.g., "prefers terse, direct answers")
5.  code_style          — language conventions, naming, formatting prefs (e.g., "functional style, type hints")
6.  tooling_pref        — preferred libraries and tools (e.g., "httpx over requests, pytest over unittest")
7.  domain_focus        — fields they keep coming back to (e.g., "backend APIs, data pipelines")
8.  emotional_pattern   — how they react to errors or friction (e.g., "gets frustrated with verbose errors")
9.  trust_boundary      — what they want the agent to do autonomously (e.g., "confirm before destructive ops")
10. contradictions      — gaps between stated and revealed preferences (e.g., "says simple code but writes clever")
11. knowledge_gaps      — things they consistently get wrong / ask about (e.g., "async/await patterns")
12. long_term_themes    — multi-week interests and projects (e.g., "building personal AI agent")

## Layer assignment guidance:
- tooling_pref is for *tools and libraries only* (httpx, pytest, Docker, etc.)
- code_style is for *how they write code* (functional vs OOP, type hints, naming)
- knowledge_gaps is for *repeated misunderstandings*, not one-off questions
- emotional_pattern requires evidence of frustration/joy, not just task behavior
- If an observation is about *how the agent should behave*, it's trust_boundary or comm_style
- If unsure which layer fits, prefer knowledge_gaps or long_term_themes over misclassifying

Be conservative. If the exchange was small talk or a one-off task, return
empty arrays and nulls. Quality over quantity. If unsure, omit.
"""


class TurnExtractor:
    """Extracts entities, facts, and user model updates after each turn."""

    def __init__(self, agent: "Agent"):
        self.agent = agent

    def extract(self, turn: "Turn") -> None:
        """Run extraction on a completed turn.

        Args:
            turn: The completed turn with user_text, assistant_text, and tool_steps.
        """
        # Skip extraction on trivial exchanges to save tokens.
        if len(turn.user_text) < 12 and turn.tool_steps == 0:
            return

        log.info(
            "turn_extraction_start user_len=%d tool_steps=%d",
            len(turn.user_text),
            turn.tool_steps,
        )

        # Build context for extraction
        snapshot = self.agent.user_model.render_for_prompt()

        # Build user message for extraction
        user_msg = (
            f"## User said\n{turn.user_text}\n\n"
            f"## Agent replied\n{turn.assistant_text or '(empty)'}\n\n"
            f"## Tool steps used\n{turn.tool_steps}\n\n"
            f"## Existing user model\n{snapshot}\n"
        )

        # Use ExtractionLLMClient for structured extraction (isolated context)
        log.info(
            "turn_extraction_using_llm model=%s",
            self.agent.extraction_llm.model,
        )
        result = self.agent.extraction_llm.extract(EXTRACTION_PROMPT, user_msg)

        if not result:
            log.warning(
                "turn_extraction_failed model=%s", self.agent.extraction_llm.model
            )
            return

        log.info(
            "turn_extraction_ok entities=%d relations=%d facts=%d updates=%d new_skill=%s improvement=%s",
            len(result.get("entities") or []),
            len(result.get("relations") or []),
            len(result.get("facts") or []),
            len(result.get("user_model_updates") or []),
            bool(result.get("new_skill")),
            bool(result.get("skill_improvement")),
        )

        # Persist extraction JSON to logs for debugging
        self._persist_extraction(result)

        # Apply the extracted knowledge
        self._apply(result, turn.tool_steps)

    def _persist_extraction(self, result: dict) -> None:
        """Append extraction JSON to a JSONL file for debugging."""
        try:
            logs_dir = self.agent.config.logs_dir
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"extraction-{time.strftime('%Y-%m-%d')}.jsonl"
            entry = {
                "ts": time.time(),
                "extraction": result,
            }
            with log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            # Silently ignore logging failures — don't break extraction
            pass

    def _apply(self, result: dict, turn_number: int = 0) -> None:
        """Apply extracted knowledge to memory systems.

        Processes entities, relations, facts, user model updates,
        and skill modifications from the extraction result.
        """
        from .memory.graph_memory import Entity, Relation

        # Process entities — add to graph and sync to vector
        for entity_data in result.get("entities", []) or []:
            try:
                name = str(entity_data["name"])
                entity_type = str(entity_data["type"])
                confidence = float(entity_data.get("confidence", 0.8))
                properties = entity_data.get("properties") or {}

                # Generate entity ID: entity_{type}_{name_normalized}
                name_normalized = name.lower().replace(" ", "_").replace("-", "_")
                entity_id = f"entity_{entity_type}_{name_normalized}"

                # Create Entity object
                entity_obj = Entity(
                    id=entity_id,
                    entity_type=entity_type,
                    name=name,
                    properties=properties,
                    confidence=confidence,
                )

                # Add to CogDB graph
                self.agent.graph_memory.add_entity(entity_obj)

                # Sync to ChromaDB
                description = f"{entity_type}: {name}"
                if properties:
                    for k, v in properties.items():
                        description += f", {k}={v}"

                self.agent.memory_sync.sync_entity_to_vector(
                    entity_id=entity_id,
                    entity_name=name,
                    entity_type=entity_type,
                    description=description,
                )
            except (KeyError, TypeError, ValueError):
                continue

        # Process relations — add to graph
        for relation_data in result.get("relations", []) or []:
            try:
                subject_name = str(relation_data["subject"])
                predicate = str(relation_data["predicate"])
                object_name = str(relation_data["object"])
                confidence = float(relation_data.get("confidence", 0.8))

                # Resolve entity IDs from names
                subject_id = self._resolve_entity_id(subject_name)
                object_id = self._resolve_entity_id(object_name)

                if subject_id and object_id:
                    # Generate relation ID
                    relation_id = f"rel_{subject_id}_{predicate}_{object_id}"

                    # Create Relation object
                    relation_obj = Relation(
                        id=relation_id,
                        source_id=subject_id,
                        target_id=object_id,
                        relation_type=predicate,
                        confidence=confidence,
                    )

                    # Add to CogDB graph
                    self.agent.graph_memory.add_relation(relation_obj)
            except (KeyError, TypeError, ValueError):
                continue

        # Process facts — add to vector store
        for fact in result.get("facts", []) or []:
            try:
                content = str(fact["content"])
                category = str(fact.get("category", "general"))
                confidence = float(fact.get("confidence", 0.7))

                fact_id = f"fact_{uuid.uuid4().hex[:8]}"

                self.agent.vector_memory.add_fact(
                    fact_id=fact_id,
                    content=content,
                    category=category,
                    confidence=confidence,
                    source_turn=turn_number,
                )

                # Link fact to entities if found
                matches = self.agent.memory_sync.extract_entities_from_fact(
                    fact_content=content,
                    category=category,
                )

                if matches:
                    entity_ids = [m["entity_id"] for m in matches]
                    self.agent.memory_sync.link_fact_to_entities(fact_id, entity_ids)
            except (KeyError, TypeError, ValueError):
                continue

        # Process user model updates
        for upd in result.get("user_model_updates", []) or []:
            try:
                self.agent.user_model.set(
                    layer=str(upd["layer"]),
                    value=str(upd["value"]),
                    evidence=upd.get("evidence"),
                )
            except (KeyError, TypeError):
                continue

        # Process new skill
        new_skill = result.get("new_skill")
        if isinstance(new_skill, dict) and new_skill.get("name"):
            try:
                self.agent.skills.create(
                    name=str(new_skill["name"]),
                    description=str(new_skill.get("description", "")),
                    body=str(new_skill.get("body", "")),
                )

                # Sync new skill to vector store
                self.agent.vector_memory.add_skill(
                    skill_name=str(new_skill["name"]),
                    description=str(new_skill.get("description", "")),
                    body=str(new_skill.get("body", "")),
                )
            except (KeyError, TypeError):
                pass

        # Process skill improvement
        improvement = result.get("skill_improvement")
        if (
            isinstance(improvement, dict)
            and improvement.get("name")
            and improvement.get("append")
        ):
            try:
                self.agent.skills.improve(
                    name=str(improvement["name"]),
                    append=str(improvement["append"]),
                )

                # Update skill in vector store
                skill = self.agent.skills.get(str(improvement["name"]))
                if skill:
                    self.agent.vector_memory.add_skill(
                        skill_name=str(improvement["name"]),
                        description=skill.description,
                        body=skill.body,
                    )
            except (KeyError, TypeError):
                pass

        # Log notes
        notes = result.get("notes")
        if notes:
            self.agent.session.add("system", f"extraction: {notes}")

    def _resolve_entity_id(self, name: str) -> str | None:
        """Resolve entity ID from name by searching graph."""
        # Normalize name
        normalized = name.lower().strip()

        # Check if it's "user" — return user entity ID
        if normalized == "user":
            return self.agent.hybrid_retriever.user_entity_id

        # Search entities in graph
        entities = self.agent.graph_memory.get_entities_by_name(normalized)
        if entities:
            return entities[0].id

        # Search by type-based ID pattern
        for entity_type in [
            "tool",
            "concept",
            "environment",
            "project",
            "person",
            "workflow",
        ]:
            expected_id = f"entity_{entity_type}_{normalized}"
            entity = self.agent.graph_memory.get_entity(expected_id)
            if entity:
                return expected_id

        return None
