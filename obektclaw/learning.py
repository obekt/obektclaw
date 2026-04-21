"""The Learning Loop.

Runs after every assistant turn. It uses the *fast* model so it's cheap.

Five steps, mirroring the Hermes orange book §03:

  1. Curate memory     — what new facts (about the user, project, env) showed up?
  2. Create skill      — was this exchange a generalizable how-to worth saving?
  3. Improve skill     — did an existing skill produce something better?
  4. (Recall is at prompt-build time, not here.)
  5. User modeling     — refine the 12-layer profile.

We do all of this with a single LLM call that returns a structured JSON
"retrospective", then apply each suggestion via the existing memory/skill APIs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent, Turn


RETRO_SYSTEM = """You are the Learning Loop for a self-improving agent.

You will be shown the most recent user request, the agent's reply, and the
agent's existing user-model snapshot. Your job is to extract any *durable*
learnings — things that should still be true a week from now — and emit them
as a JSON object with this shape:

{
  "facts": [
    {"category": "user|project|env|preference|general",
     "key": "short_snake_key",
     "value": "the fact",
     "confidence": 0.0..1.0}
  ],
  "deleted_facts": [
    {"category": "user|project|env|preference|general",
     "key": "short_snake_key"}
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


class LearningLoop:
    def __init__(self, agent: "Agent"):
        self.agent = agent

    def run(self, turn: "Turn") -> None:
        # Skip retro on trivial exchanges to save tokens.
        if len(turn.user_text) < 12 and turn.tool_steps == 0:
            return

        snapshot = self.agent.user_model.render_for_prompt()
        user_msg = (
            f"## User said\n{turn.user_text}\n\n"
            f"## Agent replied\n{turn.assistant_text or '(empty)'}\n\n"
            f"## Tool steps used\n{turn.tool_steps}\n\n"
            f"## Existing user model\n{snapshot}\n"
        )

        retro = self.agent.llm.chat_json(RETRO_SYSTEM, user_msg, fast=True)
        if not retro:
            return

        # Persist retro JSON to logs for debugging
        self._persist_retro(retro)

        self._apply(retro)

    def _persist_retro(self, retro: dict) -> None:
        """Append retro JSON to a JSONL file for debugging."""
        try:
            logs_dir = self.agent.config.logs_dir
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"learning-{time.strftime('%Y-%m-%d')}.jsonl"
            entry = {
                "ts": time.time(),
                "retro": retro,
            }
            with log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            # Silently ignore logging failures — don't break the learning loop
            pass

    def _apply(self, retro: dict) -> None:
        for fact in retro.get("facts", []) or []:
            try:
                self.agent.persistent.upsert(
                    key=str(fact["key"]),
                    value=str(fact["value"]),
                    category=str(fact.get("category", "general")),
                    confidence=float(fact.get("confidence", 0.7)),
                )
            except (KeyError, TypeError, ValueError):
                continue

        for fact in retro.get("deleted_facts", []) or []:
            try:
                self.agent.persistent.delete(
                    category=str(fact.get("category", "general")),
                    key=str(fact["key"])
                )
            except (KeyError, TypeError):
                continue

        for upd in retro.get("user_model_updates", []) or []:
            try:
                self.agent.user_model.set(
                    layer=str(upd["layer"]),
                    value=str(upd["value"]),
                    evidence=upd.get("evidence"),
                )
            except (KeyError, TypeError):
                continue

        new_skill = retro.get("new_skill")
        if isinstance(new_skill, dict) and new_skill.get("name"):
            try:
                self.agent.skills.create(
                    name=str(new_skill["name"]),
                    description=str(new_skill.get("description", "")),
                    body=str(new_skill.get("body", "")),
                )
            except (KeyError, TypeError):
                pass

        improvement = retro.get("skill_improvement")
        if isinstance(improvement, dict) and improvement.get("name") and improvement.get("append"):
            try:
                self.agent.skills.improve(
                    name=str(improvement["name"]),
                    append=str(improvement["append"]),
                )
            except (KeyError, TypeError):
                pass

        notes = retro.get("notes")
        if notes:
            self.agent.session.add("system", f"learning loop: {notes}")
