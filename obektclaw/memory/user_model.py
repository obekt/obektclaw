"""12-layer Honcho-style user model.

Each layer captures a different facet of who the user is. Layers are inferred
by the Learning Loop after each task — they aren't direct user statements.

The 12 layers are inspired by the Hermes orange book's description of
"dialectical user modeling":

  1.  technical_level     — beginner / intermediate / expert by domain
  2.  primary_goals       — what they're trying to achieve right now
  3.  work_rhythm         — when they're typically active
  4.  comm_style          — verbosity / tone preferences
  5.  code_style          — language conventions, naming, formatting
  6.  tooling_pref        — preferred libraries and tools
  7.  domain_focus        — fields they keep coming back to
  8.  emotional_pattern   — how they react to errors or friction
  9.  trust_boundary      — what they want the agent to do autonomously
 10.  contradictions      — gaps between stated and revealed preferences
 11.  knowledge_gaps      — things they consistently get wrong / ask about
 12.  long_term_themes    — multi-week interests and projects
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .store import Store


LAYERS = (
    "technical_level",
    "primary_goals",
    "work_rhythm",
    "comm_style",
    "code_style",
    "tooling_pref",
    "domain_focus",
    "emotional_pattern",
    "trust_boundary",
    "contradictions",
    "knowledge_gaps",
    "long_term_themes",
)


@dataclass
class Trait:
    layer: str
    value: str
    evidence: str | None
    updated_at: float


class UserModel:
    def __init__(self, store: Store):
        self.store = store

    def set(self, layer: str, value: str, *, evidence: str | None = None) -> None:
        if layer not in LAYERS:
            return
        self.store.execute(
            """
            INSERT INTO user_traits (layer, value, evidence, updated_at)
            VALUES (?,?,?,?)
            ON CONFLICT(layer) DO UPDATE SET
                value = excluded.value,
                evidence = excluded.evidence,
                updated_at = excluded.updated_at
            """,
            (layer, value, evidence, time.time()),
        )

    def get(self, layer: str) -> Trait | None:
        row = self.store.fetchone(
            "SELECT layer, value, evidence, updated_at FROM user_traits WHERE layer = ?",
            (layer,),
        )
        if row is None:
            return None
        return Trait(row["layer"], row["value"], row["evidence"], row["updated_at"])

    def all(self) -> list[Trait]:
        rows = self.store.fetchall(
            "SELECT layer, value, evidence, updated_at FROM user_traits ORDER BY updated_at DESC"
        )
        return [Trait(r["layer"], r["value"], r["evidence"], r["updated_at"]) for r in rows]

    def render_for_prompt(self) -> str:
        traits = self.all()
        if not traits:
            return "(no user model yet — this is an early conversation)"
        return "\n".join(f"- {t.layer}: {t.value}" for t in traits)
