"""Graph-based memory storage using CogDB (Torque API).

Provides entity and relationship storage with graph-based querying.
Entities are nodes with types and properties; relationships are typed edges.

Uses the cog.torque.Graph API for triple-store operations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from cog.torque import Graph


# Entity types supported by the knowledge graph
ENTITY_TYPES = [
    "tool",  # Software tools, libraries (e.g., httpx, pytest)
    "concept",  # Programming concepts (e.g., async, functional)
    "environment",  # Infrastructure (e.g., Hetzner, Docker)
    "project",  # Projects the user works on
    "person",  # People (including the user)
    "workflow",  # Workflows and procedures
]

# Relation types for connecting entities
RELATION_TYPES = [
    "prefers",  # User preference (user → prefers → tool)
    "uses",  # Usage relationship (user → uses → tool)
    "dislikes",  # User dislikes (user → dislikes → tool)
    "depends_on",  # Dependency (project → depends_on → tool)
    "related_to",  # Generic relation
    "owns",  # Ownership (user → owns → project)
    "works_on",  # Work relationship (user → works_on → project)
    "deployed_on",  # Deployment (project → deployed_on → environment)
]


@dataclass
class Entity:
    """A node in the knowledge graph."""

    id: str
    entity_type: str  # person, place, concept, preference, etc.
    name: str
    properties: dict = field(default_factory=dict)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "name": self.name,
            "properties": self.properties,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(
            id=data["id"],
            entity_type=data["entity_type"],
            name=data["name"],
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class Relation:
    """An edge in the knowledge graph."""

    id: str
    source_id: str
    target_id: str
    relation_type: str  # prefers, uses, located_at, related_to, etc.
    properties: dict = field(default_factory=dict)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "properties": self.properties,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Relation":
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


class GraphMemory:
    """CogDB-backed graph storage for entities and relationships.

    Uses the Torque API (cog.torque.Graph) for graph operations.
    """

    def __init__(self, db_path: Path):
        """Initialize the graph database.

        Args:
            db_path: Path to the CogDB database directory.
        """
        self.db_path = db_path
        db_path.mkdir(parents=True, exist_ok=True)
        # Use the Torque Graph API
        self.db = Graph(
            graph_name="obektclaw_graph",
            cog_home=str(db_path.parent),
            enable_caching=True,
        )

    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity in the graph.

        Args:
            entity: The entity to add.
        """
        # Store entity as triples: (entity_id, "is_entity", entity_data)
        self.db.put(entity.id, "is_entity", json.dumps(entity.to_dict()))
        # Also index by type for efficient type-based queries
        self.db.put(entity.id, "has_type", entity.entity_type)
        # Index by name for name-based queries (normalize for case-insensitive search)
        self.db.put(entity.id, "has_name", entity.name.lower())

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID.

        Args:
            entity_id: The entity's unique identifier.

        Returns:
            The entity if found, None otherwise.
        """
        # Query for entity data
        result = self.db.v(entity_id).out("is_entity").all()
        if result and result.get("result"):
            # The result contains the entity ID, we need to get the actual data
            # Look through triples for this entity
            for triple in self.db.triples():
                if triple[0] == entity_id and triple[1] == "is_entity":
                    data = json.loads(triple[2])
                    return Entity.from_dict(data)
        return None

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        """Get all entities of a specific type.

        Args:
            entity_type: The type to filter by.

        Returns:
            List of matching entities.
        """
        entities = []
        for triple in self.db.triples():
            if triple[1] == "has_type" and triple[2] == entity_type:
                entity = self.get_entity(triple[0])
                if entity:
                    entities.append(entity)
        return entities

    def get_entities_by_name(self, name: str) -> list[Entity]:
        """Get all entities matching a name (case-insensitive).

        Args:
            name: The name to search for.

        Returns:
            List of matching entities.
        """
        entities = []
        for triple in self.db.triples():
            if triple[1] == "has_name" and triple[2] == name.lower():
                entity = self.get_entity(triple[0])
                if entity:
                    entities.append(entity)
        return entities

    def update_entity(self, entity: Entity) -> None:
        """Update an existing entity.

        Args:
            entity: The entity with updated data.
        """
        entity.updated_at = datetime.utcnow()
        self.add_entity(entity)  # Graph.put overwrites

    def delete_entity(self, entity_id: str) -> None:
        """Delete an entity and all its relationships.

        Args:
            entity_id: The entity's unique identifier.
        """
        # Mark as deleted in properties (soft delete)
        entity = self.get_entity(entity_id)
        if entity:
            entity.properties["_deleted"] = True
            entity.updated_at = datetime.utcnow()
            self.add_entity(entity)

    def add_relation(self, relation: Relation) -> None:
        """Add a relationship between entities.

        Args:
            relation: The relation to add.
        """
        # Store relation metadata: (relation_id, "is_relation", relation_data)
        self.db.put(relation.id, "is_relation", json.dumps(relation.to_dict()))
        # Create the actual edge for graph traversal
        self.db.put(relation.source_id, relation.relation_type, relation.target_id)
        # Index relation by type
        self.db.put(relation.id, "has_relation_type", relation.relation_type)

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get a relation by ID."""
        for triple in self.db.triples():
            if triple[0] == relation_id and triple[1] == "is_relation":
                data = json.loads(triple[2])
                return Relation.from_dict(data)
        return None

    def get_relations_from(
        self, entity_id: str, relation_type: Optional[str] = None
    ) -> list[Relation]:
        """Get all relations originating from an entity.

        Args:
            entity_id: The source entity ID.
            relation_type: Optional filter by relation type.

        Returns:
            List of matching relations.
        """
        relations = []
        for triple in self.db.triples():
            # Skip metadata predicates
            if triple[1] in (
                "is_entity",
                "is_relation",
                "has_type",
                "has_name",
                "has_relation_type",
            ):
                continue
            if triple[0] == entity_id:
                if relation_type is None or triple[1] == relation_type:
                    # Find the relation object with metadata
                    target_id = triple[2]
                    # Search for matching relation object
                    for rtriple in self.db.triples():
                        if rtriple[1] == "is_relation":
                            rel = Relation.from_dict(json.loads(rtriple[2]))
                            if (
                                rel.source_id == entity_id
                                and rel.target_id == target_id
                            ):
                                if (
                                    relation_type is None
                                    or rel.relation_type == relation_type
                                ):
                                    relations.append(rel)
                                break
        return relations

    def get_relations_to(
        self, entity_id: str, relation_type: Optional[str] = None
    ) -> list[Relation]:
        """Get all relations pointing to an entity.

        Args:
            entity_id: The target entity ID.
            relation_type: Optional filter by relation type.

        Returns:
            List of matching relations.
        """
        relations = []
        for triple in self.db.triples():
            if triple[1] == "is_relation":
                rel = Relation.from_dict(json.loads(triple[2]))
                if rel.target_id == entity_id:
                    if relation_type is None or rel.relation_type == relation_type:
                        relations.append(rel)
        return relations

    def get_connected_entities(
        self, entity_id: str, max_depth: int = 2
    ) -> list[tuple[Entity, int]]:
        """Get entities connected to a given entity within a depth limit.

        Uses BFS to traverse the graph.

        Args:
            entity_id: Starting entity ID.
            max_depth: Maximum traversal depth.

        Returns:
            List of (entity, depth) tuples.
        """
        visited = set()
        result = []
        queue = [(entity_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth > max_depth:
                continue
            visited.add(current_id)

            entity = self.get_entity(current_id)
            if entity and not entity.properties.get("_deleted"):
                result.append((entity, depth))

            if depth < max_depth:
                # Get all connected entities (both directions)
                for rel in self.get_relations_from(current_id):
                    if rel.target_id not in visited:
                        queue.append((rel.target_id, depth + 1))
                for rel in self.get_relations_to(current_id):
                    if rel.source_id not in visited:
                        queue.append((rel.source_id, depth + 1))

        return result

    def get_all_entities(self, limit: int = 100) -> list[Entity]:
        """Get all entities (for synchronization purposes).

        Args:
            limit: Maximum number of entities to return.

        Returns:
            List of all entities.
        """
        entities = []
        for triple in self.db.triples():
            if triple[1] == "is_entity":
                entity = Entity.from_dict(json.loads(triple[2]))
                if not entity.properties.get("_deleted"):
                    entities.append(entity)
                if len(entities) >= limit:
                    break
        return entities

    def get_all_relations(self, limit: int = 100) -> list[Relation]:
        """Get all relations (for synchronization purposes).

        Args:
            limit: Maximum number of relations to return.

        Returns:
            List of all relations.
        """
        relations = []
        for triple in self.db.triples():
            if triple[1] == "is_relation":
                rel = Relation.from_dict(json.loads(triple[2]))
                # Check if both source and target entities exist and aren't deleted
                source = self.get_entity(rel.source_id)
                target = self.get_entity(rel.target_id)
                if (
                    source
                    and target
                    and not source.properties.get("_deleted")
                    and not target.properties.get("_deleted")
                ):
                    relations.append(rel)
                if len(relations) >= limit:
                    break
        return relations

    def get_user_preferences(self, user_entity_id: str) -> dict:
        """Get user's preferences and dislikes from the graph.

        Used by HybridRetriever for automatic context injection.

        Args:
            user_entity_id: The user's entity ID (e.g., "entity_person_user")

        Returns:
            Dict with 'prefers' and 'dislikes' lists containing entity info.
        """
        prefers = []
        dislikes = []

        # Get "prefers" relations
        prefers_rels = self.get_relations_from(user_entity_id, relation_type="prefers")
        for rel in prefers_rels:
            target_entity = self.get_entity(rel.target_id)
            if target_entity and not target_entity.properties.get("_deleted"):
                prefers.append(
                    {
                        "name": target_entity.name,
                        "entity_type": target_entity.entity_type,
                        "entity_id": rel.target_id,
                        "confidence": rel.confidence,
                    }
                )

        # Get "dislikes" relations
        dislikes_rels = self.get_relations_from(
            user_entity_id, relation_type="dislikes"
        )
        for rel in dislikes_rels:
            target_entity = self.get_entity(rel.target_id)
            if target_entity and not target_entity.properties.get("_deleted"):
                dislikes.append(
                    {
                        "name": target_entity.name,
                        "entity_type": target_entity.entity_type,
                        "_is_dislike": True,
                        "entity_id": rel.target_id,
                        "confidence": rel.confidence,
                    }
                )

        return {
            "prefers": prefers,
            "dislikes": dislikes,
        }

    def clear(self) -> None:
        """Clear all data from the graph. Use with caution!"""
        # Soft delete all entities
        for entity in self.get_all_entities(limit=10000):
            self.delete_entity(entity.id)

    def close(self) -> None:
        """Close the database connection."""
        # Graph handles cleanup automatically
        pass
