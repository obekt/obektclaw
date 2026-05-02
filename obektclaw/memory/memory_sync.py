"""Synchronization between CogDB (graph) and ChromaDB (vector) stores.

Ensures entity IDs are consistent across both systems.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Optional

from obektclaw.memory.graph_memory import GraphMemory, ENTITY_TYPES
from obektclaw.memory.vector_memory import VectorMemory


class MemorySync:
    """Syncs entities between CogDB and ChromaDB.

    Purpose:
    - Entities are stored in both CogDB (for relationships) and ChromaDB (for search)
    - Keeps entity IDs consistent
    - Syncs new entities from Learning Loop extraction
    """

    def __init__(
        self,
        graph_memory: GraphMemory,
        vector_memory: VectorMemory,
    ):
        self.graph = graph_memory
        self.vector = vector_memory

    def sync_entity_to_vector(
        self,
        entity_id: str,
        entity_name: str,
        entity_type: str,
        description: Optional[str] = None,
    ) -> str:
        """Sync an entity from CogDB to ChromaDB entities collection.

        Args:
            entity_id: CogDB entity ID
            entity_name: Entity name
            entity_type: Entity type
            description: Optional description (generated if not provided)

        Returns:
            Vector store entity ID
        """
        if not description:
            description = f"{entity_type}: {entity_name}"

        # Generate vector store ID (hash of graph ID for consistency)
        vector_id = hashlib.md5(entity_id.encode()).hexdigest()[:12]

        self.vector.add_entity(
            entity_id=vector_id,
            description=description,
            entity_type=entity_type,
            graph_node_id=entity_id,
        )

        return vector_id

    def sync_all_entities(self) -> dict:
        """Sync all entities from CogDB to ChromaDB.

        Returns:
            Dict with sync statistics
        """
        stats = {
            "synced": 0,
            "skipped": 0,
            "errors": 0,
            "by_type": {},
        }

        for entity_type in ENTITY_TYPES:
            entities = self.graph.get_entities_by_type(entity_type)

            for entity in entities:
                entity_name = entity.name
                entity_id = entity.id

                # Build description from properties
                description = f"{entity_type}: {entity_name}"
                props = entity.properties
                if props:
                    for key, value in props.items():
                        if key not in ["created_at", "updated_at"]:
                            description += f", {key}={value}"

                try:
                    self.sync_entity_to_vector(
                        entity_id=entity_id,
                        entity_name=entity_name,
                        entity_type=entity_type,
                        description=description,
                    )
                    stats["synced"] += 1
                    stats["by_type"][entity_type] = (
                        stats["by_type"].get(entity_type, 0) + 1
                    )
                except Exception:
                    stats["errors"] += 1

        return stats

    def extract_entities_from_fact(
        self,
        fact_content: str,
        category: str,
    ) -> list[dict]:
        """Extract potential entities from a fact.

        Uses ChromaDB entity search to find matching entities.

        Args:
            fact_content: Fact text
            category: Fact category

        Returns:
            List of potential entity matches
        """
        # Search entities collection for matches
        results = self.vector.search_similar_entities(
            query=fact_content,
            n_results=5,
        )

        matches = []
        for match in results:
            graph_node_id = match.get("metadata", {}).get("graph_node_id")
            distance = match.get("distance", 0.0)

            # Only include close matches (distance < 0.5)
            if distance < 0.5:
                entity = self.graph.get_entity(graph_node_id)
                if entity:
                    matches.append(
                        {
                            "entity_id": graph_node_id,
                            "name": entity.name,
                            "type": entity.entity_type,
                            "distance": distance,
                        }
                    )

        return matches

    def link_fact_to_entities(
        self,
        fact_id: str,
        entity_ids: list[str],
    ) -> None:
        """Update fact metadata to link it to entities.

        Args:
            fact_id: ChromaDB fact ID
            entity_ids: List of CogDB entity IDs
        """
        fact = self.vector.get_fact_by_id(fact_id)
        if not fact:
            return

        metadata = fact["metadata"]
        metadata["entity_ids"] = ",".join(entity_ids)
        metadata["linked_at"] = datetime.utcnow().isoformat()

        # Update fact with new metadata (requires re-embedding)
        self.vector.update_fact_confidence(fact_id, metadata.get("confidence", 0.8))

    def check_consistency(self) -> dict:
        """Check consistency between CogDB and ChromaDB.

        Returns:
            Dict with consistency report
        """
        report = {
            "graph_entities": 0,
            "vector_entities": 0,
            "consistent": 0,
            "missing_in_vector": [],
            "missing_in_graph": [],
        }

        # Count graph entities
        for entity_type in ENTITY_TYPES:
            entities = self.graph.get_entities_by_type(entity_type)
            report["graph_entities"] += len(entities)

        # Count vector entities
        report["vector_entities"] = self.vector.entities.count()

        # Get all graph entity IDs
        graph_entity_ids = set()
        for entity_type in ENTITY_TYPES:
            entities = self.graph.get_entities_by_type(entity_type)
            for entity in entities:
                graph_entity_ids.add(entity.id)

        # Get all vector entity graph_node_ids
        all_vector_entities = self.vector.entities.get(limit=1000)
        vector_graph_ids = set()
        if all_vector_entities["metadatas"]:
            for metadata in all_vector_entities["metadatas"]:
                graph_node_id = metadata.get("graph_node_id")
                if graph_node_id:
                    vector_graph_ids.add(graph_node_id)

        # Check consistency
        for entity_id in graph_entity_ids:
            if entity_id in vector_graph_ids:
                report["consistent"] += 1
            else:
                report["missing_in_vector"].append(entity_id)

        for graph_id in vector_graph_ids:
            if graph_id not in graph_entity_ids:
                report["missing_in_graph"].append(graph_id)

        return report
