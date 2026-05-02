"""Vector-based memory storage using ChromaDB.

Provides semantic search for facts, memories, skills, and entities.
Uses local embeddings from sentence-transformers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings

from obektclaw.memory.embedder import embed, embed_batch, get_embedding_dimension


class VectorMemory:
    """ChromaDB-backed vector memory for semantic search.

    Collections:
    - facts: Extracted facts from Learning Loop
    - memories: Conversation messages
    - skills: Skill descriptions
    - entities: Entity descriptions synced from CogDB
    """

    def __init__(self, chroma_path: Optional[Path] = None):
        """Initialize ChromaDB with persistent client.

        Args:
            chroma_path: Path to ChromaDB storage. Defaults to CONFIG.chroma_path.
        """
        from pathlib import Path

        path = chroma_path or CONFIG.chroma_path
        self.client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Initialize collections
        self._init_collections()

    def _init_collections(self) -> None:
        """Create or get all collections."""
        # Use explicit embedding function with our local embedder
        # ChromaDB will call our embedder for each document

        self.facts = self.client.get_or_create_collection(
            name="facts", metadata={"description": "Extracted facts from Learning Loop"}
        )

        self.memories = self.client.get_or_create_collection(
            name="memories", metadata={"description": "Conversation messages"}
        )

        self.skills = self.client.get_or_create_collection(
            name="skills", metadata={"description": "Skill descriptions"}
        )

        self.entities = self.client.get_or_create_collection(
            name="entities",
            metadata={"description": "Entity descriptions synced from CogDB"},
        )

    def add_fact(
        self,
        fact_id: str,
        content: str,
        category: str,
        confidence: float,
        source_turn: int,
        entity_ids: list[str] = None,
    ) -> None:
        """Add a fact to the vector store.

        Args:
            fact_id: Unique ID (e.g., "fact_001")
            content: Fact content
            category: Category (preference, environment, workflow, etc.)
            confidence: Confidence score (0.0-1.0)
            source_turn: Turn number where fact was extracted
            entity_ids: Related entity IDs from CogDB
        """
        metadata = {
            "category": category,
            "confidence": confidence,
            "source_turn": source_turn,
            "created_at": datetime.utcnow().isoformat(),
        }

        if entity_ids:
            metadata["entity_ids"] = ",".join(entity_ids)

        # Generate embedding locally
        embedding = embed(content)

        # Use upsert to allow updating existing facts
        self.facts.upsert(
            ids=[fact_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )

    def add_memory(
        self,
        memory_id: str,
        content: str,
        session_id: int,
        role: str,
        timestamp: str,
        tool_calls: list[str] = None,
    ) -> None:
        """Add a conversation memory.

        Args:
            memory_id: Unique ID
            content: Message content
            session_id: Session ID
            role: "user" or "assistant"
            timestamp: ISO timestamp
            tool_calls: List of tool names used (if assistant)
        """
        metadata = {
            "session_id": session_id,
            "role": role,
            "timestamp": timestamp,
        }

        if tool_calls:
            metadata["tool_calls"] = ",".join(tool_calls)

        embedding = embed(content)

        self.memories.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )

    def add_skill(
        self,
        skill_name: str,
        description: str,
        body: str = None,
        use_count: int = 0,
        success_count: int = 0,
    ) -> None:
        """Add a skill to vector store.

        Args:
            skill_name: Skill name (also used as ID)
            description: Short description
            body: Full skill body (optional)
            use_count: Usage count
            success_count: Success count
        """
        # Use description for search
        document = description
        if body:
            document += f"\n{body[:500]}"

        metadata = {
            "name": skill_name,
            "use_count": use_count,
            "success_count": success_count,
            "created_at": datetime.utcnow().isoformat(),
        }

        if body:
            metadata["has_full_body"] = True

        embedding = embed(document)

        self.skills.upsert(
            ids=[skill_name],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )

    def add_entity(
        self,
        entity_id: str,
        description: str,
        entity_type: str,
        graph_node_id: str,
    ) -> None:
        """Add entity description synced from CogDB.

        Args:
            entity_id: Vector store entity ID
            description: Entity description
            entity_type: Type from ENTITY_TYPES
            graph_node_id: Corresponding CogDB node ID
        """
        embedding = embed(description)

        self.entities.upsert(
            ids=[entity_id],
            embeddings=[embedding],
            documents=[description],
            metadatas=[
                {
                    "entity_type": entity_type,
                    "graph_node_id": graph_node_id,
                    "synced_at": datetime.utcnow().isoformat(),
                }
            ],
        )

    def search_similar_facts(
        self,
        query: str,
        n_results: int = 10,
        category_filter: str = None,
        min_confidence: float = None,
    ) -> list[dict]:
        """Search for similar facts.

        Args:
            query: Query text
            n_results: Max results
            category_filter: Filter by category
            min_confidence: Minimum confidence threshold

        Returns:
            List of fact dicts with id, content, metadata, distance
        """
        where_filter = None
        if category_filter or min_confidence:
            conditions = []
            if category_filter:
                conditions.append({"category": category_filter})
            if min_confidence:
                conditions.append({"confidence": {"$gte": min_confidence}})

            if len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = {"$and": conditions}

        query_embedding = embed(query)

        results = self.facts.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        facts = []
        if results["ids"] and results["ids"][0]:
            for i, doc in enumerate(results["documents"][0]):
                facts.append(
                    {
                        "id": results["ids"][0][i],
                        "content": doc,
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                        if "distances" in results
                        else 0.0,
                    }
                )

        return facts

    def search_similar_memories(
        self,
        query: str,
        n_results: int = 10,
        session_filter: int = None,
        role_filter: str = None,
    ) -> list[dict]:
        """Search for similar conversation memories.

        Args:
            query: Query text
            n_results: Max results
            session_filter: Filter by session ID
            role_filter: Filter by role

        Returns:
            List of memory dicts
        """
        where_filter = None
        if session_filter or role_filter:
            conditions = []
            if session_filter:
                conditions.append({"session_id": session_filter})
            if role_filter:
                conditions.append({"role": role_filter})

            if len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = {"$and": conditions}

        query_embedding = embed(query)

        results = self.memories.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas"],
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, doc in enumerate(results["documents"][0]):
                memories.append(
                    {
                        "id": results["ids"][0][i],
                        "content": doc,
                        "metadata": results["metadatas"][0][i],
                    }
                )

        return memories

    def search_similar_skills(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Search for similar skills.

        Args:
            query: Query text
            n_results: Max results

        Returns:
            List of skill dicts
        """
        query_embedding = embed(query)

        results = self.skills.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas"],
        )

        skills = []
        if results["ids"] and results["ids"][0]:
            for i, doc in enumerate(results["documents"][0]):
                skills.append(
                    {
                        "id": results["ids"][0][i],
                        "description": doc,
                        "metadata": results["metadatas"][0][i],
                    }
                )

        return skills

    def search_similar_entities(
        self,
        query: str,
        n_results: int = 5,
        entity_type_filter: str = None,
    ) -> list[dict]:
        """Search for similar entities.

        Args:
            query: Query text
            n_results: Max results
            entity_type_filter: Filter by entity type

        Returns:
            List of entity dicts
        """
        where_filter = None
        if entity_type_filter:
            where_filter = {"entity_type": entity_type_filter}

        query_embedding = embed(query)

        results = self.entities.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        entities = []
        if results["ids"] and results["ids"][0]:
            for i, doc in enumerate(results["documents"][0]):
                entities.append(
                    {
                        "id": results["ids"][0][i],
                        "description": doc,
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                        if "distances" in results
                        else 0.0,
                    }
                )

        return entities

    def get_fact_by_id(self, fact_id: str) -> Optional[dict]:
        """Get fact by ID."""
        results = self.facts.get(ids=[fact_id], include=["documents", "metadatas"])
        if not results["ids"]:
            return None

        return {
            "id": results["ids"][0],
            "content": results["documents"][0],
            "metadata": results["metadatas"][0],
        }

    def get_recent_facts(
        self,
        limit: int = 10,
        category: str = None,
    ) -> list[dict]:
        """Get recent facts (most recently added)."""
        where_filter = {"category": category} if category else None

        # Get all facts and sort by created_at
        results = self.facts.get(
            where=where_filter,
            include=["documents", "metadatas"],
            limit=100,
        )

        facts = []
        if results["ids"]:
            for i, id_ in enumerate(results["ids"]):
                facts.append(
                    {
                        "id": id_,
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i],
                    }
                )

            # Sort by created_at descending
            facts.sort(
                key=lambda x: x["metadata"].get("created_at", ""),
                reverse=True,
            )

        return facts[:limit]

    def update_fact_confidence(self, fact_id: str, new_confidence: float) -> None:
        """Update confidence score for a fact."""
        existing = self.get_fact_by_id(fact_id)
        if not existing:
            return

        metadata = existing["metadata"]
        metadata["confidence"] = new_confidence
        metadata["updated_at"] = datetime.utcnow().isoformat()

        embedding = embed(existing["content"])

        self.facts.upsert(
            ids=[fact_id],
            embeddings=[embedding],
            documents=[existing["content"]],
            metadatas=[metadata],
        )

    def delete_fact(self, fact_id: str) -> None:
        """Delete a fact."""
        self.facts.delete(ids=[fact_id])

    def delete_memories_by_session(self, session_id: int) -> None:
        """Delete all memories for a session."""
        results = self.memories.get(where={"session_id": session_id}, include=[])

        if results["ids"]:
            self.memories.delete(ids=results["ids"])

    def stats(self) -> dict:
        """Get vector store statistics."""
        return {
            "facts_count": self.facts.count(),
            "memories_count": self.memories.count(),
            "skills_count": self.skills.count(),
            "entities_count": self.entities.count(),
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": get_embedding_dimension(),
        }

    def clear_all(self) -> None:
        """Clear all collections."""
        self.client.reset()
        self._init_collections()

    def close(self) -> None:
        """Close the database connection."""
        # ChromaDB PersistentClient doesn't need explicit close
        pass
