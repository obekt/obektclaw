"""Three-layer memory: session (episodic), graph + vector (semantic), user model.

New components:
- GraphMemory: CogDB-backed entity/relationship storage
- VectorMemory: ChromaDB-backed semantic search
- Embedder: Local embedding generation (sentence-transformers)
- RankingAlgorithm: Multi-factor relevance scoring
- HybridRetriever: Automatic context assembly
- MemorySync: Cross-store synchronization

Legacy (kept for backward compatibility):
- PersistentMemory: SQLite-based fact storage (deprecated, use VectorMemory instead)
"""

from .store import Store
from .session import SessionMemory
from .user_model import UserModel

# New memory system components
from .graph_memory import GraphMemory, Entity, Relation, ENTITY_TYPES, RELATION_TYPES
from .vector_memory import VectorMemory
from .embedder import embed, embed_batch, get_embedding_dimension
from .ranking import RankingAlgorithm, ScoredItem, CATEGORY_PRIORITY, ENTITY_PRIORITY
from .hybrid_retriever import HybridRetriever, RetrievedContext
from .memory_sync import MemorySync

# Legacy (backward compatibility with tests)
from .persistent import PersistentMemory, Fact

__all__ = [
    # Session storage (SQLite)
    "Store",
    "SessionMemory",
    "UserModel",
    # Automatic memory system (Graph + Vector)
    "GraphMemory",
    "Entity",
    "Relation",
    "ENTITY_TYPES",
    "RELATION_TYPES",
    "VectorMemory",
    "embed",
    "embed_batch",
    "get_embedding_dimension",
    "RankingAlgorithm",
    "ScoredItem",
    "CATEGORY_PRIORITY",
    "ENTITY_PRIORITY",
    "HybridRetriever",
    "RetrievedContext",
    "MemorySync",
    # Legacy (deprecated)
    "PersistentMemory",
    "Fact",
]
