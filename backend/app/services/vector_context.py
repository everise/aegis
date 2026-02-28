"""
Vector Context Manager using ChromaDB.

Manages conversation context using a vector database for semantic storage
and retrieval. Each session maintains its own collection of message embeddings,
enabling semantic search, context statistics, and intelligent context management.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings


@dataclass
class ContextStats:
    """Statistics about the current conversation context in vector DB."""
    session_id: int
    total_vectors: int  # Total embeddings stored for the session
    user_message_count: int  # Number of user messages stored
    assistant_message_count: int  # Number of assistant messages stored
    system_message_count: int  # Number of system messages stored
    total_tokens_estimate: int  # Estimated total tokens in stored context
    context_window_usage: float  # Percentage of context window used (0.0 - 1.0)
    max_context_tokens: int  # Maximum context window size
    oldest_message_time: Optional[str] = None  # Timestamp of oldest stored message
    newest_message_time: Optional[str] = None  # Timestamp of newest stored message
    collection_name: str = ""  # ChromaDB collection name
    similar_context_count: int = 0  # Number of semantically similar contexts found


@dataclass
class ContextDocument:
    """A document stored in the vector context."""
    doc_id: str
    content: str
    role: str
    session_id: int
    timestamp: str
    token_estimate: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorContextManager:
    """
    Manages conversation context using ChromaDB vector database.
    
    Features:
    - Stores message embeddings per session
    - Provides semantic search over conversation history
    - Tracks context statistics (vector count, token usage, etc.)
    - Supports context window management
    
    ChromaDB is used as an embedded vector database with built-in
    sentence-transformer embeddings for semantic similarity search.
    """
    
    # Default context window size (tokens)
    DEFAULT_MAX_CONTEXT_TOKENS = 128_000
    # Approximate chars per token
    CHARS_PER_TOKEN = 4.0
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    ):
        """
        Initialize the Vector Context Manager.
        
        Args:
            persist_directory: Directory for ChromaDB persistence.
                              If None, uses in-memory storage.
            max_context_tokens: Maximum context window size in tokens.
        """
        self.max_context_tokens = max_context_tokens
        self._persist_directory = persist_directory
        
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                ),
            )
        else:
            self._client = chromadb.Client(
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                ),
            )
    
    def _collection_name(self, session_id: int) -> str:
        """Generate collection name for a session."""
        return f"session_{session_id}"
    
    def _get_or_create_collection(self, session_id: int):
        """Get or create a ChromaDB collection for a session."""
        name = self._collection_name(session_id)
        return self._client.get_or_create_collection(
            name=name,
            metadata={"session_id": str(session_id), "hnsw:space": "cosine"},
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return max(1, int(len(text) / self.CHARS_PER_TOKEN))
    
    def add_message(
        self,
        session_id: int,
        message_id: int,
        role: str,
        content: str,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a message to the vector context store.
        
        Args:
            session_id: Session identifier
            message_id: Message identifier
            role: Message role (user/assistant/system)
            content: Message content text
            timestamp: ISO format timestamp
            metadata: Additional metadata
            
        Returns:
            Document ID of the stored message
        """
        collection = self._get_or_create_collection(session_id)
        
        doc_id = f"msg_{session_id}_{message_id}"
        ts = timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        token_estimate = self._estimate_tokens(content)
        
        doc_metadata = {
            "session_id": str(session_id),
            "message_id": str(message_id),
            "role": role,
            "timestamp": ts,
            "token_estimate": token_estimate,
        }
        if metadata:
            # ChromaDB metadata values must be str, int, float, or bool
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    doc_metadata[k] = v
        
        # Upsert to handle duplicates gracefully
        collection.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[doc_metadata],
        )
        
        return doc_id
    
    def get_context_stats(self, session_id: int) -> ContextStats:
        """
        Get comprehensive context statistics for a session.
        
        Returns statistics about the vector-stored context including
        message counts by role, token estimates, and context window usage.
        """
        collection_name = self._collection_name(session_id)
        
        try:
            collection = self._client.get_collection(collection_name)
        except Exception:
            # Collection doesn't exist yet
            return ContextStats(
                session_id=session_id,
                total_vectors=0,
                user_message_count=0,
                assistant_message_count=0,
                system_message_count=0,
                total_tokens_estimate=0,
                context_window_usage=0.0,
                max_context_tokens=self.max_context_tokens,
                collection_name=collection_name,
            )
        
        # Get all documents in the collection
        result = collection.get(include=["metadatas"])
        
        total_vectors = len(result["ids"]) if result["ids"] else 0
        
        user_count = 0
        assistant_count = 0
        system_count = 0
        total_tokens = 0
        oldest_time = None
        newest_time = None
        
        if result["metadatas"]:
            for meta in result["metadatas"]:
                role = meta.get("role", "")
                if role == "user":
                    user_count += 1
                elif role == "assistant":
                    assistant_count += 1
                elif role == "system":
                    system_count += 1
                
                total_tokens += int(meta.get("token_estimate", 0))
                
                ts = meta.get("timestamp", "")
                if ts:
                    if oldest_time is None or ts < oldest_time:
                        oldest_time = ts
                    if newest_time is None or ts > newest_time:
                        newest_time = ts
        
        context_usage = min(1.0, total_tokens / self.max_context_tokens) if self.max_context_tokens > 0 else 0.0
        
        return ContextStats(
            session_id=session_id,
            total_vectors=total_vectors,
            user_message_count=user_count,
            assistant_message_count=assistant_count,
            system_message_count=system_count,
            total_tokens_estimate=total_tokens,
            context_window_usage=context_usage,
            max_context_tokens=self.max_context_tokens,
            oldest_message_time=oldest_time,
            newest_message_time=newest_time,
            collection_name=collection_name,
        )
    
    def search_similar(
        self,
        session_id: int,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar messages in a session's context.
        
        Args:
            session_id: Session identifier
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        try:
            collection = self._client.get_collection(self._collection_name(session_id))
        except Exception:
            return []
        
        count = collection.count()
        if count == 0:
            return []
        
        # Limit top_k to available documents
        actual_k = min(top_k, count)
        
        results = collection.query(
            query_texts=[query],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )
        
        similar = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                similar.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })
        
        return similar
    
    def get_relevant_context(
        self,
        session_id: int,
        query: str,
        max_tokens: int = 4000,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context messages within a token budget.
        
        Searches for semantically similar messages and returns as many
        as fit within the token budget, sorted by relevance.
        """
        results = self.search_similar(session_id, query, top_k=top_k)
        
        selected = []
        total_tokens = 0
        
        for result in results:
            tokens = int(result.get("metadata", {}).get("token_estimate", 0))
            if tokens == 0:
                tokens = self._estimate_tokens(result.get("content", ""))
            
            if total_tokens + tokens <= max_tokens:
                selected.append(result)
                total_tokens += tokens
            else:
                break
        
        return selected
    
    def delete_session_context(self, session_id: int) -> bool:
        """
        Delete all vector context for a session.
        
        Returns:
            True if the collection was deleted, False if it didn't exist.
        """
        try:
            self._client.delete_collection(self._collection_name(session_id))
            return True
        except Exception:
            return False
    
    def list_sessions(self) -> List[int]:
        """List all session IDs that have vector context stored."""
        collections = self._client.list_collections()
        session_ids = []
        for col in collections:
            name = col.name if hasattr(col, 'name') else str(col)
            if name.startswith("session_"):
                try:
                    sid = int(name.replace("session_", ""))
                    session_ids.append(sid)
                except ValueError:
                    pass
        return sorted(session_ids)


# Global singleton instance
_vector_context_manager: Optional[VectorContextManager] = None


def get_vector_context_manager() -> VectorContextManager:
    """Get the global VectorContextManager instance."""
    global _vector_context_manager
    if _vector_context_manager is None:
        from app.config import get_settings

        settings = get_settings()
        # Resolve chroma_persist_dir relative to the project root
        persist_dir = str(
            Path(__file__).resolve().parent.parent.parent / settings.chroma_persist_dir
        )
        _vector_context_manager = VectorContextManager(
            persist_directory=persist_dir,
            max_context_tokens=settings.max_context_tokens,
        )
    return _vector_context_manager


def reset_vector_context_manager() -> None:
    """Reset the global instance (for testing)."""
    global _vector_context_manager
    _vector_context_manager = None
