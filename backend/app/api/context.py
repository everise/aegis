"""
Context API endpoints.

Provides endpoints for querying vector context statistics
and performing semantic search over conversation history.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.vector_context import get_vector_context_manager, ContextStats


router = APIRouter()


# Response Models
class ContextStatsResponse(BaseModel):
    """Response model for context statistics."""
    session_id: int
    total_vectors: int
    user_message_count: int
    assistant_message_count: int
    system_message_count: int
    total_tokens_estimate: int
    context_window_usage: float
    max_context_tokens: int
    oldest_message_time: Optional[str] = None
    newest_message_time: Optional[str] = None
    collection_name: str = ""
    similar_context_count: int = 0


class SimilarContextItem(BaseModel):
    """A single similar context result."""
    id: str
    content: str
    role: str = ""
    distance: float = 0.0
    token_estimate: int = 0


class SimilarContextResponse(BaseModel):
    """Response for similar context search."""
    query: str
    results: List[SimilarContextItem]
    total: int


# API Endpoints
@router.get("/{session_id}/stats", response_model=ContextStatsResponse)
async def get_context_stats(session_id: int) -> ContextStatsResponse:
    """
    Get vector context statistics for a session.
    
    Returns information about the stored context including:
    - Total number of vectors (embeddings)
    - Message counts by role (user/assistant/system)
    - Estimated token usage and context window utilization
    - Timestamp range of stored messages
    """
    vcm = get_vector_context_manager()
    stats = vcm.get_context_stats(session_id)
    
    return ContextStatsResponse(
        session_id=stats.session_id,
        total_vectors=stats.total_vectors,
        user_message_count=stats.user_message_count,
        assistant_message_count=stats.assistant_message_count,
        system_message_count=stats.system_message_count,
        total_tokens_estimate=stats.total_tokens_estimate,
        context_window_usage=stats.context_window_usage,
        max_context_tokens=stats.max_context_tokens,
        oldest_message_time=stats.oldest_message_time,
        newest_message_time=stats.newest_message_time,
        collection_name=stats.collection_name,
        similar_context_count=stats.similar_context_count,
    )


@router.get("/{session_id}/search", response_model=SimilarContextResponse)
async def search_context(
    session_id: int,
    query: str = Query(..., min_length=1, description="Search query text"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results"),
) -> SimilarContextResponse:
    """
    Search for semantically similar messages in a session's context.
    
    Uses ChromaDB's vector similarity search to find messages that
    are semantically related to the query text.
    """
    vcm = get_vector_context_manager()
    results = vcm.search_similar(session_id, query, top_k=top_k)
    
    items = []
    for r in results:
        meta = r.get("metadata", {})
        items.append(SimilarContextItem(
            id=r.get("id", ""),
            content=r.get("content", ""),
            role=meta.get("role", ""),
            distance=r.get("distance", 0.0),
            token_estimate=int(meta.get("token_estimate", 0)),
        ))
    
    return SimilarContextResponse(
        query=query,
        results=items,
        total=len(items),
    )


@router.delete("/{session_id}")
async def delete_session_context(session_id: int):
    """Delete all vector context for a session."""
    vcm = get_vector_context_manager()
    deleted = vcm.delete_session_context(session_id)
    
    return {
        "session_id": session_id,
        "deleted": deleted,
        "message": f"Context for session {session_id} {'deleted' if deleted else 'not found'}",
    }
