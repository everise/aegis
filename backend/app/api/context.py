"""
Context API endpoints.

Provides endpoints for querying vector context statistics,
performing semantic search over conversation history, and
inspecting working memory compression state.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.vector_context import get_vector_context_manager, ContextStats
from app.services.memory_manager import get_memory_manager
from app.services.dual_retrieval import get_knowledge_base


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
    
    # Also clean up working memory
    mm = get_memory_manager()
    mm.delete_session(session_id)
    
    return {
        "session_id": session_id,
        "deleted": deleted,
        "message": f"Context for session {session_id} {'deleted' if deleted else 'not found'}",
    }


# ── Working Memory endpoints ─────────────────────────────────────

class MemoryStatsResponse(BaseModel):
    """Response model for working memory statistics."""
    session_id: int
    message_count: int
    compressed_count: int
    total_tokens: int
    max_tokens: int
    usage_ratio: float
    image_url_count: int
    compression_count: int


class MemoryMessageResponse(BaseModel):
    """A single message in working memory."""
    role: str
    content: str
    is_compressed: bool = False
    original_count: int = 1
    image_urls: List[str] = []
    quality_score: Optional[float] = None
    token_estimate: int = 0


class MemoryContextResponse(BaseModel):
    """Response for working memory context."""
    session_id: int
    messages: List[MemoryMessageResponse]
    total_tokens: int
    is_compressed: bool


@router.get("/{session_id}/memory/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(session_id: int) -> MemoryStatsResponse:
    """
    Get working memory statistics for a session.
    
    Returns compression state, token usage, and capacity information
    for the in-memory context window.
    """
    mm = get_memory_manager()
    stats = mm.get_stats(session_id)
    return MemoryStatsResponse(
        session_id=stats.session_id,
        message_count=stats.message_count,
        compressed_count=stats.compressed_count,
        total_tokens=stats.total_tokens,
        max_tokens=stats.max_tokens,
        usage_ratio=stats.usage_ratio,
        image_url_count=stats.image_url_count,
        compression_count=stats.compression_count,
    )


@router.get("/{session_id}/memory/context", response_model=MemoryContextResponse)
async def get_memory_context(session_id: int) -> MemoryContextResponse:
    """
    Get the current working memory context for a session.
    
    Returns the (potentially compressed) message list that would
    be sent to the planning model.
    """
    mm = get_memory_manager()
    mem = mm.get_session_memory(session_id)
    messages = await mem.get_context()
    
    total_tokens = sum(m.token_estimate or 0 for m in messages)
    has_compressed = any(m.is_compressed for m in messages)
    
    return MemoryContextResponse(
        session_id=session_id,
        messages=[
            MemoryMessageResponse(
                role=m.role.value,
                content=m.content,
                is_compressed=m.is_compressed,
                original_count=m.original_count,
                image_urls=m.image_urls,
                quality_score=m.quality_score,
                token_estimate=m.token_estimate,
            )
            for m in messages
        ],
        total_tokens=total_tokens,
        is_compressed=has_compressed,
    )


@router.delete("/{session_id}/memory")
async def clear_memory(session_id: int):
    """Clear working memory for a session (does not affect SQLite or vector DB)."""
    mm = get_memory_manager()
    mem = mm.get_session_memory(session_id)
    await mem.clear()
    return {
        "session_id": session_id,
        "message": "Working memory cleared",
    }


# ── Dual-Level Retrieval endpoints ────────────────────────────────

class RetrievalResultItem(BaseModel):
    """A single retrieval result."""
    doc_id: str
    content: str
    category: str = ""
    score: float = 0.0
    rank: int = 0
    retrieval_stage: str = "rrf_fused"


class RetrievalQueryResponse(BaseModel):
    """Response for retrieval query."""
    query: str
    results: List[RetrievalResultItem]
    total_candidates: int
    coarse_time_ms: float
    fine_time_ms: float
    fusion_time_ms: float


class PromptSuggestionResponse(BaseModel):
    """Response for prompt suggestions."""
    original_prompt: str
    relevant_knowledge: List[dict]
    enhancement_tips: List[str]


@router.get("/retrieval/query", response_model=RetrievalQueryResponse)
async def retrieval_query(
    query: str = Query(..., min_length=1, description="Search query text"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results"),
) -> RetrievalQueryResponse:
    """
    Query the image-generation knowledge base using Dual-Level Retrieval.

    Uses BM25 coarse retrieval + ChromaDB semantic retrieval fused via
    Reciprocal Rank Fusion (RRF) to find the most relevant knowledge.
    """
    kb = get_knowledge_base()
    await kb.ensure_loaded()
    response = await kb.retriever.retrieve(query, top_k)

    items = [
        RetrievalResultItem(
            doc_id=r.document.doc_id,
            content=r.document.content,
            category=r.document.metadata.get("category", ""),
            score=r.score,
            rank=r.rank,
            retrieval_stage=r.retrieval_stage,
        )
        for r in response.results
    ]

    return RetrievalQueryResponse(
        query=query,
        results=items,
        total_candidates=response.total_candidates,
        coarse_time_ms=round(response.coarse_time_ms, 3),
        fine_time_ms=round(response.fine_time_ms, 3),
        fusion_time_ms=round(response.fusion_time_ms, 3),
    )


@router.get("/retrieval/suggest", response_model=PromptSuggestionResponse)
async def retrieval_suggest(
    prompt: str = Query(..., min_length=1, description="Base prompt to get suggestions for"),
    top_k: int = Query(default=5, ge=1, le=10, description="Number of knowledge items"),
) -> PromptSuggestionResponse:
    """
    Get prompt improvement suggestions from the knowledge base.

    Returns relevant knowledge and actionable tips to enhance the
    given image-generation prompt.
    """
    kb = get_knowledge_base()
    suggestions = await kb.get_prompt_suggestions(prompt, top_k=top_k)

    return PromptSuggestionResponse(
        original_prompt=suggestions["original_prompt"],
        relevant_knowledge=suggestions["relevant_knowledge"],
        enhancement_tips=suggestions["enhancement_tips"],
    )
