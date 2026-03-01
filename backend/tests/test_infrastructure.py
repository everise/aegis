"""
Unit tests for infrastructure services.

Tests dual retrieval, model router, and governance.
"""

import pytest
from datetime import datetime, timedelta

from app.services.dual_retrieval import (
    DualLevelRetriever,
    BM25Retriever,
    SemanticRetriever,
    Document,
    ImageGenerationKnowledgeBase,
    tokenize,
    reciprocal_rank_fusion,
)
from app.services.model_router import (
    ModelRouter,
    ModelEndpoint,
    ModelCapability,
    RoutingStrategy,
    create_image_generation_router,
)
from app.services.governance import (
    GovernanceManager,
    ContentModerator,
    RateLimiter,
    AccessController,
    AuditLogger,
    GovernanceAction,
    GovernanceDecision,
    ViolationType,
)


# ============== Dual Retrieval Tests ==============

class TestTokenizer:
    """Tests for CJK-aware tokenizer."""

    def test_english_tokenization(self):
        """Test English text tokenization with stop-word removal."""
        tokens = tokenize("The quick brown fox jumps over the lazy dog")
        assert "quick" in tokens
        assert "fox" in tokens
        # Stop words should be removed
        assert "the" not in tokens
        assert "is" not in tokens

    def test_chinese_tokenization(self):
        """Test Chinese text is split into unigrams."""
        tokens = tokenize("美丽的日落")
        assert "美" in tokens
        assert "丽" in tokens
        assert "日" in tokens
        assert "落" in tokens

    def test_mixed_cjk_english(self):
        """Test mixed Chinese-English tokenization."""
        tokens = tokenize("8K分辨率 photorealistic style")
        assert "分" in tokens
        assert "辨" in tokens
        assert "photorealistic" in tokens
        assert "style" in tokens


class TestRRF:
    """Tests for Reciprocal Rank Fusion."""

    def test_rrf_fusion(self):
        """Test RRF correctly fuses two ranked lists."""
        list1 = [("a", 10.0), ("b", 8.0), ("c", 5.0)]
        list2 = [("b", 0.9), ("c", 0.8), ("d", 0.7)]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        ids = [doc_id for doc_id, _ in fused]
        # "b" appears in both lists so should rank highest
        assert ids[0] == "b"
        assert "a" in ids
        assert "d" in ids

    def test_rrf_single_list(self):
        """Test RRF with a single list."""
        single = [("x", 5.0), ("y", 3.0)]
        fused = reciprocal_rank_fusion([single], k=60)
        assert fused[0][0] == "x"
        assert fused[1][0] == "y"


class TestBM25Retriever:
    """Tests for BM25Retriever."""
    
    @pytest.mark.asyncio
    async def test_add_and_retrieve(self):
        """Test adding documents and retrieving."""
        retriever = BM25Retriever()
        
        docs = [
            Document(doc_id="1", content="The quick brown fox"),
            Document(doc_id="2", content="The lazy dog sleeps"),
            Document(doc_id="3", content="Quick foxes are fast"),
        ]
        await retriever.add_documents(docs)
        
        results = await retriever.retrieve("quick fox", top_k=2)
        
        assert len(results) == 2
        # Documents with "quick" and "fox" should rank higher
        doc_ids = [doc.doc_id for doc, _ in results]
        assert "1" in doc_ids or "3" in doc_ids
    
    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Test retrieval with empty query."""
        retriever = BM25Retriever()
        await retriever.add_documents([Document(doc_id="1", content="test document")])
        
        results = await retriever.retrieve("", top_k=5)
        
        assert results == []

    @pytest.mark.asyncio
    async def test_remove_documents(self):
        """Test removing documents from the index."""
        retriever = BM25Retriever()
        docs = [
            Document(doc_id="1", content="first document"),
            Document(doc_id="2", content="second document"),
        ]
        await retriever.add_documents(docs)
        assert retriever.count() == 2

        await retriever.remove_documents(["1"])
        assert retriever.count() == 1

        results = await retriever.retrieve("first", top_k=5)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_chinese_retrieval(self):
        """Test BM25 retrieval with Chinese content."""
        retriever = BM25Retriever()
        docs = [
            Document(doc_id="1", content="水墨画风格的山水"),
            Document(doc_id="2", content="油画风格的肖像"),
            Document(doc_id="3", content="动漫风格的角色"),
        ]
        await retriever.add_documents(docs)
        results = await retriever.retrieve("水墨", top_k=2)
        assert len(results) >= 1
        assert results[0][0].doc_id == "1"


class TestSemanticRetriever:
    """Tests for SemanticRetriever (ChromaDB-backed)."""
    
    @pytest.mark.asyncio
    async def test_add_and_retrieve(self):
        """Test semantic retrieval with real embeddings."""
        retriever = SemanticRetriever()  # in-memory ChromaDB
        
        docs = [
            Document(doc_id="1", content="Machine learning algorithms"),
            Document(doc_id="2", content="Deep neural networks"),
            Document(doc_id="3", content="Cooking recipes for dinner"),
        ]
        await retriever.add_documents(docs)
        
        results = await retriever.retrieve("artificial intelligence", top_k=2)
        
        assert len(results) == 2
        # ML/DL docs should rank higher than cooking
        doc_ids = [doc.doc_id for doc, _ in results]
        assert "3" not in doc_ids or doc_ids.index("3") > 0

    @pytest.mark.asyncio
    async def test_similarity_scores(self):
        """Test that similarity scores are in [0, 1]."""
        retriever = SemanticRetriever()
        await retriever.add_documents([
            Document(doc_id="1", content="Beautiful sunset over the ocean"),
        ])
        results = await retriever.retrieve("sunset", top_k=1)
        assert len(results) == 1
        _, score = results[0]
        assert 0.0 <= score <= 1.0


class TestDualLevelRetriever:
    """Tests for DualLevelRetriever (BM25 + Semantic + RRF)."""
    
    @pytest.mark.asyncio
    async def test_dual_retrieval(self):
        """Test two-stage retrieval with RRF fusion."""
        retriever = DualLevelRetriever(coarse_k=10, fine_k=10)
        
        docs = [
            Document(doc_id=f"doc-{i}", content=f"Document about topic {i}")
            for i in range(20)
        ]
        await retriever.add_documents(docs)
        
        response = await retriever.retrieve("topic", top_k=3)
        
        assert len(response.results) == 3
        assert response.coarse_time_ms >= 0
        assert response.fine_time_ms >= 0
        assert response.fusion_time_ms >= 0
        # Results should use RRF fusion stage
        assert all(r.retrieval_stage == "rrf_fused" for r in response.results)

    @pytest.mark.asyncio
    async def test_empty_retrieval(self):
        """Test retrieval with no documents."""
        retriever = DualLevelRetriever()
        response = await retriever.retrieve("anything", top_k=5)
        assert len(response.results) == 0
        assert response.total_candidates == 0


class TestImageGenerationKnowledgeBase:
    """Tests for ImageGenerationKnowledgeBase."""
    
    @pytest.mark.asyncio
    async def test_query_knowledge(self):
        """Test querying the knowledge base."""
        kb = ImageGenerationKnowledgeBase()
        
        results = await kb.query("how to improve image quality", top_k=3)
        
        assert len(results) <= 3
        assert all(hasattr(r, 'document') for r in results)
    
    @pytest.mark.asyncio
    async def test_prompt_suggestions(self):
        """Test getting prompt suggestions."""
        kb = ImageGenerationKnowledgeBase()
        
        suggestions = await kb.get_prompt_suggestions("a sunset over mountains")
        
        assert "original_prompt" in suggestions
        assert "relevant_knowledge" in suggestions
        assert "enhancement_tips" in suggestions

    @pytest.mark.asyncio
    async def test_augmented_context(self):
        """Test augmented context generation for planner."""
        kb = ImageGenerationKnowledgeBase()

        context = await kb.get_augmented_context("oil painting of a sunset")

        assert "[Retrieved Knowledge]" in context
        assert "[/Retrieved Knowledge]" in context

    @pytest.mark.asyncio
    async def test_chinese_query(self):
        """Test knowledge base works with Chinese queries."""
        kb = ImageGenerationKnowledgeBase()

        results = await kb.query("如何提高图像质量", top_k=3)
        assert len(results) >= 1


# ============== Model Router Tests ==============

class TestModelEndpoint:
    """Tests for ModelEndpoint."""
    
    def test_can_handle_capability(self):
        """Test capability checking."""
        endpoint = ModelEndpoint(
            endpoint_id="test",
            name="Test",
            base_url="http://test",
            capabilities=[ModelCapability.TEXT_TO_IMAGE],
        )
        
        assert endpoint.can_handle(ModelCapability.TEXT_TO_IMAGE)
        assert not endpoint.can_handle(ModelCapability.INPAINTING)
    
    def test_load_factor(self):
        """Test load factor calculation."""
        endpoint = ModelEndpoint(
            endpoint_id="test",
            name="Test",
            base_url="http://test",
            capabilities=[],
            max_concurrent=10,
            current_load=5,
        )
        
        assert endpoint.load_factor == 0.5
    
    def test_record_request(self):
        """Test recording request statistics."""
        endpoint = ModelEndpoint(
            endpoint_id="test",
            name="Test",
            base_url="http://test",
            capabilities=[],
        )
        
        endpoint.record_request(success=True, latency_ms=100)
        endpoint.record_request(success=False, latency_ms=0)
        
        assert endpoint.success_count == 1
        assert endpoint.error_count == 1


class TestModelRouter:
    """Tests for ModelRouter."""
    
    def test_register_endpoint(self):
        """Test registering endpoints."""
        router = ModelRouter()
        
        endpoint = ModelEndpoint(
            endpoint_id="test",
            name="Test",
            base_url="http://test",
            capabilities=[ModelCapability.TEXT_TO_IMAGE],
        )
        router.register_endpoint(endpoint)
        
        assert router.get_endpoint("test") is endpoint
    
    def test_route_to_available_endpoint(self):
        """Test routing to available endpoint."""
        router = ModelRouter()
        
        endpoint = ModelEndpoint(
            endpoint_id="test",
            name="Test",
            base_url="http://test",
            capabilities=[ModelCapability.TEXT_TO_IMAGE],
        )
        router.register_endpoint(endpoint)
        
        decision = router.route(ModelCapability.TEXT_TO_IMAGE)
        
        assert decision.endpoint.endpoint_id == "test"
    
    def test_route_no_available_endpoint(self):
        """Test routing when no endpoint available."""
        router = ModelRouter()
        
        with pytest.raises(ValueError):
            router.route(ModelCapability.INPAINTING)
    
    def test_acquire_and_release(self):
        """Test acquiring and releasing endpoints."""
        router = ModelRouter()
        
        endpoint = ModelEndpoint(
            endpoint_id="test",
            name="Test",
            base_url="http://test",
            capabilities=[ModelCapability.TEXT_TO_IMAGE],
            max_concurrent=10,
        )
        router.register_endpoint(endpoint)
        
        acquired = router.acquire_endpoint(ModelCapability.TEXT_TO_IMAGE)
        assert acquired.current_load == 1
        
        router.release_endpoint(acquired, success=True, latency_ms=100)
        assert acquired.current_load == 0


class TestCreateImageGenerationRouter:
    """Tests for pre-configured router."""
    
    def test_has_required_endpoints(self):
        """Test pre-configured router has required endpoints."""
        router = create_image_generation_router()
        
        # Should have endpoints for various capabilities
        text_to_image = router.list_endpoints(ModelCapability.TEXT_TO_IMAGE)
        assert len(text_to_image) > 0
        
        evaluation = router.list_endpoints(ModelCapability.EVALUATION)
        assert len(evaluation) > 0


# ============== Governance Tests ==============

class TestContentModerator:
    """Tests for ContentModerator."""
    
    def test_safe_content_allowed(self):
        """Test safe content is allowed."""
        moderator = ContentModerator()
        
        decision = moderator.check_content("A beautiful sunset over the ocean")
        
        assert decision.allowed
        assert decision.action == GovernanceAction.ALLOW
    
    def test_unsafe_content_blocked(self):
        """Test unsafe content is blocked."""
        moderator = ContentModerator()
        
        decision = moderator.check_content("Content with violence and gore")
        
        assert not decision.allowed
        assert decision.action == GovernanceAction.BLOCK
        assert ViolationType.CONTENT_UNSAFE in decision.violations
    
    def test_sensitive_content_warning(self):
        """Test sensitive content gets warning."""
        moderator = ContentModerator(strict_mode=False)
        
        decision = moderator.check_content("A political rally scene")
        
        assert decision.allowed
        assert decision.action == GovernanceAction.WARN


class TestRateLimiter:
    """Tests for RateLimiter."""
    
    def test_allows_within_limit(self):
        """Test requests within limit are allowed."""
        limiter = RateLimiter(requests_per_minute=10, burst_size=5)
        
        # First few requests should be allowed
        for _ in range(5):
            decision = limiter.check_rate_limit("user1")
            assert decision.allowed
    
    def test_blocks_over_limit(self):
        """Test requests over limit are blocked."""
        limiter = RateLimiter(requests_per_minute=10, burst_size=2)
        
        # Exhaust burst
        limiter.check_rate_limit("user1")
        limiter.check_rate_limit("user1")
        
        # Next should be rate limited
        decision = limiter.check_rate_limit("user1")
        assert not decision.allowed
        assert decision.action == GovernanceAction.RATE_LIMIT


class TestAccessController:
    """Tests for AccessController."""
    
    def test_default_user_permissions(self):
        """Test default user has basic permissions."""
        controller = AccessController()
        
        decision = controller.check_access("new_user", "view")
        assert decision.allowed
    
    def test_admin_has_all_permissions(self):
        """Test admin role has all permissions."""
        controller = AccessController()
        controller.assign_role("admin_user", "admin")
        
        decision = controller.check_access("admin_user", "any_action")
        assert decision.allowed
    
    def test_unauthorized_action_blocked(self):
        """Test unauthorized action is blocked."""
        controller = AccessController()
        
        # Default user doesn't have 'train' permission
        decision = controller.check_access("regular_user", "train")
        assert not decision.allowed
        assert ViolationType.UNAUTHORIZED in decision.violations


class TestAuditLogger:
    """Tests for AuditLogger."""
    
    def test_log_entry_created(self):
        """Test logging creates entry."""
        logger = AuditLogger()
        
        from app.services.governance import GovernanceDecision
        
        decision = GovernanceDecision(
            action=GovernanceAction.ALLOW,
            allowed=True,
            reason="Test",
        )
        
        entry = logger.log(
            action="test_action",
            resource="test_resource",
            decision=decision,
            user_id="user1",
        )
        
        assert entry.entry_id is not None
        assert entry.action == "test_action"
    
    def test_query_by_user(self):
        """Test querying logs by user."""
        logger = AuditLogger()
        
        decision = GovernanceDecision(
            action=GovernanceAction.ALLOW,
            allowed=True,
            reason="Test",
        )
        
        logger.log("action1", "resource", decision, user_id="user1")
        logger.log("action2", "resource", decision, user_id="user2")
        
        results = logger.query(user_id="user1")
        
        assert len(results) == 1
        assert results[0].user_id == "user1"


class TestGovernanceManager:
    """Tests for GovernanceManager."""
    
    def test_comprehensive_check_passes(self):
        """Test request passing all checks."""
        manager = GovernanceManager()
        
        decision = manager.check_request(
            user_id="user1",
            action="view",
            content="Safe content",
            resource="images",
        )
        
        assert decision.allowed
    
    def test_content_moderation_blocks(self):
        """Test content moderation blocking request."""
        manager = GovernanceManager()
        
        decision = manager.check_request(
            user_id="user1",
            action="generate",
            content="Generate violence and gore",
        )
        
        assert not decision.allowed
    
    def test_statistics_tracking(self):
        """Test audit statistics are tracked."""
        manager = GovernanceManager()
        
        manager.check_request("user1", "view")
        manager.check_request("user2", "generate")
        
        stats = manager.get_audit_statistics()
        
        assert stats["total_entries"] == 2
