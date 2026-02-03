"""
Unit tests for infrastructure services.

Tests context pruner, dual retrieval, model router, and governance.
"""

import pytest
from datetime import datetime, timedelta

from app.services.context_pruner import (
    ContextPruner,
    TruncationPruner,
    SlidingWindowPruner,
    ImportancePruner,
    Message,
    PruningStrategy,
    TokenCounter,
)
from app.services.dual_retrieval import (
    DualLevelRetriever,
    BM25Retriever,
    SemanticRetriever,
    Document,
    ImageGenerationKnowledgeBase,
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


# ============== Context Pruner Tests ==============

class TestTokenCounter:
    """Tests for TokenCounter."""
    
    def test_count_simple_text(self):
        """Test counting tokens in simple text."""
        counter = TokenCounter(chars_per_token=4.0)
        
        # 20 chars / 4 = 5 tokens
        count = counter.count("This is a test text")
        assert count == 4  # 19 chars / 4 = 4.75 -> 4
    
    def test_count_message(self):
        """Test counting tokens in message with overhead."""
        counter = TokenCounter()
        msg = Message(role="user", content="Hello world")
        
        count = counter.count_message(msg)
        # Content + overhead
        assert count > counter.count("Hello world")


class TestTruncationPruner:
    """Tests for TruncationPruner."""
    
    def test_no_pruning_needed(self):
        """Test when messages fit within limit."""
        pruner = TruncationPruner(max_tokens=1000)
        
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        
        result = pruner.prune(messages)
        
        assert len(result.messages) == 2
        assert result.pruned_count == 0
    
    def test_pruning_from_start(self):
        """Test pruning removes oldest messages."""
        pruner = TruncationPruner(max_tokens=50)
        
        messages = [
            Message(role="user", content="Message 1" * 20),  # Long
            Message(role="user", content="Message 2" * 20),  # Long
            Message(role="user", content="Short"),  # Short
        ]
        
        result = pruner.prune(messages)
        
        # Should keep only recent messages that fit
        assert result.pruned_count > 0
    
    def test_system_messages_preserved(self):
        """Test system messages are always kept."""
        pruner = TruncationPruner(max_tokens=100)
        
        messages = [
            Message(role="system", content="System instruction"),
            Message(role="user", content="User message " * 50),
            Message(role="assistant", content="Response"),
        ]
        
        result = pruner.prune(messages)
        
        # System message should be preserved
        system_msgs = [m for m in result.messages if m.role == "system"]
        assert len(system_msgs) == 1


class TestSlidingWindowPruner:
    """Tests for SlidingWindowPruner."""
    
    def test_window_size_respected(self):
        """Test window size is respected."""
        pruner = SlidingWindowPruner(max_tokens=10000, window_size=3)
        
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(10)
        ]
        
        result = pruner.prune(messages)
        
        # Should keep only last 3 (window_size)
        assert len([m for m in result.messages if m.role != "system"]) == 3


class TestContextPruner:
    """Tests for main ContextPruner class."""
    
    def test_prune_conversation(self):
        """Test convenience method for conversation dicts."""
        pruner = ContextPruner(
            strategy=PruningStrategy.SLIDING_WINDOW,
            max_tokens=1000,
            window_size=2,
        )
        
        messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
        ]
        
        result = pruner.prune_conversation(messages)
        
        assert isinstance(result, list)
        assert all("role" in m and "content" in m for m in result)


# ============== Dual Retrieval Tests ==============

class TestBM25Retriever:
    """Tests for BM25Retriever."""
    
    def test_add_and_retrieve(self):
        """Test adding documents and retrieving."""
        retriever = BM25Retriever()
        
        docs = [
            Document(doc_id="1", content="The quick brown fox"),
            Document(doc_id="2", content="The lazy dog sleeps"),
            Document(doc_id="3", content="Quick foxes are fast"),
        ]
        retriever.add_documents(docs)
        
        results = retriever.retrieve("quick fox", top_k=2)
        
        assert len(results) == 2
        # Documents with "quick" and "fox" should rank higher
        doc_ids = [doc.doc_id for doc, _ in results]
        assert "1" in doc_ids or "3" in doc_ids
    
    def test_empty_query(self):
        """Test retrieval with empty query."""
        retriever = BM25Retriever()
        retriever.add_documents([Document(doc_id="1", content="test")])
        
        results = retriever.retrieve("", top_k=5)
        
        assert results == []


class TestSemanticRetriever:
    """Tests for SemanticRetriever."""
    
    def test_add_and_retrieve(self):
        """Test semantic retrieval."""
        retriever = SemanticRetriever()
        
        docs = [
            Document(doc_id="1", content="Machine learning algorithms"),
            Document(doc_id="2", content="Deep neural networks"),
            Document(doc_id="3", content="Cooking recipes"),
        ]
        retriever.add_documents(docs)
        
        results = retriever.retrieve("artificial intelligence", top_k=2)
        
        assert len(results) == 2


class TestDualLevelRetriever:
    """Tests for DualLevelRetriever."""
    
    def test_dual_retrieval(self):
        """Test two-stage retrieval."""
        retriever = DualLevelRetriever(coarse_k=10, fine_k=3)
        
        docs = [
            Document(doc_id=f"doc-{i}", content=f"Document about topic {i}")
            for i in range(20)
        ]
        retriever.add_documents(docs)
        
        response = retriever.retrieve("topic", top_k=3)
        
        assert len(response.results) == 3
        assert response.coarse_time_ms >= 0
        assert response.fine_time_ms >= 0


class TestImageGenerationKnowledgeBase:
    """Tests for ImageGenerationKnowledgeBase."""
    
    def test_query_knowledge(self):
        """Test querying the knowledge base."""
        kb = ImageGenerationKnowledgeBase()
        
        results = kb.query("how to improve image quality", top_k=3)
        
        assert len(results) <= 3
        assert all(hasattr(r, 'document') for r in results)
    
    def test_prompt_suggestions(self):
        """Test getting prompt suggestions."""
        kb = ImageGenerationKnowledgeBase()
        
        suggestions = kb.get_prompt_suggestions("a sunset over mountains")
        
        assert "original_prompt" in suggestions
        assert "relevant_knowledge" in suggestions


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
