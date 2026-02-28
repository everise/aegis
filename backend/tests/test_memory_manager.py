"""
Tests for the memory_manager module.

Covers:
- MemoryMessage creation and serialisation
- TokenCounter estimation
- HeuristicCompressor compression logic
- SessionMemory auto-compression lifecycle
- MemoryManager global API
- Multimodal-specific behaviour (image URL preservation, quality scores)
"""

import asyncio
import copy
import time
from typing import Any, Dict, List, Optional

import pytest
import pytest_asyncio

from app.services.memory_manager import (
    BaseCompressor,
    CompressionResult,
    CompressionStrategy,
    HeuristicCompressor,
    LLMCompressor,
    MemoryManager,
    MemoryMessage,
    MemoryRole,
    MemoryStats,
    SessionMemory,
    TokenCounter,
    _extract_urls,
    _truncate,
    get_memory_manager,
    reset_memory_manager,
)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _make_msg(
    role: str = "user",
    content: str = "hello",
    image_urls: Optional[List[str]] = None,
    plan_json: Optional[Dict[str, Any]] = None,
    quality_score: Optional[float] = None,
) -> MemoryMessage:
    return MemoryMessage(
        role=MemoryRole(role),
        content=content,
        image_urls=image_urls or [],
        plan_json=plan_json,
        quality_score=quality_score,
    )


def _make_assistant_with_plan(
    image_url: str = "https://example.com/img.png",
    score: float = 0.85,
) -> MemoryMessage:
    """Create a realistic assistant message with plan_json."""
    return _make_msg(
        role="assistant",
        content=f"Generated image: {image_url}",
        image_urls=[image_url],
        quality_score=score,
        plan_json={
            "steps": [
                {
                    "action": "generate",
                    "thought": "I will generate the image requested by the user",
                    "action_input": {"skill": "text_to_image", "params": {"prompt": "a cat"}},
                    "observation": {
                        "skill_name": "text_to_image",
                        "status": "completed",
                        "result": {"image_url": image_url},
                    },
                    "status": "completed",
                },
                {
                    "action": "evaluate",
                    "thought": "Now evaluating the image quality",
                    "action_input": {"skill": "evaluate_image", "params": {"image_url": image_url}},
                    "observation": {
                        "skill_name": "evaluate_image",
                        "status": "completed",
                        "result": {"overall_score": score, "image_url": image_url},
                    },
                    "status": "completed",
                },
            ],
            "status": "completed",
            "final_result": {"image_url": image_url, "result": "success"},
        },
    )


# ──────────────────────────────────────────────────────────────────
# MemoryMessage tests
# ──────────────────────────────────────────────────────────────────

class TestMemoryMessage:
    def test_to_dict_minimal(self):
        msg = _make_msg()
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"
        assert "image_urls" not in d  # empty list omitted

    def test_to_dict_with_multimodal(self):
        msg = _make_msg(
            role="assistant",
            content="result",
            image_urls=["https://img.png"],
            quality_score=0.9,
        )
        d = msg.to_dict()
        assert d["image_urls"] == ["https://img.png"]
        assert d["quality_score"] == 0.9

    def test_roundtrip_serialisation(self):
        original = _make_assistant_with_plan()
        d = original.to_dict()
        restored = MemoryMessage.from_dict(d)
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.image_urls == original.image_urls
        assert restored.quality_score == original.quality_score

    def test_compressed_flag(self):
        msg = _make_msg()
        msg.is_compressed = True
        msg.original_count = 5
        d = msg.to_dict()
        assert d["is_compressed"] is True
        assert d["original_count"] == 5


# ──────────────────────────────────────────────────────────────────
# TokenCounter tests
# ──────────────────────────────────────────────────────────────────

class TestTokenCounter:
    def test_count_text(self):
        tc = TokenCounter()
        assert tc.count_text("") == 0
        # 20 chars / 4.0 = 5
        assert tc.count_text("a" * 20) == 5

    def test_count_message_includes_overhead(self):
        tc = TokenCounter()
        msg = _make_msg(content="a" * 100)
        tokens = tc.count_message(msg)
        # 100/4 + 4 overhead = 29
        assert tokens == 29

    def test_image_ref_overhead(self):
        tc = TokenCounter()
        msg = _make_msg(content="x", image_urls=["https://img1.png", "https://img2.png"])
        tokens = tc.count_message(msg)
        # text(1/4=1) + 2*85 + 4 = 175
        assert tokens == 175


# ──────────────────────────────────────────────────────────────────
# HeuristicCompressor tests
# ──────────────────────────────────────────────────────────────────

class TestHeuristicCompressor:
    @pytest.mark.asyncio
    async def test_no_compression_needed(self):
        compressor = HeuristicCompressor()
        tc = TokenCounter()
        msgs = [_make_msg(content="short")]
        result = await compressor.compress(msgs, max_tokens=10_000, token_counter=tc)
        assert result.compressed_messages == msgs
        assert result.tokens_before == result.tokens_after

    @pytest.mark.asyncio
    async def test_compression_preserves_system_messages(self):
        compressor = HeuristicCompressor(protected_pairs=1)
        tc = TokenCounter()
        msgs = [
            _make_msg(role="system", content="You are a helpful agent."),
            _make_msg(role="user", content="Old request " * 50),
            _make_assistant_with_plan(),
            _make_msg(role="user", content="New request"),
            _make_msg(role="assistant", content="New response"),
        ]
        result = await compressor.compress(msgs, max_tokens=500, token_counter=tc)
        roles = [m.role for m in result.compressed_messages]
        assert MemoryRole.SYSTEM in roles
        assert result.compressed_count < result.original_count

    @pytest.mark.asyncio
    async def test_compression_preserves_image_urls(self):
        compressor = HeuristicCompressor(protected_pairs=0)
        tc = TokenCounter()
        url = "https://example.com/generated.png"
        msgs = [
            _make_msg(role="user", content="Generate an image " * 100),
            _make_assistant_with_plan(image_url=url),
            _make_msg(role="user", content="Another request " * 100),
            _make_msg(role="assistant", content="Another response " * 100),
        ]
        result = await compressor.compress(msgs, max_tokens=200, token_counter=tc)

        # Image URL must survive compression
        all_urls = []
        for m in result.compressed_messages:
            all_urls.extend(m.image_urls)
        assert url in all_urls

    @pytest.mark.asyncio
    async def test_compression_preserves_quality_score(self):
        compressor = HeuristicCompressor(protected_pairs=0)
        tc = TokenCounter()
        # Need enough messages to trigger compression (protected_pairs=0
        # still protects a trailing user message, so add more)
        msgs = [
            _make_msg(role="user", content="Make image " * 80),
            _make_assistant_with_plan(score=0.72),
            _make_msg(role="user", content="Follow up " * 80),
            _make_msg(role="assistant", content="Response " * 80),
        ]
        result = await compressor.compress(msgs, max_tokens=100, token_counter=tc)

        # Quality score must survive in the compressed block
        all_scores = [m.quality_score for m in result.compressed_messages if m.quality_score is not None]
        assert 0.72 in all_scores

    @pytest.mark.asyncio
    async def test_empty_input(self):
        compressor = HeuristicCompressor()
        tc = TokenCounter()
        result = await compressor.compress([], max_tokens=100, token_counter=tc)
        assert result.compressed_messages == []


# ──────────────────────────────────────────────────────────────────
# LLMCompressor tests
# ──────────────────────────────────────────────────────────────────

class TestLLMCompressor:
    @pytest.mark.asyncio
    async def test_llm_compressor_calls_fn(self):
        call_log = []

        async def mock_llm(prompt: str) -> str:
            call_log.append(prompt)
            return "<compressed_memory>Summary of conversation</compressed_memory>"

        compressor = LLMCompressor(llm_fn=mock_llm)
        tc = TokenCounter()
        msgs = [
            _make_msg(role="user", content="request " * 50),
            _make_assistant_with_plan(),
        ]
        result = await compressor.compress(msgs, max_tokens=500, token_counter=tc)
        assert len(call_log) == 1
        assert result.strategy == CompressionStrategy.LLM
        assert any(m.is_compressed for m in result.compressed_messages)

    @pytest.mark.asyncio
    async def test_llm_compressor_fallback_on_error(self):
        """Should fall back to heuristic when LLM fails."""
        async def failing_llm(prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        compressor = LLMCompressor(llm_fn=failing_llm)
        tc = TokenCounter()
        msgs = [
            _make_msg(role="user", content="request " * 50),
            _make_msg(role="assistant", content="response " * 50),
        ]
        result = await compressor.compress(msgs, max_tokens=500, token_counter=tc)
        # Falls back to heuristic — should still produce a result
        assert result.compressed_messages is not None

    @pytest.mark.asyncio
    async def test_preserves_image_urls_from_originals(self):
        url = "https://example.com/img.png"

        async def mock_llm(prompt: str) -> str:
            return "<compressed_memory>short summary</compressed_memory>"

        compressor = LLMCompressor(llm_fn=mock_llm)
        tc = TokenCounter()
        msgs = [_make_assistant_with_plan(image_url=url)]
        result = await compressor.compress(msgs, max_tokens=500, token_counter=tc)

        all_urls = []
        for m in result.compressed_messages:
            all_urls.extend(m.image_urls)
        assert url in all_urls


# ──────────────────────────────────────────────────────────────────
# SessionMemory tests
# ──────────────────────────────────────────────────────────────────

class TestSessionMemory:
    @pytest.mark.asyncio
    async def test_add_and_get(self):
        mem = SessionMemory(session_id=1, max_tokens=100_000)
        await mem.add(_make_msg(content="hello"))
        ctx = await mem.get_context()
        assert len(ctx) == 1
        assert ctx[0].content == "hello"

    @pytest.mark.asyncio
    async def test_auto_compression_on_add(self):
        """Memory should compress automatically when token limit is exceeded."""
        mem = SessionMemory(session_id=1, max_tokens=200, compress_on_add=True)

        # Add enough messages to exceed 200 tokens
        for i in range(10):
            await mem.add(_make_msg(content=f"Message number {i} with some extra padding text " * 5))

        stats = mem.stats()
        # Compression should have been triggered at least once
        assert stats.compression_count >= 1
        # Token count should be reduced
        assert stats.total_tokens <= 400  # generous upper bound

    @pytest.mark.asyncio
    async def test_auto_compression_on_get(self):
        """Compression should also work lazily on get_context()."""
        mem = SessionMemory(session_id=1, max_tokens=200, compress_on_add=False, compress_on_get=True)

        for i in range(10):
            await mem.add(_make_msg(content=f"Verbose message {i} " * 10))

        # Before get, no compression yet
        assert mem._compression_count == 0

        ctx = await mem.get_context()
        assert mem._compression_count >= 1

    @pytest.mark.asyncio
    async def test_get_context_for_planning(self):
        mem = SessionMemory(session_id=1, max_tokens=100_000)
        await mem.add(_make_msg(role="user", content="Generate a cat"))
        await mem.add(_make_msg(role="assistant", content="Done"))

        formatted = await mem.get_context_for_planning()
        assert isinstance(formatted, list)
        assert formatted[0]["role"] == "user"
        assert formatted[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_compressed_block_becomes_system_in_planning(self):
        """Compressed blocks should have system role for planner awareness."""
        mem = SessionMemory(session_id=1, max_tokens=150, compress_on_add=True)

        for i in range(8):
            await mem.add(_make_msg(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Turn {i} text with padding " * 8,
            ))

        formatted = await mem.get_context_for_planning()
        # There should be at least one system-role compressed block
        system_msgs = [m for m in formatted if m["role"] == "system"]
        compressed_markers = [m for m in system_msgs if "<compressed_memory>" in m["content"]]
        assert len(compressed_markers) >= 1

    @pytest.mark.asyncio
    async def test_stats(self):
        mem = SessionMemory(session_id=42, max_tokens=100_000)
        await mem.add(_make_msg(content="test message"))
        stats = mem.stats()
        assert stats.session_id == 42
        assert stats.message_count == 1
        assert stats.total_tokens > 0

    @pytest.mark.asyncio
    async def test_clear(self):
        mem = SessionMemory(session_id=1, max_tokens=100_000)
        await mem.add(_make_msg(content="test"))
        await mem.clear()
        ctx = await mem.get_context()
        assert len(ctx) == 0

    @pytest.mark.asyncio
    async def test_state_dict_roundtrip(self):
        mem = SessionMemory(session_id=7, max_tokens=100_000)
        await mem.add(_make_msg(content="persistent"))
        state = mem.state_dict()

        mem2 = SessionMemory(session_id=0, max_tokens=1)
        mem2.load_state_dict(state)
        assert mem2.session_id == 7
        assert mem2.max_tokens == 100_000
        ctx = await mem2.get_context()
        assert len(ctx) == 1
        assert ctx[0].content == "persistent"

    @pytest.mark.asyncio
    async def test_recent_n(self):
        mem = SessionMemory(session_id=1, max_tokens=100_000)
        for i in range(5):
            await mem.add(_make_msg(content=f"msg-{i}"))
        ctx = await mem.get_context(recent_n=2)
        assert len(ctx) == 2
        assert ctx[-1].content == "msg-4"


# ──────────────────────────────────────────────────────────────────
# MemoryManager tests
# ──────────────────────────────────────────────────────────────────

class TestMemoryManager:
    @pytest.mark.asyncio
    async def test_add_and_retrieve(self):
        mgr = MemoryManager(default_max_tokens=100_000)
        await mgr.add_message(1, "user", "Hello")
        ctx = await mgr.get_context_for_planning(1)
        assert len(ctx) == 1
        assert ctx[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_session_isolation(self):
        mgr = MemoryManager(default_max_tokens=100_000)
        await mgr.add_message(1, "user", "Session 1")
        await mgr.add_message(2, "user", "Session 2")
        ctx1 = await mgr.get_context_for_planning(1)
        ctx2 = await mgr.get_context_for_planning(2)
        assert len(ctx1) == 1
        assert len(ctx2) == 1
        assert ctx1[0]["content"] == "Session 1"
        assert ctx2[0]["content"] == "Session 2"

    @pytest.mark.asyncio
    async def test_delete_session(self):
        mgr = MemoryManager(default_max_tokens=100_000)
        await mgr.add_message(1, "user", "test")
        mgr.delete_session(1)
        assert 1 not in mgr.list_sessions()

    @pytest.mark.asyncio
    async def test_multimodal_message(self):
        mgr = MemoryManager(default_max_tokens=100_000)
        await mgr.add_message(
            session_id=1,
            role="assistant",
            content="Generated image",
            image_urls=["https://example.com/img.png"],
            quality_score=0.88,
        )
        mem = mgr.get_session_memory(1)
        ctx = await mem.get_context()
        assert ctx[0].image_urls == ["https://example.com/img.png"]
        assert ctx[0].quality_score == 0.88

    @pytest.mark.asyncio
    async def test_stats(self):
        mgr = MemoryManager(default_max_tokens=5000)
        await mgr.add_message(1, "user", "test")
        stats = mgr.get_stats(1)
        assert stats.session_id == 1
        assert stats.max_tokens == 5000
        assert stats.message_count == 1


# ──────────────────────────────────────────────────────────────────
# Singleton tests
# ──────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_returns_same_instance(self):
        reset_memory_manager()
        a = get_memory_manager()
        b = get_memory_manager()
        assert a is b

    def test_reset(self):
        reset_memory_manager()
        a = get_memory_manager()
        reset_memory_manager()
        b = get_memory_manager()
        assert a is not b


# ──────────────────────────────────────────────────────────────────
# Utility tests
# ──────────────────────────────────────────────────────────────────

class TestUtils:
    def test_extract_urls(self):
        text = "Check https://example.com/img.png and http://other.com/file"
        urls = _extract_urls(text)
        assert len(urls) == 2
        assert "https://example.com/img.png" in urls

    def test_truncate(self):
        assert _truncate("short", 10) == "short"
        assert _truncate("a" * 20, 10) == "a" * 7 + "..."

    def test_truncate_newlines(self):
        assert "\n" not in _truncate("line1\nline2", 50)
