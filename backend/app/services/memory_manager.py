"""
Memory Manager with automatic compression for multimodal agent.

Manages conversation memory per session with dual storage:
- Full history: Complete, unmodified message history (SQLite)
- Working memory: Compressed memory optimised for context window (in-memory)

Key differences from text-only LLM memory:
- Preserves image URLs and multimodal references during compression
- Retains critical observations (quality scores, skill results)
- Compresses verbose ReAct reasoning chains into concise summaries
- Keeps recent plan steps intact for planning continuity

Compression is triggered automatically when working memory exceeds
the configured token threshold. Both LLM-based and heuristic-based
compression strategies are supported.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)


# ──────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────

class MemoryRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    COMPRESSED = "compressed"  # synthetic role for compression summaries


@dataclass
class MemoryMessage:
    """
    A message in working memory.

    Extends the basic role/content pair with multimodal metadata
    so that compression logic can make informed decisions about
    what to keep and what to summarise.
    """
    role: MemoryRole
    content: str
    timestamp: float = field(default_factory=time.time)
    # ── Multimodal fields ────────────────────────────────────────
    image_urls: List[str] = field(default_factory=list)
    plan_json: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    skill_results: List[Dict[str, Any]] = field(default_factory=list)
    # ── Bookkeeping ──────────────────────────────────────────────
    token_estimate: int = 0
    is_compressed: bool = False
    original_count: int = 1  # how many original messages this represents
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.image_urls:
            d["image_urls"] = self.image_urls
        if self.plan_json:
            d["plan_json"] = self.plan_json
        if self.quality_score is not None:
            d["quality_score"] = self.quality_score
        if self.skill_results:
            d["skill_results"] = self.skill_results
        if self.is_compressed:
            d["is_compressed"] = True
            d["original_count"] = self.original_count
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryMessage":
        role = MemoryRole(d.get("role", "user"))
        return cls(
            role=role,
            content=d.get("content", ""),
            timestamp=d.get("timestamp", time.time()),
            image_urls=d.get("image_urls", []),
            plan_json=d.get("plan_json"),
            quality_score=d.get("quality_score"),
            skill_results=d.get("skill_results", []),
            token_estimate=d.get("token_estimate", 0),
            is_compressed=d.get("is_compressed", False),
            original_count=d.get("original_count", 1),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_db_message(cls, msg: Any) -> "MemoryMessage":
        """Build from SQLAlchemy ``Message`` model instance."""
        role = MemoryRole(msg.role.value if hasattr(msg.role, "value") else msg.role)
        image_urls: List[str] = []
        quality_score: Optional[float] = None
        skill_results: List[Dict[str, Any]] = []

        # Extract multimodal data from plan_json
        if msg.plan_json:
            plan = msg.plan_json
            for step in plan.get("steps", []):
                obs = step.get("observation") or step.get("data", {}).get("observation", {})
                if isinstance(obs, dict):
                    # Collect image URLs
                    result = obs.get("result", {})
                    if isinstance(result, dict):
                        url = result.get("image_url")
                        if url:
                            image_urls.append(url)
                        score = result.get("overall_score")
                        if score is not None:
                            quality_score = score
                    skill_results.append(obs)

            # Also check final_result
            fr = plan.get("final_result", {})
            if isinstance(fr, dict):
                url = fr.get("image_url")
                if url and url not in image_urls:
                    image_urls.append(url)

        # Detect image URLs in plain content
        for url in _extract_urls(msg.content or ""):
            if url not in image_urls:
                image_urls.append(url)

        ts = msg.created_at.timestamp() if msg.created_at else time.time()

        return cls(
            role=role,
            content=msg.content or "",
            timestamp=ts,
            image_urls=image_urls,
            plan_json=msg.plan_json,
            quality_score=quality_score,
            skill_results=skill_results,
            metadata={"db_message_id": msg.id, "session_id": msg.session_id},
        )


# ──────────────────────────────────────────────────────────────────
# Token estimation
# ──────────────────────────────────────────────────────────────────

class TokenCounter:
    """
    Estimates token count with configurable ratio.

    For multimodal content the image reference counts as a fixed
    token overhead (e.g. 85 tokens for a URL) rather than the
    full image embedding cost, because Aegis passes image URLs
    (not raw pixels) in planning context.
    """
    CHARS_PER_TOKEN = 4.0
    IMAGE_REF_TOKENS = 85  # fixed overhead per image URL

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / self.CHARS_PER_TOKEN))

    def count_message(self, msg: MemoryMessage) -> int:
        tokens = self.count_text(msg.content)
        tokens += len(msg.image_urls) * self.IMAGE_REF_TOKENS
        if msg.plan_json:
            tokens += self.count_text(json.dumps(msg.plan_json, ensure_ascii=False))
        tokens += 4  # per-message framing overhead
        return tokens


# ──────────────────────────────────────────────────────────────────
# Compression strategies
# ──────────────────────────────────────────────────────────────────

class CompressionStrategy(str, Enum):
    HEURISTIC = "heuristic"  # rule-based, no LLM needed
    LLM = "llm"              # uses an LLM to create a summary


@dataclass
class CompressionResult:
    compressed_messages: List[MemoryMessage]
    original_count: int
    compressed_count: int
    tokens_before: int
    tokens_after: int
    strategy: CompressionStrategy


class BaseCompressor(ABC):
    """Interface for memory compression implementations."""

    @abstractmethod
    async def compress(
        self,
        messages: List[MemoryMessage],
        max_tokens: int,
        token_counter: TokenCounter,
    ) -> CompressionResult:
        """Compress *messages* to fit within *max_tokens*."""
        ...


class HeuristicCompressor(BaseCompressor):
    """
    Rule-based compressor tailored for the multimodal agent.

    Strategy:
    1. System messages are always preserved in full.
    2. The *protected window* (last N user-assistant pairs) is kept
       intact so the planner has recent context.
    3. Older messages are compressed into a structured summary that
       preserves:
       - Image URLs and quality scores (critical for the agent)
       - Skill execution outcomes
       - A brief narrative of the conversation flow
    4. Intermediate ReAct reasoning chains are reduced to one-line
       summaries per step to save tokens.
    """

    def __init__(
        self,
        protected_pairs: int = 2,
        max_plan_steps_in_summary: int = 3,
    ):
        self.protected_pairs = protected_pairs
        self.max_plan_steps_in_summary = max_plan_steps_in_summary

    async def compress(
        self,
        messages: List[MemoryMessage],
        max_tokens: int,
        token_counter: TokenCounter,
    ) -> CompressionResult:
        if not messages:
            return CompressionResult([], 0, 0, 0, 0, CompressionStrategy.HEURISTIC)

        tokens_before = sum(token_counter.count_message(m) for m in messages)

        # Partition: system | old-compressible | protected recent
        system_msgs: List[MemoryMessage] = []
        non_system: List[MemoryMessage] = []
        for m in messages:
            if m.role == MemoryRole.SYSTEM:
                system_msgs.append(m)
            else:
                non_system.append(m)

        # Protect last N pairs (user+assistant) plus any trailing user msg
        protect_count = self.protected_pairs * 2
        if non_system and non_system[-1].role == MemoryRole.USER:
            protect_count += 1
        protected = non_system[-protect_count:] if len(non_system) > protect_count else non_system
        to_compress = non_system[: len(non_system) - len(protected)]

        if not to_compress:
            return CompressionResult(
                compressed_messages=messages,
                original_count=len(messages),
                compressed_count=len(messages),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                strategy=CompressionStrategy.HEURISTIC,
            )

        # Build structured summary
        summary = self._build_summary(to_compress)
        summary_msg = MemoryMessage(
            role=MemoryRole.COMPRESSED,
            content=summary["text"],
            timestamp=to_compress[0].timestamp,
            image_urls=summary["image_urls"],
            quality_score=summary.get("last_quality_score"),
            is_compressed=True,
            original_count=len(to_compress),
            metadata={"compression_strategy": "heuristic"},
        )
        summary_msg.token_estimate = token_counter.count_message(summary_msg)

        result_msgs = system_msgs + [summary_msg] + protected
        tokens_after = sum(token_counter.count_message(m) for m in result_msgs)

        # If still over budget, further compress protected messages
        # by collapsing plan_json to a minimal form
        if tokens_after > max_tokens:
            for m in result_msgs:
                if m.plan_json and not m.is_compressed:
                    m.plan_json = self._minimise_plan(m.plan_json)
            tokens_after = sum(token_counter.count_message(m) for m in result_msgs)

        return CompressionResult(
            compressed_messages=result_msgs,
            original_count=len(messages),
            compressed_count=len(result_msgs),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            strategy=CompressionStrategy.HEURISTIC,
        )

    # ── Helpers ───────────────────────────────────────────────────

    def _build_summary(self, messages: List[MemoryMessage]) -> Dict[str, Any]:
        """Create a structured text summary of *messages*."""
        lines: List[str] = ["<compressed_memory>"]
        all_image_urls: List[str] = []
        last_quality: Optional[float] = None
        turn_count = 0

        for msg in messages:
            # Collect image URLs
            for url in msg.image_urls:
                if url not in all_image_urls:
                    all_image_urls.append(url)

            if msg.quality_score is not None:
                last_quality = msg.quality_score

            # Summarise each message as one short line
            if msg.role == MemoryRole.USER:
                turn_count += 1
                lines.append(f"[Turn {turn_count}] User: {_truncate(msg.content, 120)}")

            elif msg.role in (MemoryRole.ASSISTANT, MemoryRole.COMPRESSED):
                summary_line = _truncate(msg.content, 100)
                if msg.plan_json:
                    steps = msg.plan_json.get("steps", [])
                    step_summaries = []
                    for s in steps[: self.max_plan_steps_in_summary]:
                        action = s.get("action", "?")
                        thought = _truncate(s.get("thought", ""), 60)
                        step_summaries.append(f"{action}: {thought}")
                    if step_summaries:
                        summary_line += " | Steps: " + "; ".join(step_summaries)
                    if len(steps) > self.max_plan_steps_in_summary:
                        summary_line += f" ... (+{len(steps) - self.max_plan_steps_in_summary} more)"
                lines.append(f"[Turn {turn_count}] Assistant: {summary_line}")

            # Skill result one-liners
            for sr in msg.skill_results:
                skill = sr.get("skill_name", "unknown")
                status = sr.get("status", "?")
                result = sr.get("result", {})
                if isinstance(result, dict):
                    url = result.get("image_url")
                    score = result.get("overall_score")
                    info = ""
                    if url:
                        info += f" url={url}"
                    if score is not None:
                        info += f" score={score}"
                    lines.append(f"  Skill({skill}): {status}{info}")

        # Image reference section
        if all_image_urls:
            lines.append("Images referenced: " + ", ".join(all_image_urls))
        if last_quality is not None:
            lines.append(f"Last quality score: {last_quality}")

        lines.append("</compressed_memory>")

        return {
            "text": "\n".join(lines),
            "image_urls": all_image_urls,
            "last_quality_score": last_quality,
        }

    @staticmethod
    def _minimise_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce a plan_json to essential fields to save tokens."""
        minimised_steps = []
        for step in plan.get("steps", []):
            minimised_steps.append({
                "action": step.get("action"),
                "status": step.get("status"),
            })
        return {
            "steps": minimised_steps,
            "status": plan.get("status"),
            "final_result": plan.get("final_result"),
        }


class ImportanceCompressor(BaseCompressor):
    """
    Importance-based compressor for multimodal agent memory.

    Scores each message by a weighted combination of recency, role,
    content richness, and multimodal importance (image URLs, quality
    scores, skill results).  Low-importance messages are dropped first;
    system messages are always preserved.

    This replaces the former ``ImportancePruner`` from the standalone
    context_pruner module, with added multimodal awareness.
    """

    def __init__(
        self,
        importance_threshold: float = 0.3,
        protected_pairs: int = 2,
    ):
        self.importance_threshold = importance_threshold
        self.protected_pairs = protected_pairs

    def compute_importance(
        self,
        msg: MemoryMessage,
        position: int,
        total: int,
    ) -> float:
        """Compute importance score in [0, 1] for a message.

        Factors:
        - Recency (newer → higher)
        - Role weight (user > system > assistant)
        - Content length (longer → richer context)
        - Multimodal value (image URLs, quality scores, skill results)
        """
        recency = position / total if total > 0 else 1.0
        role_weights = {
            MemoryRole.USER: 1.0,
            MemoryRole.SYSTEM: 0.9,
            MemoryRole.ASSISTANT: 0.8,
            MemoryRole.COMPRESSED: 0.85,
        }
        role_factor = role_weights.get(msg.role, 0.5)
        content_factor = min(1.0, len(msg.content) / 500)

        # Multimodal bonus
        mm_bonus = 0.0
        if msg.image_urls:
            mm_bonus += 0.15
        if msg.quality_score is not None:
            mm_bonus += 0.10
        if msg.skill_results:
            mm_bonus += 0.05
        mm_bonus = min(mm_bonus, 0.3)

        return (
            0.30 * recency
            + 0.25 * role_factor
            + 0.10 * content_factor
            + 0.15 * mm_bonus / 0.3 if mm_bonus else 0.0
        ) + 0.20 * (1.0 if msg.image_urls or msg.quality_score is not None else 0.0)

    async def compress(
        self,
        messages: List[MemoryMessage],
        max_tokens: int,
        token_counter: TokenCounter,
    ) -> CompressionResult:
        if not messages:
            return CompressionResult([], 0, 0, 0, 0, CompressionStrategy.HEURISTIC)

        tokens_before = sum(token_counter.count_message(m) for m in messages)

        # System messages always kept
        system_msgs = [m for m in messages if m.role == MemoryRole.SYSTEM]
        non_system = [m for m in messages if m.role != MemoryRole.SYSTEM]

        # Score every non-system message
        scored = [
            (m, self.compute_importance(m, i, len(non_system)))
            for i, m in enumerate(non_system)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Greedily select within budget
        system_tokens = sum(token_counter.count_message(m) for m in system_msgs)
        budget = max_tokens - system_tokens
        selected: List[MemoryMessage] = []
        used = 0

        for m, score in scored:
            if score < self.importance_threshold:
                continue
            t = token_counter.count_message(m)
            if used + t <= budget:
                selected.append(m)
                used += t

        # Re-sort by original timestamp for coherent ordering
        selected.sort(key=lambda m: m.timestamp)

        result_msgs = system_msgs + selected
        tokens_after = sum(token_counter.count_message(m) for m in result_msgs)

        return CompressionResult(
            compressed_messages=result_msgs,
            original_count=len(messages),
            compressed_count=len(result_msgs),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            strategy=CompressionStrategy.HEURISTIC,
        )


class LLMCompressor(BaseCompressor):
    """
    LLM-powered compressor that delegates summarisation to a model.

    Accepts an async callable ``llm_fn`` with the signature::

        async def llm_fn(prompt: str) -> str
            ...

    The prompt instructs the model to produce a structured summary
    that **must** preserve image URLs and quality scores.
    """

    DEFAULT_PROMPT_TEMPLATE = (
        "You are a memory compression assistant for a multimodal image-generation agent.\n"
        "Summarise the following conversation history into a concise structured format.\n\n"
        "CRITICAL: You MUST preserve ALL of the following verbatim:\n"
        "- Image URLs (http/https links to generated or repaired images)\n"
        "- Quality/evaluation scores\n"
        "- Skill execution outcomes (success/failure)\n"
        "- The user's original intent\n\n"
        "Compress verbose reasoning and intermediate thoughts into brief summaries.\n"
        "Target token budget: {max_tokens} tokens.\n\n"
        "Conversation history:\n{messages_json}\n\n"
        "Output your summary wrapped in <compressed_memory> ... </compressed_memory> tags."
    )

    def __init__(
        self,
        llm_fn: Callable[[str], Awaitable[str]],
        prompt_template: Optional[str] = None,
    ):
        self.llm_fn = llm_fn
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

    async def compress(
        self,
        messages: List[MemoryMessage],
        max_tokens: int,
        token_counter: TokenCounter,
    ) -> CompressionResult:
        if not messages:
            return CompressionResult([], 0, 0, 0, 0, CompressionStrategy.LLM)

        tokens_before = sum(token_counter.count_message(m) for m in messages)

        # Separate system messages (never compressed via LLM)
        system_msgs = [m for m in messages if m.role == MemoryRole.SYSTEM]
        non_system = [m for m in messages if m.role != MemoryRole.SYSTEM]

        if not non_system:
            return CompressionResult(messages, len(messages), len(messages),
                                     tokens_before, tokens_before, CompressionStrategy.LLM)

        # Build prompt
        msgs_json = json.dumps(
            [m.to_dict() for m in non_system],
            ensure_ascii=False,
            indent=2,
        )
        prompt = self.prompt_template.format(
            max_tokens=max_tokens,
            messages_json=msgs_json,
        )

        # Call LLM
        try:
            summary_text = await self.llm_fn(prompt)
        except Exception as exc:
            # Fallback to heuristic on LLM failure
            fallback = HeuristicCompressor()
            return await fallback.compress(messages, max_tokens, token_counter)

        # Collect all image URLs from original messages
        all_image_urls: List[str] = []
        last_quality: Optional[float] = None
        for m in non_system:
            for url in m.image_urls:
                if url not in all_image_urls:
                    all_image_urls.append(url)
            if m.quality_score is not None:
                last_quality = m.quality_score

        summary_msg = MemoryMessage(
            role=MemoryRole.COMPRESSED,
            content=summary_text,
            timestamp=non_system[0].timestamp,
            image_urls=all_image_urls,
            quality_score=last_quality,
            is_compressed=True,
            original_count=len(non_system),
            metadata={"compression_strategy": "llm"},
        )
        summary_msg.token_estimate = token_counter.count_message(summary_msg)

        result_msgs = system_msgs + [summary_msg]
        tokens_after = sum(token_counter.count_message(m) for m in result_msgs)

        return CompressionResult(
            compressed_messages=result_msgs,
            original_count=len(messages),
            compressed_count=len(result_msgs),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            strategy=CompressionStrategy.LLM,
        )


# ──────────────────────────────────────────────────────────────────
# Session Memory — the per-session working memory container
# ──────────────────────────────────────────────────────────────────

@dataclass
class MemoryStats:
    """Statistics about a session's working memory."""
    session_id: int
    message_count: int
    compressed_count: int
    total_tokens: int
    max_tokens: int
    usage_ratio: float
    image_url_count: int
    compression_count: int  # how many times compression has been triggered


class SessionMemory:
    """
    Working memory for a single session.

    Holds an ordered list of ``MemoryMessage`` objects and automatically
    triggers compression when the token count exceeds ``max_tokens``.

    Compression is checked on ``add()`` by default (``compress_on_add=True``).
    It can also be checked lazily on ``get_context()`` (``compress_on_get``).
    """

    def __init__(
        self,
        session_id: int,
        max_tokens: int = 32_000,
        compressor: Optional[BaseCompressor] = None,
        token_counter: Optional[TokenCounter] = None,
        compress_on_add: bool = True,
        compress_on_get: bool = True,
    ):
        self.session_id = session_id
        self.max_tokens = max_tokens
        self._compressor = compressor or HeuristicCompressor()
        self._token_counter = token_counter or TokenCounter()
        self._messages: List[MemoryMessage] = []
        self._compress_on_add = compress_on_add
        self._compress_on_get = compress_on_get
        self._compression_count = 0
        self._lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────

    async def add(
        self,
        msg: Union[MemoryMessage, Sequence[MemoryMessage]],
    ) -> Optional[CompressionResult]:
        """Add one or more messages to working memory.

        Returns a ``CompressionResult`` if compression was triggered,
        otherwise ``None``.
        """
        msgs = [msg] if isinstance(msg, MemoryMessage) else list(msg)
        for m in msgs:
            m = copy.deepcopy(m)
            if m.token_estimate == 0:
                m.token_estimate = self._token_counter.count_message(m)
            self._messages.append(m)

        if self._compress_on_add:
            return await self._check_and_compress()
        return None

    async def get_context(
        self,
        recent_n: Optional[int] = None,
    ) -> List[MemoryMessage]:
        """
        Return working memory, optionally compressing first.

        Args:
            recent_n: If set, return only the last *recent_n* messages
                      (system messages are always included).
        """
        if self._compress_on_get:
            await self._check_and_compress()

        messages = list(self._messages)

        if recent_n is not None and recent_n < len(messages):
            system = [m for m in messages if m.role == MemoryRole.SYSTEM]
            non_system = [m for m in messages if m.role != MemoryRole.SYSTEM]
            messages = system + non_system[-recent_n:]

        return messages

    async def get_context_for_planning(self) -> List[Dict[str, str]]:
        """
        Return context formatted for the planning model.

        Produces the ``[{role, content}]`` list expected by
        ``BasePlanningModel.chat_completion()``.  Compressed memory
        blocks are sent as ``system`` role so the planner can
        distinguish them from live conversation.
        """
        messages = await self.get_context()
        formatted: List[Dict[str, str]] = []

        for m in messages:
            role = m.role.value
            content = m.content

            # Compressed blocks → system role for planner awareness
            if m.is_compressed:
                role = "system"
                # Append image references if not already in content
                if m.image_urls:
                    url_list = ", ".join(m.image_urls)
                    if url_list not in content:
                        content += f"\n[Referenced images: {url_list}]"

            formatted.append({"role": role, "content": content})

        return formatted

    async def clear(self) -> None:
        """Clear all working memory."""
        async with self._lock:
            self._messages.clear()
            self._compression_count = 0

    def stats(self) -> MemoryStats:
        total_tokens = sum(self._token_counter.count_message(m) for m in self._messages)
        compressed_count = sum(1 for m in self._messages if m.is_compressed)
        image_urls: set = set()
        for m in self._messages:
            image_urls.update(m.image_urls)

        return MemoryStats(
            session_id=self.session_id,
            message_count=len(self._messages),
            compressed_count=compressed_count,
            total_tokens=total_tokens,
            max_tokens=self.max_tokens,
            usage_ratio=min(1.0, total_tokens / self.max_tokens) if self.max_tokens > 0 else 0.0,
            image_url_count=len(image_urls),
            compression_count=self._compression_count,
        )

    def state_dict(self) -> Dict[str, Any]:
        """Serialise state for persistence."""
        return {
            "session_id": self.session_id,
            "max_tokens": self.max_tokens,
            "compression_count": self._compression_count,
            "messages": [m.to_dict() for m in self._messages],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from a previous ``state_dict()`` call."""
        self.session_id = state.get("session_id", self.session_id)
        self.max_tokens = state.get("max_tokens", self.max_tokens)
        self._compression_count = state.get("compression_count", 0)
        self._messages = [
            MemoryMessage.from_dict(d) for d in state.get("messages", [])
        ]

    # ── Internal ──────────────────────────────────────────────────

    async def _check_and_compress(self) -> Optional[CompressionResult]:
        """Compress if over token budget.

        Returns a ``CompressionResult`` if compression was triggered,
        otherwise ``None``.
        """
        total = sum(self._token_counter.count_message(m) for m in self._messages)
        if total <= self.max_tokens:
            return None

        async with self._lock:
            # Re-check under lock
            total = sum(self._token_counter.count_message(m) for m in self._messages)
            if total <= self.max_tokens:
                return None

            result = await self._compressor.compress(
                self._messages, self.max_tokens, self._token_counter,
            )
            self._messages = result.compressed_messages
            self._compression_count += 1
            return result


# ──────────────────────────────────────────────────────────────────
# MemoryManager — global manager that owns per-session memories
# ──────────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Global memory manager that maintains ``SessionMemory`` instances.

    Provides a high-level API consumed by the chat endpoints and the
    ReAct planner.
    """

    def __init__(
        self,
        default_max_tokens: int = 32_000,
        compressor: Optional[BaseCompressor] = None,
        token_counter: Optional[TokenCounter] = None,
        compress_on_add: bool = True,
        compress_on_get: bool = True,
    ):
        self.default_max_tokens = default_max_tokens
        self._compressor = compressor or HeuristicCompressor()
        self._token_counter = token_counter or TokenCounter()
        self._compress_on_add = compress_on_add
        self._compress_on_get = compress_on_get
        self._sessions: Dict[int, SessionMemory] = {}

    def get_session_memory(self, session_id: int) -> SessionMemory:
        """Get or create working memory for a session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory(
                session_id=session_id,
                max_tokens=self.default_max_tokens,
                compressor=self._compressor,
                token_counter=self._token_counter,
                compress_on_add=self._compress_on_add,
                compress_on_get=self._compress_on_get,
            )
        return self._sessions[session_id]

    async def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        plan_json: Optional[Dict[str, Any]] = None,
        image_urls: Optional[List[str]] = None,
        quality_score: Optional[float] = None,
    ) -> Optional[CompressionResult]:
        """Convenience: create a MemoryMessage and add to session memory.

        Returns a ``CompressionResult`` if compression was triggered.
        """
        msg = MemoryMessage(
            role=MemoryRole(role),
            content=content,
            image_urls=image_urls or [],
            plan_json=plan_json,
            quality_score=quality_score,
        )
        mem = self.get_session_memory(session_id)
        return await mem.add(msg)

    async def add_db_message(self, msg: Any) -> Optional[CompressionResult]:
        """Add a SQLAlchemy Message instance to working memory.

        Returns a ``CompressionResult`` if compression was triggered.
        """
        mm = MemoryMessage.from_db_message(msg)
        session_id = msg.session_id
        mem = self.get_session_memory(session_id)
        return await mem.add(mm)

    async def get_context_for_planning(
        self,
        session_id: int,
    ) -> List[Dict[str, str]]:
        """Return compressed context suitable for the planner."""
        mem = self.get_session_memory(session_id)
        return await mem.get_context_for_planning()

    async def load_from_db(
        self,
        session_id: int,
        db_messages: Sequence[Any],
    ) -> None:
        """
        Bootstrap working memory from SQLite history.

        Called when a session is loaded for the first time or after
        a server restart.
        """
        mem = self.get_session_memory(session_id)
        await mem.clear()
        memory_msgs = [MemoryMessage.from_db_message(m) for m in db_messages]
        if memory_msgs:
            await mem.add(memory_msgs)

    def get_stats(self, session_id: int) -> MemoryStats:
        mem = self.get_session_memory(session_id)
        return mem.stats()

    def delete_session(self, session_id: int) -> None:
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[int]:
        return list(self._sessions.keys())


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def _extract_urls(text: str) -> List[str]:
    """Extract HTTP(S) URLs from text."""
    return _URL_RE.findall(text)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# ──────────────────────────────────────────────────────────────────
# Global singleton
# ──────────────────────────────────────────────────────────────────

_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the global MemoryManager instance."""
    global _memory_manager
    if _memory_manager is None:
        from app.config import get_settings

        settings = get_settings()
        _memory_manager = MemoryManager(
            default_max_tokens=settings.memory_max_tokens,
            compressor=HeuristicCompressor(
                protected_pairs=settings.memory_protected_pairs,
            ),
            compress_on_add=settings.memory_compress_on_add,
            compress_on_get=settings.memory_compress_on_get,
        )
    return _memory_manager


def reset_memory_manager() -> None:
    """Reset the global instance (for testing)."""
    global _memory_manager
    _memory_manager = None
