"""
Context Pruning for reducing token costs.

Implements strategies for pruning conversation context
to fit within token limits while preserving important information.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import re


class PruningStrategy(str, Enum):
    """Available pruning strategies."""
    TRUNCATE = "truncate"          # Simple truncation from start
    SLIDING_WINDOW = "sliding_window"  # Keep recent context
    IMPORTANCE = "importance"       # Keep important messages
    SUMMARY = "summary"            # Summarize older context
    HYBRID = "hybrid"              # Combination of strategies


@dataclass
class Message:
    """Represents a message in the conversation."""
    role: str  # user, assistant, system
    content: str
    token_count: int = 0
    importance: float = 1.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PruningResult:
    """Result of context pruning."""
    messages: List[Message]
    total_tokens: int
    pruned_count: int
    strategy_used: PruningStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenCounter:
    """
    Estimates token count for text.
    
    Uses a simple heuristic based on word/character count.
    In production, use tiktoken or model-specific tokenizer.
    """
    
    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token
    
    def count(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple heuristic: ~4 chars per token for English
        # More accurate for CJK: ~1-2 chars per token
        return max(1, int(len(text) / self.chars_per_token))
    
    def count_message(self, message: Message) -> int:
        """Count tokens in a message including role overhead."""
        # Add overhead for role and formatting
        overhead = 4  # Approximate overhead per message
        content_tokens = self.count(message.content)
        return content_tokens + overhead


class BasePruner(ABC):
    """Abstract base class for context pruners."""
    
    def __init__(
        self,
        max_tokens: int = 4000,
        token_counter: Optional[TokenCounter] = None,
    ):
        self.max_tokens = max_tokens
        self.token_counter = token_counter or TokenCounter()
    
    @abstractmethod
    def prune(self, messages: List[Message]) -> PruningResult:
        """Prune messages to fit within token limit."""
        pass
    
    def _count_total_tokens(self, messages: List[Message]) -> int:
        """Count total tokens in message list."""
        return sum(self.token_counter.count_message(m) for m in messages)


class TruncationPruner(BasePruner):
    """
    Simple truncation pruner.
    
    Removes oldest messages until within token limit.
    Always preserves system messages.
    """
    
    def prune(self, messages: List[Message]) -> PruningResult:
        """Prune by truncating from the start."""
        if not messages:
            return PruningResult(
                messages=[],
                total_tokens=0,
                pruned_count=0,
                strategy_used=PruningStrategy.TRUNCATE,
            )
        
        # Separate system messages (always keep)
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]
        
        system_tokens = self._count_total_tokens(system_msgs)
        available_tokens = self.max_tokens - system_tokens
        
        # Keep messages from end until limit reached
        kept_msgs = []
        current_tokens = 0
        
        for msg in reversed(other_msgs):
            msg_tokens = self.token_counter.count_message(msg)
            if current_tokens + msg_tokens <= available_tokens:
                kept_msgs.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        result_msgs = system_msgs + kept_msgs
        pruned_count = len(messages) - len(result_msgs)
        
        return PruningResult(
            messages=result_msgs,
            total_tokens=current_tokens + system_tokens,
            pruned_count=pruned_count,
            strategy_used=PruningStrategy.TRUNCATE,
        )


class SlidingWindowPruner(BasePruner):
    """
    Sliding window pruner.
    
    Keeps a fixed window of recent messages plus system context.
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        window_size: int = 10,
        token_counter: Optional[TokenCounter] = None,
    ):
        super().__init__(max_tokens, token_counter)
        self.window_size = window_size
    
    def prune(self, messages: List[Message]) -> PruningResult:
        """Prune using sliding window."""
        if not messages:
            return PruningResult(
                messages=[],
                total_tokens=0,
                pruned_count=0,
                strategy_used=PruningStrategy.SLIDING_WINDOW,
            )
        
        # Keep system messages
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]
        
        # Take last window_size messages
        windowed = other_msgs[-self.window_size:] if len(other_msgs) > self.window_size else other_msgs
        
        # Further prune if still over token limit
        result_msgs = system_msgs + windowed
        total_tokens = self._count_total_tokens(result_msgs)
        
        while total_tokens > self.max_tokens and windowed:
            windowed.pop(0)
            result_msgs = system_msgs + windowed
            total_tokens = self._count_total_tokens(result_msgs)
        
        return PruningResult(
            messages=result_msgs,
            total_tokens=total_tokens,
            pruned_count=len(messages) - len(result_msgs),
            strategy_used=PruningStrategy.SLIDING_WINDOW,
            metadata={"window_size": self.window_size},
        )


class ImportancePruner(BasePruner):
    """
    Importance-based pruner.
    
    Keeps messages with highest importance scores.
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        token_counter: Optional[TokenCounter] = None,
        importance_threshold: float = 0.3,
    ):
        super().__init__(max_tokens, token_counter)
        self.importance_threshold = importance_threshold
    
    def compute_importance(self, message: Message, position: int, total: int) -> float:
        """
        Compute importance score for a message.
        
        Factors:
        - Recency (newer = more important)
        - Role (user messages often more important)
        - Content length (longer may be more detailed)
        - Explicit importance score
        """
        # Recency factor (0.0 to 1.0)
        recency = position / total if total > 0 else 1.0
        
        # Role factor
        role_weights = {"user": 1.0, "assistant": 0.8, "system": 0.9}
        role_factor = role_weights.get(message.role, 0.5)
        
        # Content factor (normalize by length)
        content_factor = min(1.0, len(message.content) / 500)
        
        # Combine factors
        computed = (
            0.4 * recency +
            0.3 * role_factor +
            0.1 * content_factor +
            0.2 * message.importance
        )
        
        return computed
    
    def prune(self, messages: List[Message]) -> PruningResult:
        """Prune by importance score."""
        if not messages:
            return PruningResult(
                messages=[],
                total_tokens=0,
                pruned_count=0,
                strategy_used=PruningStrategy.IMPORTANCE,
            )
        
        # Compute importance for all messages
        scored_msgs = []
        for i, msg in enumerate(messages):
            importance = self.compute_importance(msg, i, len(messages))
            scored_msgs.append((msg, importance))
        
        # Sort by importance (keep system messages at top)
        system_msgs = [(m, s) for m, s in scored_msgs if m.role == "system"]
        other_msgs = [(m, s) for m, s in scored_msgs if m.role != "system"]
        other_msgs.sort(key=lambda x: x[1], reverse=True)
        
        # Select messages within token budget
        selected = [m for m, _ in system_msgs]
        current_tokens = self._count_total_tokens(selected)
        
        for msg, score in other_msgs:
            if score < self.importance_threshold:
                continue
            
            msg_tokens = self.token_counter.count_message(msg)
            if current_tokens + msg_tokens <= self.max_tokens:
                selected.append(msg)
                current_tokens += msg_tokens
        
        # Re-sort by original order for coherence
        original_order = {id(m): i for i, m in enumerate(messages)}
        selected.sort(key=lambda m: original_order.get(id(m), 0))
        
        return PruningResult(
            messages=selected,
            total_tokens=current_tokens,
            pruned_count=len(messages) - len(selected),
            strategy_used=PruningStrategy.IMPORTANCE,
        )


class ContextPruner:
    """
    Main context pruner with configurable strategy.
    """
    
    PRUNERS = {
        PruningStrategy.TRUNCATE: TruncationPruner,
        PruningStrategy.SLIDING_WINDOW: SlidingWindowPruner,
        PruningStrategy.IMPORTANCE: ImportancePruner,
    }
    
    def __init__(
        self,
        strategy: PruningStrategy = PruningStrategy.SLIDING_WINDOW,
        max_tokens: int = 4000,
        **kwargs,
    ):
        self.strategy = strategy
        self.max_tokens = max_tokens
        
        pruner_class = self.PRUNERS.get(strategy, SlidingWindowPruner)
        self.pruner = pruner_class(max_tokens=max_tokens, **kwargs)
    
    def prune(self, messages: List[Message]) -> PruningResult:
        """Prune messages using configured strategy."""
        return self.pruner.prune(messages)
    
    def prune_conversation(
        self,
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Convenience method to prune conversation dicts.
        
        Args:
            messages: List of {"role": ..., "content": ...} dicts
            
        Returns:
            Pruned list of message dicts
        """
        # Convert to Message objects
        msg_objects = [
            Message(
                role=m.get("role", "user"),
                content=m.get("content", ""),
            )
            for m in messages
        ]
        
        # Prune
        result = self.prune(msg_objects)
        
        # Convert back to dicts
        return [
            {"role": m.role, "content": m.content}
            for m in result.messages
        ]
