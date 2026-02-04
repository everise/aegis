"""
Built-in Governance for safety and compliance.

Implements content moderation, rate limiting, access control,
and audit logging for the agent system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum
from collections import defaultdict
import re
import hashlib


class GovernanceAction(str, Enum):
    """Actions taken by governance system."""
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    RATE_LIMIT = "rate_limit"
    REQUIRE_REVIEW = "require_review"


class ViolationType(str, Enum):
    """Types of policy violations."""
    CONTENT_UNSAFE = "content_unsafe"
    RATE_EXCEEDED = "rate_exceeded"
    UNAUTHORIZED = "unauthorized"
    POLICY_VIOLATION = "policy_violation"
    SUSPICIOUS_PATTERN = "suspicious_pattern"


@dataclass
class GovernanceDecision:
    """Result of a governance check."""
    action: GovernanceAction
    allowed: bool
    reason: str
    violations: List[ViolationType] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AuditLogEntry:
    """Entry in the audit log."""
    entry_id: str
    timestamp: datetime
    user_id: Optional[str]
    action: str
    resource: str
    decision: GovernanceDecision
    request_data: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None


class ContentModerator:
    """
    Content moderation for prompts and outputs.
    
    Checks for unsafe, harmful, or policy-violating content.
    """
    
    # Patterns for unsafe content (simplified - use proper ML models in production)
    UNSAFE_PATTERNS = [
        r'\b(violence|gore|blood)\b',
        r'\b(explicit|nsfw|nude)\b',
        r'\b(hate|racist|discriminat)\b',
        r'\b(illegal|drugs|weapons)\b',
    ]
    
    # Patterns for sensitive topics that may need review
    SENSITIVE_PATTERNS = [
        r'\b(political|election|vote)\b',
        r'\b(religious|faith|worship)\b',
        r'\b(medical|diagnosis|treatment)\b',
    ]
    
    def __init__(
        self,
        unsafe_patterns: Optional[List[str]] = None,
        sensitive_patterns: Optional[List[str]] = None,
        strict_mode: bool = False,
    ):
        self.unsafe_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (unsafe_patterns or self.UNSAFE_PATTERNS)
        ]
        self.sensitive_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (sensitive_patterns or self.SENSITIVE_PATTERNS)
        ]
        self.strict_mode = strict_mode
    
    def check_content(self, content: str) -> GovernanceDecision:
        """
        Check content for policy violations.
        
        Args:
            content: Text content to check
            
        Returns:
            GovernanceDecision with action and reason
        """
        violations = []
        
        # Check for unsafe content
        for pattern in self.unsafe_patterns:
            if pattern.search(content):
                violations.append(ViolationType.CONTENT_UNSAFE)
                break
        
        # Check for sensitive content
        sensitive_matches = []
        for pattern in self.sensitive_patterns:
            if pattern.search(content):
                sensitive_matches.append(pattern.pattern)
        
        # Determine action
        if ViolationType.CONTENT_UNSAFE in violations:
            return GovernanceDecision(
                action=GovernanceAction.BLOCK,
                allowed=False,
                reason="Content contains unsafe or prohibited material",
                violations=violations,
            )
        
        if sensitive_matches and self.strict_mode:
            return GovernanceDecision(
                action=GovernanceAction.REQUIRE_REVIEW,
                allowed=False,
                reason=f"Content contains sensitive topics: {', '.join(sensitive_matches)}",
                violations=[ViolationType.POLICY_VIOLATION],
                metadata={"sensitive_patterns": sensitive_matches},
            )
        
        if sensitive_matches:
            return GovernanceDecision(
                action=GovernanceAction.WARN,
                allowed=True,
                reason=f"Content may contain sensitive topics",
                metadata={"sensitive_patterns": sensitive_matches},
            )
        
        return GovernanceDecision(
            action=GovernanceAction.ALLOW,
            allowed=True,
            reason="Content passed moderation checks",
        )


class RateLimiter:
    """
    Rate limiting for API requests.
    
    Implements token bucket algorithm for flexible rate limiting.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        
        # Token buckets per user
        self._buckets: Dict[str, Dict[str, Any]] = {}
    
    def _get_bucket(self, user_id: str) -> Dict[str, Any]:
        """Get or create token bucket for user."""
        if user_id not in self._buckets:
            self._buckets[user_id] = {
                "tokens": float(self.burst_size),
                "last_update": datetime.utcnow(),
            }
        return self._buckets[user_id]
    
    def _refill_bucket(self, bucket: Dict[str, Any]) -> None:
        """Refill tokens based on time elapsed."""
        now = datetime.utcnow()
        elapsed = (now - bucket["last_update"]).total_seconds()
        
        # Tokens per second
        rate = self.requests_per_minute / 60.0
        new_tokens = elapsed * rate
        
        bucket["tokens"] = min(self.burst_size, bucket["tokens"] + new_tokens)
        bucket["last_update"] = now
    
    def check_rate_limit(self, user_id: str) -> GovernanceDecision:
        """
        Check if request is within rate limits.
        
        Args:
            user_id: User identifier
            
        Returns:
            GovernanceDecision indicating if request is allowed
        """
        bucket = self._get_bucket(user_id)
        self._refill_bucket(bucket)
        
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return GovernanceDecision(
                action=GovernanceAction.ALLOW,
                allowed=True,
                reason="Request within rate limits",
                metadata={
                    "remaining_tokens": bucket["tokens"],
                    "limit": self.requests_per_minute,
                },
            )
        
        # Calculate retry time
        tokens_needed = 1.0 - bucket["tokens"]
        rate = self.requests_per_minute / 60.0
        retry_after = tokens_needed / rate
        
        return GovernanceDecision(
            action=GovernanceAction.RATE_LIMIT,
            allowed=False,
            reason=f"Rate limit exceeded. Retry after {retry_after:.1f} seconds",
            violations=[ViolationType.RATE_EXCEEDED],
            metadata={
                "retry_after_seconds": retry_after,
                "limit": self.requests_per_minute,
            },
        )


class AccessController:
    """
    Access control for resources and actions.
    
    Implements role-based access control (RBAC).
    """
    
    def __init__(self):
        # Role -> permissions mapping
        self._role_permissions: Dict[str, Set[str]] = {
            "admin": {"*"},  # All permissions
            "user": {"generate", "evaluate", "view"},
            "viewer": {"view"},
            "trainer": {"generate", "evaluate", "view", "train"},
        }
        
        # User -> roles mapping
        self._user_roles: Dict[str, Set[str]] = {}
    
    def assign_role(self, user_id: str, role: str) -> None:
        """Assign a role to a user."""
        if user_id not in self._user_roles:
            self._user_roles[user_id] = set()
        self._user_roles[user_id].add(role)
    
    def revoke_role(self, user_id: str, role: str) -> None:
        """Revoke a role from a user."""
        if user_id in self._user_roles:
            self._user_roles[user_id].discard(role)
    
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user."""
        permissions: Set[str] = set()
        
        roles = self._user_roles.get(user_id, {"user"})  # Default to user role
        
        for role in roles:
            role_perms = self._role_permissions.get(role, set())
            if "*" in role_perms:
                return {"*"}  # Admin has all permissions
            permissions.update(role_perms)
        
        return permissions
    
    def check_access(
        self,
        user_id: str,
        permission: str,
    ) -> GovernanceDecision:
        """
        Check if user has permission for an action.
        
        Args:
            user_id: User identifier
            permission: Required permission
            
        Returns:
            GovernanceDecision indicating if access is allowed
        """
        permissions = self.get_user_permissions(user_id)
        
        if "*" in permissions or permission in permissions:
            return GovernanceDecision(
                action=GovernanceAction.ALLOW,
                allowed=True,
                reason=f"User has '{permission}' permission",
                metadata={"permissions": list(permissions)},
            )
        
        return GovernanceDecision(
            action=GovernanceAction.BLOCK,
            allowed=False,
            reason=f"User lacks '{permission}' permission",
            violations=[ViolationType.UNAUTHORIZED],
            metadata={"required": permission, "user_permissions": list(permissions)},
        )


class AuditLogger:
    """
    Audit logging for compliance and monitoring.
    
    Records all governance decisions and significant actions.
    """
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self._entries: List[AuditLogEntry] = []
        self._entry_counter = 0
    
    def _generate_entry_id(self) -> str:
        """Generate unique entry ID."""
        self._entry_counter += 1
        timestamp = datetime.utcnow().isoformat()
        return hashlib.sha256(f"{timestamp}-{self._entry_counter}".encode()).hexdigest()[:16]
    
    def log(
        self,
        action: str,
        resource: str,
        decision: GovernanceDecision,
        user_id: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> AuditLogEntry:
        """
        Log an audit entry.
        
        Args:
            action: Action being performed
            resource: Resource being accessed
            decision: Governance decision
            user_id: User performing the action
            request_data: Additional request context
            ip_address: Client IP address
            
        Returns:
            Created AuditLogEntry
        """
        entry = AuditLogEntry(
            entry_id=self._generate_entry_id(),
            timestamp=datetime.utcnow(),
            user_id=user_id,
            action=action,
            resource=resource,
            decision=decision,
            request_data=request_data or {},
            ip_address=ip_address,
        )
        
        self._entries.append(entry)
        
        # Maintain max entries
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]
        
        return entry
    
    def query(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        violations_only: bool = False,
    ) -> List[AuditLogEntry]:
        """Query audit log entries."""
        results = self._entries
        
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        
        if action:
            results = [e for e in results if e.action == action]
        
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]
        
        if violations_only:
            results = [e for e in results if e.decision.violations]
        
        return results
    
    def get_statistics(
        self,
        period_hours: int = 24,
    ) -> Dict[str, Any]:
        """Get audit statistics for a time period."""
        cutoff = datetime.utcnow() - timedelta(hours=period_hours)
        recent = [e for e in self._entries if e.timestamp >= cutoff]
        
        stats = {
            "total_entries": len(recent),
            "actions": defaultdict(int),
            "decisions": defaultdict(int),
            "violations": defaultdict(int),
            "unique_users": set(),
        }
        
        for entry in recent:
            stats["actions"][entry.action] += 1
            stats["decisions"][entry.decision.action.value] += 1
            for v in entry.decision.violations:
                stats["violations"][v.value] += 1
            if entry.user_id:
                stats["unique_users"].add(entry.user_id)
        
        stats["unique_users"] = len(stats["unique_users"])
        stats["actions"] = dict(stats["actions"])
        stats["decisions"] = dict(stats["decisions"])
        stats["violations"] = dict(stats["violations"])
        
        return stats


class GovernanceManager:
    """
    Main governance manager that coordinates all governance components.
    """
    
    def __init__(
        self,
        content_moderator: Optional[ContentModerator] = None,
        rate_limiter: Optional[RateLimiter] = None,
        access_controller: Optional[AccessController] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.content_moderator = content_moderator or ContentModerator()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.access_controller = access_controller or AccessController()
        self.audit_logger = audit_logger or AuditLogger()
    
    def check_request(
        self,
        user_id: str,
        action: str,
        content: Optional[str] = None,
        resource: str = "default",
        ip_address: Optional[str] = None,
    ) -> GovernanceDecision:
        """
        Perform comprehensive governance check.
        
        Checks in order:
        1. Access control
        2. Rate limiting
        3. Content moderation (if content provided)
        
        Args:
            user_id: User identifier
            action: Action being performed
            content: Optional content to moderate
            resource: Resource being accessed
            ip_address: Client IP
            
        Returns:
            Final GovernanceDecision
        """
        # Check access control
        access_decision = self.access_controller.check_access(user_id, action)
        if not access_decision.allowed:
            self._log_decision(action, resource, access_decision, user_id, ip_address)
            return access_decision
        
        # Check rate limit
        rate_decision = self.rate_limiter.check_rate_limit(user_id)
        if not rate_decision.allowed:
            self._log_decision(action, resource, rate_decision, user_id, ip_address)
            return rate_decision
        
        # Check content moderation
        if content:
            content_decision = self.content_moderator.check_content(content)
            if not content_decision.allowed:
                self._log_decision(action, resource, content_decision, user_id, ip_address)
                return content_decision
            
            # Return content decision if it has warnings
            if content_decision.action == GovernanceAction.WARN:
                self._log_decision(action, resource, content_decision, user_id, ip_address)
                return content_decision
        
        # All checks passed
        final_decision = GovernanceDecision(
            action=GovernanceAction.ALLOW,
            allowed=True,
            reason="All governance checks passed",
        )
        
        self._log_decision(action, resource, final_decision, user_id, ip_address)
        return final_decision
    
    def _log_decision(
        self,
        action: str,
        resource: str,
        decision: GovernanceDecision,
        user_id: str,
        ip_address: Optional[str],
    ) -> None:
        """Log governance decision."""
        self.audit_logger.log(
            action=action,
            resource=resource,
            decision=decision,
            user_id=user_id,
            ip_address=ip_address,
        )
    
    def get_audit_statistics(self, period_hours: int = 24) -> Dict[str, Any]:
        """Get audit statistics."""
        return self.audit_logger.get_statistics(period_hours)
