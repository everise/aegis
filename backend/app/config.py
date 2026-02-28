"""
Configuration management for Aegis.

Settings are loaded from the project-root ``aegis.yaml`` file.
The resolution order for the config file path is:

1. Explicit *config_path* argument passed to ``Settings()``.
2. ``AEGIS_CONFIG`` environment variable.
3. ``<project_root>/aegis.yaml``  (auto-detected via this file's location).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ── helpers ──────────────────────────────────────────────────────
def _find_config_path() -> Path:
    """Return the default ``aegis.yaml`` path (project root)."""
    # backend/app/config.py  →  project root is two levels up from backend/
    return Path(__file__).resolve().parent.parent.parent / "aegis.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return a dict (empty dict if file missing)."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _flatten(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the nested YAML structure into the flat field names used by
    the ``Settings`` model.

    Mapping (YAML path → Settings field):
        server.host              → host
        server.port              → port
        server.debug             → debug
        database.url             → database_url
        remote.api_base_url      → remote_api_base_url
        skills.dir               → skills_dir
        training.max_trajectory_steps → max_trajectory_steps
        training.replay_buffer_size   → replay_buffer_size
        training.batch_size           → training_batch_size
        training.discount_factor      → discount_factor
        sse.ping_interval        → sse_ping_interval
        memory.max_tokens        → memory_max_tokens
        memory.compress_on_add   → memory_compress_on_add
        memory.compress_on_get   → memory_compress_on_get
        memory.protected_pairs   → memory_protected_pairs
        vector.chroma_persist_dir    → chroma_persist_dir
        vector.max_context_tokens    → max_context_tokens
    """
    flat: Dict[str, Any] = {}

    mapping = {
        ("server", "host"): "host",
        ("server", "port"): "port",
        ("server", "debug"): "debug",
        ("database", "url"): "database_url",
        ("remote", "api_base_url"): "remote_api_base_url",
        ("skills", "dir"): "skills_dir",
        ("training", "max_trajectory_steps"): "max_trajectory_steps",
        ("training", "replay_buffer_size"): "replay_buffer_size",
        ("training", "batch_size"): "training_batch_size",
        ("training", "discount_factor"): "discount_factor",
        ("sse", "ping_interval"): "sse_ping_interval",
        ("memory", "max_tokens"): "memory_max_tokens",
        ("memory", "compress_on_add"): "memory_compress_on_add",
        ("memory", "compress_on_get"): "memory_compress_on_get",
        ("memory", "protected_pairs"): "memory_protected_pairs",
        ("vector", "chroma_persist_dir"): "chroma_persist_dir",
        ("vector", "max_context_tokens"): "max_context_tokens",
        ("retrieval", "coarse_k"): "retrieval_coarse_k",
        ("retrieval", "fine_k"): "retrieval_fine_k",
        ("retrieval", "rrf_k"): "retrieval_rrf_k",
        ("retrieval", "top_k"): "retrieval_top_k",
        ("retrieval", "enabled"): "retrieval_enabled",
    }

    for (section, key), field_name in mapping.items():
        section_data = data.get(section)
        if isinstance(section_data, dict) and key in section_data:
            flat[field_name] = section_data[key]

    return flat


# ── Settings model ───────────────────────────────────────────────
class Settings(BaseModel):
    """Application settings loaded from ``aegis.yaml``."""

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/aegis.db",
        description="Database connection URL",
    )

    # Remote service
    remote_api_base_url: str = Field(
        default="http://localhost:8000/api/v1/tasks",
        description="Base URL for remote skill services",
    )

    # Skills
    skills_dir: str = Field(
        default="skills",
        description="Directory containing skill definitions",
    )

    # RL Training
    max_trajectory_steps: int = Field(default=10, description="Maximum steps per trajectory")
    replay_buffer_size: int = Field(default=10000, description="Replay buffer capacity")
    training_batch_size: int = Field(default=32, description="Training batch size")
    discount_factor: float = Field(default=0.99, description="Reward discount factor")

    # SSE
    sse_ping_interval: int = Field(default=30, description="SSE ping interval in seconds")

    # Memory compression
    memory_max_tokens: int = Field(
        default=4000,
        description="Max tokens per session before compression triggers",
    )
    memory_compress_on_add: bool = Field(
        default=True,
        description="Check and compress memory when adding messages",
    )
    memory_compress_on_get: bool = Field(
        default=True,
        description="Check and compress memory when retrieving context",
    )
    memory_protected_pairs: int = Field(
        default=2,
        description="Number of recent user-assistant pairs protected from compression",
    )

    # Vector database
    chroma_persist_dir: str = Field(
        default="data/chroma_db",
        description="ChromaDB persistent storage directory",
    )
    max_context_tokens: int = Field(
        default=128000,
        description="Maximum context window token count",
    )

    # Dual-Level Retrieval
    retrieval_coarse_k: int = Field(
        default=50,
        description="BM25 coarse retrieval: number of candidates",
    )
    retrieval_fine_k: int = Field(
        default=50,
        description="Semantic fine retrieval: number of candidates",
    )
    retrieval_rrf_k: int = Field(
        default=60,
        description="RRF smoothing constant (higher = more weight to lower ranks)",
    )
    retrieval_top_k: int = Field(
        default=3,
        description="Number of final retrieval results to inject as context",
    )
    retrieval_enabled: bool = Field(
        default=True,
        description="Enable dual-level retrieval augmentation in the ReAct planner",
    )

    # ── validators ───────────────────────────────────────────────
    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {v}")
        return v

    @field_validator("discount_factor")
    @classmethod
    def validate_discount_factor(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Discount factor must be between 0 and 1, got {v}")
        return v


# ── Global singleton ─────────────────────────────────────────────
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Get the global settings instance.

    Resolution order for the config file:
    1. *config_path* argument (if provided).
    2. ``AEGIS_CONFIG`` environment variable.
    3. Auto-detected ``aegis.yaml`` in the project root.

    The YAML values are merged with Pydantic defaults — any key absent
    from the file simply falls back to its default.
    """
    global _settings
    if _settings is None:
        path_str = config_path or os.environ.get("AEGIS_CONFIG")
        path = Path(path_str) if path_str else _find_config_path()
        raw = _load_yaml(path)
        flat = _flatten(raw)
        _settings = Settings(**flat)
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
