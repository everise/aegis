"""Configuration management for Aegis."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Database settings
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/aegis.db",
        description="Database connection URL"
    )
    
    # Remote service settings (mock service URL)
    remote_api_base_url: str = Field(
        default="http://localhost:8000/api/v1/tasks",
        description="Base URL for remote skill services"
    )
    
    # Skill settings
    skills_dir: str = Field(
        default="app/skills",
        description="Directory containing skill definitions"
    )
    
    # RL Training settings
    max_trajectory_steps: int = Field(default=10, description="Maximum steps per trajectory")
    replay_buffer_size: int = Field(default=10000, description="Replay buffer capacity")
    training_batch_size: int = Field(default=32, description="Training batch size")
    discount_factor: float = Field(default=0.99, description="Reward discount factor")
    
    # SSE settings
    sse_ping_interval: int = Field(default=30, description="SSE ping interval in seconds")
    
    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {v}")
        return v
    
    @field_validator("discount_factor")
    @classmethod
    def validate_discount_factor(cls, v: float) -> float:
        """Validate discount factor is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Discount factor must be between 0 and 1, got {v}")
        return v
    
    class Config:
        env_prefix = "AEGIS_"
        env_file = ".env"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
