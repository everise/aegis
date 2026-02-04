"""
API module for Aegis.

Contains all FastAPI routers for the application.
"""

from app.api import sessions, messages, skills, training

__all__ = ["sessions", "messages", "skills", "training"]
