"""
FastAPI application entry point for Aegis.

Provides the main application factory and startup/shutdown events.
Serves both API and frontend static files from the same port.
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.database import init_db, close_db

# Frontend build directory (relative to project root)
FRONTEND_DIST_DIR = Path(__file__).parent.parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for database initialization
    and cleanup.
    """
    # Startup
    await init_db()
    yield
    # Shutdown
    await close_db()


def create_app() -> FastAPI:
    """
    Application factory for creating the FastAPI instance.
    
    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    settings = get_settings()
    
    app = FastAPI(
        title="Aegis",
        description="AI Agent Framework for Image Generation with Multi-Turn RL Training",
        version="0.1.0",
        debug=settings.debug,
        lifespan=lifespan,
    )
    
    # CORS middleware for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routes
    register_routes(app)
    
    # Register exception handlers
    register_exception_handlers(app)
    
    # Register static file serving (must be last to catch-all routes)
    register_static_files(app)
    
    return app


def register_routes(app: FastAPI) -> None:
    """Register all API routes."""
    from app.api import sessions, messages, skills, training
    from app.api import planning_models
    from app.api import prompts
    from app.api import context
    from app.services.mock_remote import router as mock_router
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": "0.1.0"}
    
    # API routes
    app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["sessions"])
    app.include_router(messages.router, prefix="/api/v1", tags=["messages"])
    app.include_router(skills.router, prefix="/api/v1/skills", tags=["skills"])
    app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
    app.include_router(planning_models.router, prefix="/api/v1/planning-models", tags=["planning-models"])
    app.include_router(prompts.router, prefix="/api/v1/prompts", tags=["prompts"])
    app.include_router(context.router, prefix="/api/v1/context", tags=["context"])
    
    # Mock remote API for development/testing
    app.include_router(mock_router, prefix="/api/v1/tasks", tags=["mock"])


def register_static_files(app: FastAPI) -> None:
    """Register static file serving for frontend."""
    if not FRONTEND_DIST_DIR.exists():
        return  # Frontend not built yet
    
    # Mount static assets directory
    assets_dir = FRONTEND_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    
    # Serve index.html for SPA routes (catch-all)
    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        """Serve frontend SPA for non-API routes."""
        # Skip API routes
        if full_path.startswith("api/") or full_path == "health" or full_path == "docs" or full_path == "openapi.json":
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        
        # Try to serve static file first
        file_path = FRONTEND_DIST_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        
        # Serve index.html for SPA routing
        index_path = FRONTEND_DIST_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        
        return JSONResponse(status_code=404, content={"detail": "Frontend not built"})


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)},
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        settings = get_settings()
        if settings.debug:
            return JSONResponse(
                status_code=500,
                content={"detail": str(exc), "type": type(exc).__name__},
            )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
