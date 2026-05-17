"""Compatibility entrypoint for Render/uvicorn startup.

Primary target should be `src.app:app`.
This module re-exports from `src.app` directly to avoid an extra import hop.
"""

from src.app import app

__all__ = ["app"]
