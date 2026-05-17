"""Compatibility entrypoint for deployments importing `backend.app:app`.

The canonical FastAPI application now lives in `src.app`.
Keeping this module as a thin re-export prevents divergence and
syntax/merge issues between two separate app implementations.
"""

from src.app import app
