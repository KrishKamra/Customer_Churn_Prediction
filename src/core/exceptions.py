"""Domain exception hierarchy.

All errors raised intentionally by ChurnAI derive from :class:`ChurnAIError`, so callers
(CLI, UI, tests) can catch the whole family without swallowing unrelated bugs.
"""

from __future__ import annotations


class ChurnAIError(Exception):
    """Base class for every ChurnAI domain error."""


class DataValidationError(ChurnAIError):
    """Input data violates the expected schema (missing columns, bad types, bad levels)."""

    def __init__(self, message: str, *, missing_columns: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing_columns = missing_columns or []


class ModelNotFoundError(ChurnAIError):
    """The serialized production pipeline could not be located on disk."""


class ModelValidationError(ChurnAIError):
    """A trained model failed the quality gate thresholds."""


class DatabaseError(ChurnAIError):
    """A persistence-layer operation failed (connection dropout, integrity error, ...)."""
