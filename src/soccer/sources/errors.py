"""Shared source errors.

`SourceUnavailableError` is deliberately the only failure type adapters raise for a
dead feed. The invariant it protects: a source that fails must never return an
empty-but-successful result, because downstream an empty fixture list and a broken
API are indistinguishable. Callers either get data, get flagged stale data, or get
this exception.
"""

from __future__ import annotations


class SourceUnavailableError(RuntimeError):
    """A source failed and no cached fallback was available."""
