"""CLI for the frozen periodic-hill observation and leakage protocol."""

from __future__ import annotations

from .periodic_hill_observations import main

__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
