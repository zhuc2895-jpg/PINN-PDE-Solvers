"""CLI for frozen-run evaluation of the periodic-hill protocol."""

from __future__ import annotations

from .periodic_hill_evaluation import main

__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
