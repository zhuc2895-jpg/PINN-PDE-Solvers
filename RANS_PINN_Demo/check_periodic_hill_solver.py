"""CLI for the six-gate periodic-hill solver acceptance."""

from __future__ import annotations

from .periodic_hill_acceptance import main

__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
