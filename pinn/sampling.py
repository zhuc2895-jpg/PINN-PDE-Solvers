"""Collocation-point sampling shared by all problems.

The original scripts each re-implemented "n points on the four edges of the unit
square" or "n points on the two ends of the strip" with slightly different
rounding, so the effective point counts differed between scripts for the same
``N_BOUNDARY``.  There is one definition of each layout now.
"""

from __future__ import annotations

from typing import Dict

import torch

__all__ = ["uniform", "unit_square_boundary", "strip_boundary"]


def uniform(n: int, lo: float, hi: float, device: torch.device) -> torch.Tensor:
    """``n`` points in ``[lo, hi)`` as a ``(n, 1)`` tensor that carries grad."""
    return (torch.rand(n, 1, device=device) * (hi - lo) + lo).requires_grad_(True)


def unit_square_boundary(
    n: int, lo: float, hi: float, device: torch.device
) -> Dict[str, torch.Tensor]:
    """Split ``n`` points over the four edges of ``[lo, hi]^2``.

    Same convention as the original scripts (the remainder is spread over the
    first edges, so exactly ``n`` points come back).
    """
    edges = []
    for edge in range(4):
        count = n // 4 + (1 if edge < n % 4 else 0)
        t = uniform(count, lo, hi, device)
        if edge == 0:    # y = lo
            edges.append({"x": t, "y": torch.full_like(t, lo)})
        elif edge == 1:  # y = hi
            edges.append({"x": t, "y": torch.full_like(t, hi)})
        elif edge == 2:  # x = lo
            edges.append({"x": torch.full_like(t, lo), "y": t})
        else:            # x = hi
            edges.append({"x": torch.full_like(t, hi), "y": t})
    return {k: torch.cat([e[k] for e in edges], dim=0) for k in ("x", "y")}


def strip_boundary(
    n: int, x_lo: float, x_hi: float, t_lo: float, t_hi: float, device: torch.device
) -> Dict[str, torch.Tensor]:
    """``n`` points on the two ends of a space-time strip (x = x_lo, x = x_hi)."""
    half = n // 2
    t_left = uniform(half, t_lo, t_hi, device)
    t_right = uniform(n - half, t_lo, t_hi, device)
    return {
        "x": torch.cat([torch.full_like(t_left, x_lo), torch.full_like(t_right, x_hi)], dim=0),
        "t": torch.cat([t_left, t_right], dim=0),
    }
