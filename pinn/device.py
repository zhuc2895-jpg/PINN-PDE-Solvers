"""Device selection, written with the silent-GPU-hang in mind.

``torch.cuda.is_available()`` answers "is a CUDA device present and was this
torch built with CUDA?" -- it does not answer "can this build execute a kernel
on this device?".  A build compiled for an older compute capability reports
True on a newer card and then either raises or, worse, hangs on the first
kernel, which is exactly what happened in this repository.

So the default is CPU.  ``"auto"`` escalates to CUDA only after a real matmul
comes back, and ``--device cuda`` remains available for a machine whose torch
matches its card.
"""

from __future__ import annotations

import torch

__all__ = ["resolve_device", "DEVICE_CHOICES"]

DEVICE_CHOICES = ("cpu", "cuda", "auto")


def resolve_device(preferred: str = "cpu") -> torch.device:
    """Turn a ``--device`` string into a ``torch.device``, safely.

    Note that a *hanging* kernel cannot be caught by ``try``/``except``; the
    real protection is that the default is ``"cpu"`` and CUDA is opt-in.
    """
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested, but torch.cuda.is_available() is False"
            )
        return torch.device("cuda")
    if preferred == "auto":
        if torch.cuda.is_available() and _cuda_runs_a_kernel():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"unknown device {preferred!r}; pick one of {DEVICE_CHOICES}")


def _cuda_runs_a_kernel() -> bool:
    """Best-effort probe. Catches raised errors, not hangs."""
    try:
        a = torch.randn(32, 32, device="cuda")
        torch.cuda.synchronize()
        _ = a @ a
        torch.cuda.synchronize()
        return True
    except Exception:
        return False
