"""Two questions, answered with numbers instead of opinions.

The prototype these numbers came from has been promoted to the ``pinn`` package;
this script drives that package now, so re-running it re-measures the real code.

Part 1 -- does one seam really cover all four PDEs?  Each problem runs a short
training through the identical generic loop; what differs is only the PDE
object handed to it.

Part 2 -- which network architecture is actually better here?  Same problem,
same collocation points, same seed, same optimiser schedule; only the network
changes.  Poisson is used because it is the only one of the four with an
analytic solution, so the L2 column is real accuracy, not a proxy.

Run:  python explore/architecture_sweep.py --device cpu
      python explore/architecture_sweep.py --epochs 400 --device cpu   (quick)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# the promoted package lives at the repo root; this file stays a thin runner
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinn import MLP, TrainConfig, inputs_from, train  # noqa: E402
from pinn.problems import (  # noqa: E402
    Burgers1D,
    ConvectionDiffusion2D,
    LidDrivenCavity,
    Poisson2D,
)

NETWORKS = [
    ("mlp-tanh (current)", dict(activation="tanh", width=64, depth=4, n_fourier=0)),
    ("mlp-tanh+fourier", dict(activation="tanh", width=64, depth=4, n_fourier=16, fourier_sigma=1.5)),
    ("mlp-sin (SIREN init)", dict(activation="sin", width=64, depth=4, n_fourier=0)),
    ("mlp-silu", dict(activation="silu", width=64, depth=4, n_fourier=0)),
]

SMOKE = [
    ("2d-poisson", Poisson2D, dict(width=64, depth=4),
     dict(n_interior=512, n_boundary=128, n_initial=0), 300),
    ("2d-convection-diffusion", ConvectionDiffusion2D, dict(width=64, depth=4),
     dict(n_interior=512, n_boundary=128, n_initial=0), 300),
    ("1d-burgers", Burgers1D, dict(width=32, depth=4),
     dict(n_interior=2000, n_boundary=200, n_initial=200), 300),
    ("2d-lid-driven-cavity", LidDrivenCavity, dict(width=64, depth=4),
     dict(n_interior=800, n_boundary=400, n_initial=0), 300),
]


def relative_l2(problem: Poisson2D, model: torch.nn.Module, n: int = 128) -> float:
    axis = torch.linspace(0.0, 1.0, n)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    coords = {"x": x.reshape(-1, 1), "y": y.reshape(-1, 1)}
    with torch.no_grad():
        pred = model(inputs_from(coords, problem.coords))[:, 0]
    true = problem.exact(coords)["u"].reshape(-1)
    return float(torch.norm(pred - true) / torch.norm(true))


def sweep_networks(epochs: int, device: str) -> list[tuple[str, float, float, float]]:
    rows = []
    for label, kwargs in NETWORKS:
        problem = Poisson2D()
        model = MLP(len(problem.coords), problem.out_dim, seed=0, **kwargs)
        config = TrainConfig(
            epochs=epochs, device=device, verbose=False,
            n_interior=512, n_boundary=128,
            lr_step=1500,  # mirrors pinn_2d_poisson.py, whose StepLR never fires
        )
        start = time.perf_counter()
        history = train(problem, model, config)
        elapsed = time.perf_counter() - start
        err = relative_l2(problem, model)
        rows.append((label, err, history["total"][-1], elapsed))
        print(f"  {label:22s}  L2={err:9.3e}  final_loss={history['total'][-1]:9.3e}  ({elapsed:5.1f}s)")
    return rows


def smoke_all(device: str) -> bool:
    ok = True
    for name, cls, net_kwargs, data_kwargs, epochs in SMOKE:
        problem = cls()
        model = MLP(len(problem.coords), problem.out_dim, seed=0, **net_kwargs)
        config = TrainConfig(epochs=epochs, device=device, verbose=False, **data_kwargs)
        start = time.perf_counter()
        try:
            history = train(problem, model, config)
        except Exception as exc:  # noqa: BLE001 - this is a smoke test
            print(f"  {name:24s}  FAILED  {type(exc).__name__}: {exc}")
            ok = False
            continue
        first, last = history["total"][0], history["total"][-1]
        terms = sorted(k for k in history if k != "total")
        print(f"  {name:24s}  loss {first:.2e} -> {last:.2e} ({last / first:5.3f}x)  "
              f"terms={terms}  ({time.perf_counter() - start:4.1f}s)")
        ok = ok and last < first
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1500,
                        help="1500 mirrors the original Poisson script")
    parser.add_argument("--device", default="cpu",
                        help="cpu by default: a torch build that cannot run on this GPU "
                             "still reports cuda.is_available() == True and then hangs")
    args = parser.parse_args()

    print(f"device={args.device}\n")
    print("[1] all four PDEs through the same sample/losses/weights seam (300 epochs each)")
    ok = smoke_all(args.device)

    print(f"\n[2] 2-D Poisson network sweep, {args.epochs} epochs, relative L2 on a 128x128 grid")
    sweep_networks(args.epochs, args.device)

    print("\nreferring to the committed scripts: pinn_2d_poisson.py reaches 0.110987 (11.10%)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

