"""2-D Poisson PINN:  -Laplacian(u) = f on [0,1]^2,  u = 0 on the boundary.

    u_exact = sin(pi x) sin(pi y)

A thin entry point: the mathematics is ``pinn.problems.Poisson2D`` and the
training loop is ``pinn.trainer``.  Run with ``python pinn_2d_poisson.py``.

Hyperparameters match the committed script, including ``lr_step=1500`` (the
StepLR fires only after the final epoch, so the learning rate never actually
decays).  That is preserved deliberately: this refactor is meant to change the
structure, not the numbers.  See docs/architecture-exploration.md section 2.2.
"""

from __future__ import annotations

import argparse

import torch

from pinn import MLP, TrainConfig, resolve_device, train
from pinn.evaluate import predict_field, relative_l2, unit_square_grid
from pinn.plotting import plot_loss_history, plt, save_figure
from pinn.problems import Poisson2D

OUTPUT = "pinn_poisson_result.png"
N_INTERIOR = 512
N_BOUNDARY = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"),
                        help="cpu by default: a mismatched CUDA build hangs, see pinn/device.py")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=OUTPUT)
    return parser.parse_args()


def evaluate(problem, model, device):
    n = 128
    X, Y, coords = unit_square_grid(n, device=device)
    u_pred = predict_field(model, problem, coords).reshape(n, n)
    u_exact = problem.exact(coords)["u"].reshape(n, n).cpu()
    return X, Y, u_pred, u_exact


def plot(history, X, Y, u_pred, u_exact, error, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    plot_loss_history(axes[0, 0], history)

    ax = axes[0, 1]
    im = ax.pcolormesh(X.numpy(), Y.numpy(), u_pred.numpy(), shading="auto", cmap="viridis")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("PINN Prediction u(x, y)")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    im = ax.pcolormesh(X.numpy(), Y.numpy(), u_exact.numpy(), shading="auto", cmap="viridis")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("Exact Solution u(x, y)")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    im = ax.pcolormesh(X.numpy(), Y.numpy(), torch.abs(u_pred - u_exact).numpy(),
                       shading="auto", cmap="hot")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("|PINN - Exact|")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    mid = X.shape[0] // 2
    ax.plot(X[:, mid].numpy(), u_pred[:, mid].numpy(), "b-", label="PINN", lw=2)
    ax.plot(X[:, mid].numpy(), u_exact[:, mid].numpy(), "r--", label="Exact", lw=2)
    ax.set_xlabel("x"); ax.set_ylabel("u(x, 0.5)")
    ax.set_title(f"Cross-section at y = 0.5   (rel. L2 = {error*100:.2f}%)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    interior = torch.rand(500, 2)
    ax.scatter(interior[:, 0], interior[:, 1], s=2, alpha=0.5, label="Interior (sample)")
    edge = torch.linspace(0, 1, 50)
    zeros, ones = torch.zeros(50), torch.ones(50)
    ax.scatter(torch.cat([edge, edge, zeros, ones]), torch.cat([zeros, ones, edge, edge]),
               s=8, c="red", label="Boundary")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("Collocation Point Layout")
    ax.legend(markerscale=3, fontsize=7); ax.set_aspect("equal")

    save_figure(fig, out)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    problem = Poisson2D()
    model = MLP(len(problem.coords), problem.out_dim, width=64, depth=4, seed=args.seed)
    config = TrainConfig(
        epochs=args.epochs, lr=1e-3, lr_step=1500, log_every=max(args.epochs // 6, 1),
        n_interior=N_INTERIOR, n_boundary=N_BOUNDARY, device=args.device, seed=args.seed,
    )

    print("Training PINN for 2-D Poisson...\n")
    history = train(problem, model, config)

    X, Y, u_pred, u_exact = evaluate(problem, model, device)
    error = relative_l2(u_pred.reshape(-1, 1), u_exact.reshape(-1, 1))
    print(f"\nL2 Relative Error: {error:.6f} ({error*100:.3f}%)\n")

    plot(history, X, Y, u_pred, u_exact, error, args.out)
    print("\nDone. Try `pinn_2d_convection_diffusion.py` next!")


if __name__ == "__main__":
    main()
