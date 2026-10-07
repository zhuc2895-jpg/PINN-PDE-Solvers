"""1-D viscous Burgers PINN:  u_t + u u_x - nu u_xx = 0.

    x in [-1, 1], t in [0, 1],  u(x, 0) = -sin(pi x),  u(+-1, t) = 0,
    nu = 0.01 / pi.

A thin entry point over ``pinn.problems.Burgers1D``.  There is no closed-form
solution, so the figure shows the predicted space-time field and profiles rather
than an error map.
"""

from __future__ import annotations

import argparse

import torch

from pinn import MLP, TrainConfig, resolve_device, train
from pinn.evaluate import predict_field
from pinn.plotting import plot_loss_history, plt, save_figure
from pinn.problems import Burgers1D

OUTPUT = "pinn_burgers_result.png"
N_INTERIOR = 2000
N_BOUNDARY = 200
N_INITIAL = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=OUTPUT)
    return parser.parse_args()


def evaluate(problem, model, device):
    nx, nt = 256, 100
    x = torch.linspace(-1.0, 1.0, nx, device=device)
    t = torch.linspace(0.0, 1.0, nt, device=device)
    X, T = torch.meshgrid(x, t, indexing="ij")
    coords = {"x": X.reshape(-1, 1), "t": T.reshape(-1, 1)}
    u = predict_field(model, problem, coords).reshape(nx, nt)
    return x, X, T, u


def plot(history, x, X, T, u, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    plot_loss_history(axes[0], history,
                      keys=[k for k in ("pde", "ic", "bc") if k in history])

    ax = axes[1]
    im = ax.pcolormesh(T.numpy(), X.numpy(), u.numpy(), shading="auto", cmap="RdBu_r")
    ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title("u(x, t) -- PINN Prediction")
    plt.colorbar(im, ax=ax)

    ax = axes[2]
    nt = u.shape[1]
    for t_val in (0.0, 0.25, 0.5, 0.75, 1.0):
        ax.plot(x.numpy(), u[:, int(t_val * (nt - 1))].numpy(), label=f"t={t_val:.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.set_title("Solution Profiles"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    save_figure(fig, out)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    problem = Burgers1D()
    model = MLP(len(problem.coords), problem.out_dim, width=32, depth=4, seed=args.seed)
    config = TrainConfig(
        epochs=args.epochs, lr=1e-3, lr_step=3000, log_every=max(args.epochs // 10, 1),
        n_interior=N_INTERIOR, n_boundary=N_BOUNDARY, n_initial=N_INITIAL,
        device=args.device, seed=args.seed,
    )

    print("Training PINN for 1D Burgers' equation...\n")
    history = train(problem, model, config)

    x, X, T, u = evaluate(problem, model, device)
    print(f"\nu(x, 0) at t=0:  max = {u[:, 0].max():.3f},  min = {u[:, 0].min():.3f}")
    print(f"u(x, 1) at t=1:  max = {u[:, -1].max():.3f},  min = {u[:, -1].min():.3f}\n")

    plot(history, x, X, T, u, args.out)
    print("\nDone.")


if __name__ == "__main__":
    main()
