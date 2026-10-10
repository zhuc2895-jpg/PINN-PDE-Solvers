"""2-D steady lid-driven cavity (Navier-Stokes), Re = 100.

Three field outputs (u, v, p), three PDE residuals, four walls with different
targets.  No analytic solution exists, so accuracy is judged against the Ghia
et al. (1982) centreline benchmark table.

A thin entry point over ``pinn.problems.LidDrivenCavity``.
"""

from __future__ import annotations

import argparse

import torch

from pinn import MLP, TrainConfig, resolve_device, train
from pinn.evaluate import ghia_u_centreline, predict_field, unit_square_grid
from pinn.plotting import plot_loss_history, plt, save_figure
from pinn.problems import LidDrivenCavity

OUTPUT = "pinn_navier_stokes_result.png"
RE = 100.0
N_INTERIOR = 2048
N_BOUNDARY = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=OUTPUT)
    return parser.parse_args()


def evaluate(problem, model, device):
    n = 128
    X, Y, coords = unit_square_grid(n, device=device)
    u = predict_field(model, problem, coords, column=0).reshape(n, n)
    v = predict_field(model, problem, coords, column=1).reshape(n, n)
    p = predict_field(model, problem, coords, column=2).reshape(n, n)
    speed = torch.sqrt(u**2 + v**2)
    return X, Y, u, v, p, speed


def ghia_error(u):
    """Compare the predicted u along the vertical centreline x=0.5 (Ghia et al.).

    ``X, Y`` come from ``meshgrid(..., indexing="ij")``, so ``u[i, j]`` is
    ``u(x_i, y_j)``: fixing x = 0.5 means fixing the *first* index.
    """
    n = u.shape[0]
    centreline = u[n // 2, :]
    y_table, u_table = ghia_u_centreline()
    idx = (y_table * (n - 1)).round().long().clamp(0, n - 1)
    return float(torch.max(torch.abs(centreline[idx] - u_table)))


def plot(history, X, Y, u, v, p, speed, max_dev, out):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Lid-Driven Cavity -- PINN Prediction (Re = {RE:.0f})", fontsize=13)

    plot_loss_history(axes[0, 0], history,
                      keys=[k for k in ("cont", "mom_x", "mom_y", "bc") if k in history])

    ax = axes[0, 1]
    im = ax.pcolormesh(X.numpy(), Y.numpy(), u.numpy(), shading="auto", cmap="RdBu_r")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("u-velocity")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    im = ax.pcolormesh(X.numpy(), Y.numpy(), v.numpy(), shading="auto", cmap="RdBu_r")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("v-velocity")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    im = ax.pcolormesh(X.numpy(), Y.numpy(), speed.numpy(), shading="auto", cmap="viridis")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("Speed |(u, v)|")
    plt.colorbar(im, ax=ax)

    # Skip every few points: a 128x128 streamplot grid is needlessly dense.
    ax = axes[1, 1]
    step = 8
    ax.streamplot(X[::step, 0].numpy(), Y[0, ::step].numpy(),
                  u[::step, ::step].numpy(), v[::step, ::step].numpy(),
                  color=speed[::step, ::step].numpy(), cmap="viridis", density=1.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title("Streamlines")

    ax = axes[1, 2]
    n = u.shape[0]
    y_axis = torch.linspace(0, 1, n)
    ax.plot(u[n // 2, :].numpy(), y_axis.numpy(), "b-", lw=2, label="PINN")
    y_table, u_table = ghia_u_centreline()
    ax.plot(u_table.numpy(), y_table.numpy(), "ro", ms=4, label="Ghia et al. (1982)")
    ax.set_xlabel("u"); ax.set_ylabel("y")
    ax.set_title(f"u at x = 0.5   (max dev = {max_dev:.3f})")
    ax.legend(); ax.grid(True, alpha=0.3)

    save_figure(fig, out)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    problem = LidDrivenCavity(re=RE)
    model = MLP(len(problem.coords), problem.out_dim, width=64, depth=5, seed=args.seed)
    config = TrainConfig(
        epochs=args.epochs, lr=1e-3, lr_step=2000, log_every=max(args.epochs // 10, 1),
        n_interior=N_INTERIOR, n_boundary=N_BOUNDARY, device=args.device, seed=args.seed,
    )

    print("Training PINN for the lid-driven cavity...\n")
    print(f"  Re = {RE:.0f}  (nu = {problem.nu:.4f})")
    print(f"  Interior: {N_INTERIOR}  Boundary: {N_BOUNDARY} (100/wall)")
    print(f"  Loss weights: {problem.loss_weights()}\n")
    history = train(problem, model, config)

    X, Y, u, v, p, speed = evaluate(problem, model, device)
    max_dev = ghia_error(u)
    print(f"\nMax deviation from the Ghia centreline table: {max_dev:.4f}\n")

    plot(history, X, Y, u, v, p, speed, max_dev, args.out)
    print("\nDone.")


if __name__ == "__main__":
    main()

