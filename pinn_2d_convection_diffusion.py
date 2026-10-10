"""2-D convection-diffusion PINN:  beta.grad(u) - eps*Laplacian(u) = f.

    u_exact = sin(pi x) sin(pi y),  beta = (1, 1), eps = 0.1  (Pe ~ 10)

A thin entry point over ``pinn.problems.ConvectionDiffusion2D``.  Change BETA or
EPSILON below to see how the Peclet number affects training.
"""

from __future__ import annotations

import argparse

import torch

from pinn import MLP, TrainConfig, resolve_device, train
from pinn.evaluate import predict_field, relative_l2, unit_square_grid
from pinn.plotting import plot_loss_history, plt, save_figure
from pinn.problems import ConvectionDiffusion2D

OUTPUT = "pinn_convection_diffusion_result.png"
BETA = (1.0, 1.0)
EPSILON = 0.1
N_INTERIOR = 512
N_BOUNDARY = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=OUTPUT)
    return parser.parse_args()


def evaluate(problem, model, device):
    n = 128
    X, Y, coords = unit_square_grid(n, device=device)
    u_pred = predict_field(model, problem, coords).reshape(n, n)
    u_exact = problem.exact(coords)["u"].reshape(n, n).cpu()
    return X, Y, u_pred, u_exact


def plot(history, X, Y, u_pred, u_exact, error, pe, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Convection-Diffusion:  beta=({BETA[0]},{BETA[1]}),  "
                 f"eps={EPSILON},  Pe~{pe:.0f}", fontsize=13)

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
    im = ax.pcolormesh(X.numpy(), Y.numpy(), u_exact.numpy(), shading="auto", cmap="viridis")
    ax.arrow(0.5, 0.5, BETA[0] * 0.15, BETA[1] * 0.15, color="red",
             head_width=0.04, length_includes_head=True)
    ax.text(0.5 + BETA[0] * 0.19, 0.5 + BETA[1] * 0.19, f"beta=({BETA[0]},{BETA[1]})",
            color="red", fontsize=10, ha="center")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(f"Advection direction (Pe~{pe:.0f})")
    plt.colorbar(im, ax=ax)

    save_figure(fig, out)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    problem = ConvectionDiffusion2D(beta=BETA, eps=EPSILON)
    pe = max(abs(BETA[0]), abs(BETA[1])) / EPSILON if EPSILON > 0 else float("inf")

    model = MLP(len(problem.coords), problem.out_dim, width=64, depth=4, seed=args.seed)
    config = TrainConfig(
        epochs=args.epochs, lr=1e-3, lr_step=800, log_every=max(args.epochs // 8, 1),
        n_interior=N_INTERIOR, n_boundary=N_BOUNDARY, device=args.device, seed=args.seed,
    )

    print("Training PINN for 2-D convection-diffusion...\n")
    print(f"  beta = ({BETA[0]}, {BETA[1]}),  eps = {EPSILON},  Pe ~ {pe:.0f}")
    history = train(problem, model, config)

    X, Y, u_pred, u_exact = evaluate(problem, model, device)
    error = relative_l2(u_pred.reshape(-1, 1), u_exact.reshape(-1, 1))
    print(f"\nL2 Relative Error: {error:.6f} ({error*100:.3f}%)\n")

    plot(history, X, Y, u_pred, u_exact, error, pe, args.out)
    print("\nDone.")


if __name__ == "__main__":
    main()
