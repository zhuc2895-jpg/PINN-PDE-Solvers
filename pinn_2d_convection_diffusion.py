"""
PINN Step 2: 2D Steady Convection-Diffusion Equation
======================================================
Solves:  β·∇u - ε·∇²u = f(x, y)   on Ω = [0,1] × [0,1]
          u = 0                     on ∂Ω

More explicitly:
    β_x·u_x + β_y·u_y - ε(u_xx + u_yy) = f(x,y)

Key difference from Poisson: the convection term β·∇u is FIRST-order (not second),
making the PDE non-symmetric and introducing a preferred transport direction.

Manufactured solution (same as Step 1 for comparison):
    u_exact(x, y) = sin(π·x) · sin(π·y)
    → f(x,y) = β_x·π·cos(πx)·sin(πy)
              + β_y·π·sin(πx)·cos(πy)
              + ε·2π²·sin(πx)·sin(πy)

What you learn here:
  • Computing first-order spatial derivatives via autograd
  • The convection vs diffusion balance (Péclet number Pe = |β|/ε)
  • Why high Pe is hard: sharp boundary layers need many collocation points
  • Loss weighting — sometimes you need to upweight BC/PDE differently
  • Comparing with Poisson: same exact solution, different physics

Try changing BETA and EPSILON to see how Pe affects training.

Run:  python pinn_2d_convection_diffusion.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — tweak these to experiment
# ──────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Physics parameters
BETA_X = 1.0                 # convection velocity in x
BETA_Y = 1.0                 # convection velocity in y
EPSILON = 0.1                # diffusion coefficient
# Péclet number ≈ max(|β_x|, |β_y|) / ε = 1/0.1 = 10 (moderate convection)
# Try: ε=0.01 → Pe=100 (hard), ε=1.0 → Pe=1 (easy, diffusion-dominated)

# Training
HIDDEN_LAYERS = [2, 64, 64, 64, 64, 1]
N_INTERIOR = 512
N_BOUNDARY = 128
EPOCHS = 2000
LR = 1e-3

# ──────────────────────────────────────────────────────────────────────────────
# 1. NETWORK (same architecture as Poisson)
# ──────────────────────────────────────────────────────────────────────────────

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        blocks = []
        for i in range(len(layers) - 1):
            blocks.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                blocks.append(nn.Tanh())
        self.net = nn.Sequential(*blocks)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# 2. PHYSICS LOSS — the convection-diffusion residual
# ──────────────────────────────────────────────────────────────────────────────
# Residual:  R = β_x·u_x + β_y·u_y - ε(u_xx + u_yy) - f(x,y)
#
# This has TWO kinds of derivatives:
#   First-order  (u_x, u_y)  — easy, one autograd call
#   Second-order (u_xx, u_yy) — need create_graph=True chain (same as Poisson)

def source_term(x, y):
    """f(x,y) derived from the manufactured solution u = sin(πx)sin(πy)."""
    # β·∇u part
    conv = BETA_X * np.pi * torch.cos(np.pi * x) * torch.sin(np.pi * y) \
         + BETA_Y * np.pi * torch.sin(np.pi * x) * torch.cos(np.pi * y)
    # -ε∇²u part: -ε * (-2π²·sin(πx)sin(πy)) = +ε·2π²·sin(πx)sin(πy)
    diff = EPSILON * 2.0 * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y)
    return conv + diff


def pde_residual(model, x, y):
    """MSE of the convection-diffusion residual."""
    x.requires_grad_(True)
    y.requires_grad_(True)

    u = model(x, y)

    # First derivatives (for convection)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]

    # Second derivatives (for diffusion)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                               create_graph=True)[0]

    # Convection-diffusion residual:
    #   β·∇u - ε·∇²u - f = 0
    convection = BETA_X * u_x + BETA_Y * u_y
    diffusion  = EPSILON * (u_xx + u_yy)
    f = source_term(x, y)

    residual = convection - diffusion - f
    return torch.mean(residual ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# 3. BOUNDARY CONDITION LOSS
# ──────────────────────────────────────────────────────────────────────────────

def boundary_loss(model, x_bc, y_bc):
    u_pred = model(x_bc, y_bc)
    return torch.mean(u_pred ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# 4. DATA GENERATION (same as Poisson)
# ──────────────────────────────────────────────────────────────────────────────

def sample_interior(n):
    x = torch.rand(n, 1).to(DEVICE)
    y = torch.rand(n, 1).to(DEVICE)
    return x, y


def sample_boundary(n):
    n_edge = n // 4
    r = n % 4

    x0 = torch.rand(n_edge + (1 if r > 0 else 0), 1).to(DEVICE)
    y0 = torch.zeros_like(x0)

    x1 = torch.rand(n_edge + (1 if r > 1 else 0), 1).to(DEVICE)
    y1 = torch.ones_like(x1)

    y2 = torch.rand(n_edge + (1 if r > 2 else 0), 1).to(DEVICE)
    x2 = torch.zeros_like(y2)

    y3 = torch.rand(n_edge, 1).to(DEVICE)
    x3 = torch.ones_like(y3)

    x = torch.cat([x0, x1, x2, x3], dim=0)
    y = torch.cat([y0, y1, y2, y3], dim=0)
    return x, y


# ──────────────────────────────────────────────────────────────────────────────
# 5. TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def exact_solution(x, y):
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)


def train():
    model = PINN(HIDDEN_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)

    x_int, y_int = sample_interior(N_INTERIOR)
    x_bc, y_bc   = sample_boundary(N_BOUNDARY)

    history = {"pde": [], "bc": [], "total": []}

    Pe = max(abs(BETA_X), abs(BETA_Y)) / EPSILON if EPSILON > 0 else float("inf")
    print(f"Training 2D Convection-Diffusion PINN on {DEVICE}")
    print(f"  β = ({BETA_X}, {BETA_Y}),  ε = {EPSILON},  Pe ≈ {Pe:.0f}")
    print(f"  Interior: {x_int.shape[0]}  Boundary: {x_bc.shape[0]}")
    print(f"  Architecture: {HIDDEN_LAYERS}")
    print()
    print(f"{'Epoch':>6s}  {'Total':>12s}  {'PDE':>10s}  {'BC':>10s}")
    print("-" * 46)

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()

        loss_pde = pde_residual(model, x_int, y_int)
        loss_bc  = boundary_loss(model, x_bc, y_bc)
        loss_total = loss_pde + loss_bc

        loss_total.backward()
        optimizer.step()
        scheduler.step()

        history["pde"].append(loss_pde.item())
        history["bc"].append(loss_bc.item())
        history["total"].append(loss_total.item())

        if epoch % 500 == 0:
            print(f"{epoch:6d}  {loss_total.item():12.6e}  "
                  f"{loss_pde.item():10.2e}  {loss_bc.item():10.2e}")

    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# 6. VISUALIZATION — now includes flow direction and Pe annotation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_and_plot(model, history):
    model.eval()

    N = 128
    x = torch.linspace(0, 1, N, device=DEVICE)
    y = torch.linspace(0, 1, N, device=DEVICE)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)

    with torch.no_grad():
        u_pred = model(x_flat, y_flat).reshape(N, N).cpu()

    u_exact = exact_solution(X, Y).cpu()
    error = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
    Pe = max(abs(BETA_X), abs(BETA_Y)) / EPSILON if EPSILON > 0 else float("inf")

    print(f"\nL2 Relative Error: {error.item():.6f} ({error.item()*100:.3f}%)")

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Convection-Diffusion:  β=({BETA_X},{BETA_Y}),  ε={EPSILON},  Pe≈{Pe:.0f}",
                 fontsize=13, fontweight="bold")

    # Loss curves
    ax = axes[0, 0]
    ax.semilogy(history["total"], label="Total", color="black", lw=1.2)
    ax.semilogy(history["pde"], label="PDE", alpha=0.7)
    ax.semilogy(history["bc"], label="BC", alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Losses"); ax.legend(); ax.grid(True, alpha=0.3)

    # PINN prediction
    ax = axes[0, 1]
    im = ax.pcolormesh(X.numpy(), Y.numpy(), u_pred.numpy(),
                       shading="auto", cmap="viridis")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("PINN Prediction u(x, y)")
    plt.colorbar(im, ax=ax)

    # Exact solution
    ax = axes[0, 2]
    im = ax.pcolormesh(X.numpy(), Y.numpy(), u_exact.numpy(),
                       shading="auto", cmap="viridis")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Exact Solution u(x, y)")
    plt.colorbar(im, ax=ax)

    # Absolute error
    ax = axes[1, 0]
    abs_err = torch.abs(u_pred - u_exact)
    im = ax.pcolormesh(X.numpy(), Y.numpy(), abs_err.numpy(),
                       shading="auto", cmap="hot")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"|Error| — L2 = {error.item()*100:.2f}%")
    plt.colorbar(im, ax=ax)

    # Cross-section at y = 0.5
    ax = axes[1, 1]
    mid = N // 2
    ax.plot(x.cpu(), u_pred[:, mid], 'b-', label="PINN", lw=2)
    ax.plot(x.cpu(), u_exact[:, mid], 'r--', label="Exact", lw=2)
    ax.set_xlabel("x"); ax.set_ylabel("u(x, 0.5)")
    ax.set_title("Cross-section at y = 0.5")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Flow field + convection direction
    ax = axes[1, 2]
    # Plot the convection vector field β
    ax.arrow(0.5, 0.5, BETA_X * 0.15, BETA_Y * 0.15,
             head_width=0.04, head_length=0.05, fc='red', ec='red', lw=2)
    ax.text(0.5 + BETA_X * 0.17, 0.5 + BETA_Y * 0.17,
            f"β=({BETA_X},{BETA_Y})", color='red', fontsize=10, ha='center')
    ax.text(0.5, 0.9, f"Pe ≈ {Pe:.0f}", fontsize=11, ha='center',
            bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    ax.text(0.5, 0.2, "Convection-dominated\n→ boundary layers", fontsize=9,
            ha='center', alpha=0.6 if Pe > 5 else 0.0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Convection Direction & Pe")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("pinn_convection_diffusion_result.png", dpi=150)
    plt.close()
    print("Saved: pinn_convection_diffusion_result.png")


# ──────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, history = train()
    evaluate_and_plot(model, history)

    # Quick comparison note
    Pe = max(abs(BETA_X), abs(BETA_Y)) / EPSILON if EPSILON > 0 else float("inf")
    print(f"\nPe = {Pe:.0f} — ", end="")
    if Pe < 2:
        print("Diffusion-dominated. Easy to train, similar to Poisson.")
    elif Pe < 20:
        print("Moderate convection. PINN should still handle this well.")
    else:
        print("Convection-dominated. Expect slower convergence and higher error.")
        print("To fix high-Pe: use more points near boundaries, or add upwind-biased sampling.")

    print("\nDone. Try `pinn_2d_navier_stokes.py` for Step 3!")
