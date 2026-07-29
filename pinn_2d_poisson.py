"""
PINN Step 1: 2D Poisson Equation
=================================
Solves:  -∇²u = f(x, y)   on Ω = [0,1] × [0,1]
          u = 0            on ∂Ω           (homogeneous Dirichlet BC)

We use a manufactured solution so we can measure error:
    u_exact(x, y) = sin(π·x) · sin(π·y)
    → f(x, y)     = 2π² · sin(π·x) · sin(π·y)

What you learn from this example:
  • Sampling collocation points in 2D (interior vs boundary)
  • Computing ∇²u = u_xx + u_yy via torch.autograd
  • Building the loss: MSE(PDE residual) + MSE(BC mismatch)
  • Measuring L² error against the known exact solution
  • Visualizing predictions vs ground truth on a 2D grid

Run:  python pinn_2d_poisson.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive — no GUI, save to file only
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network architecture
HIDDEN_LAYERS = [2, 64, 64, 64, 64, 1]  # 2 inputs (x,y) → 1 output u

# Training
N_INTERIOR = 512         # collocation points in the interior
N_BOUNDARY = 128         # points on the boundary (4 edges)
EPOCHS = 1500
LR = 1e-3

# ──────────────────────────────────────────────────────────────────────────────
# 1. THE NEURAL NETWORK
# ──────────────────────────────────────────────────────────────────────────────
# Input:  (x, y)  — a point in the 2D domain
# Output: u(x, y) — the predicted solution at that point
#
# Why tanh?  It's smooth (C∞), so its second derivative is well-behaved.
# ReLU would give zero second derivatives, which kills PDE residual learning.
# Sin activation also works well for periodic problems; tanh is the safe default.


class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        blocks = []
        for i in range(len(layers) - 1):
            blocks.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:          # no activation after output layer
                blocks.append(nn.Tanh())
        self.net = nn.Sequential(*blocks)

    def forward(self, x, y):
        """
        Args:
            x: (N, 1) tensor of x-coordinates
            y: (N, 1) tensor of y-coordinates
        Returns:
            u: (N, 1) tensor of predicted u(x, y)
        """
        return self.net(torch.cat([x, y], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# 2. THE PHYSICS LOSS
# ──────────────────────────────────────────────────────────────────────────────
# The Poisson residual is:  R = -∇²u - f(x,y)  → should be 0
#
# We compute ∇²u = ∂²u/∂x² + ∂²u/∂y² using automatic differentiation.
# torch.autograd.grad with create_graph=True lets us differentiate through
# the first derivative to get the second derivative AND keep the computation
# graph so we can backpropagate through it.

def source_term(x, y):
    """f(x,y) = 2π²·sin(πx)·sin(πy) — from the manufactured solution."""
    return 2.0 * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y)


def pde_residual(model, x, y):
    """
    Compute MSE of the Poisson residual at collocation points.
    This is the physics-informed loss — the heart of a PINN.
    """
    x.requires_grad_(True)
    y.requires_grad_(True)

    u = model(x, y)

    # ── First derivatives ──
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]

    # ── Second derivatives ──
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                               create_graph=True)[0]

    laplacian = u_xx + u_yy              # ∇²u
    f = source_term(x, y)

    residual = -laplacian - f            # -∇²u - f  → target is 0
    return torch.mean(residual ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# 3. THE BOUNDARY CONDITION LOSS
# ──────────────────────────────────────────────────────────────────────────────
# u = 0 on all four edges: x=0, x=1, y=0, y=1
#
# We sample points on each edge, predict u, and penalize any non-zero value.

def boundary_loss(model, x_bc, y_bc):
    u_pred = model(x_bc, y_bc)
    return torch.mean(u_pred ** 2)       # target is u=0


# ──────────────────────────────────────────────────────────────────────────────
# 4. DATA GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def sample_interior(n):
    """Uniform random points inside the unit square [0,1]×[0,1]."""
    x = torch.rand(n, 1).to(DEVICE)
    y = torch.rand(n, 1).to(DEVICE)
    return x, y


def sample_boundary(n):
    """
    Sample points on the four edges of [0,1]×[0,1].
    We assign equal numbers to each edge so BCs are enforced evenly.
    """
    n_edge = n // 4
    r = n % 4  # remainder goes to the first edge

    # Bottom edge: y=0, x∈[0,1]
    x0 = torch.rand(n_edge + (1 if r > 0 else 0), 1).to(DEVICE)
    y0 = torch.zeros_like(x0)

    # Top edge: y=1, x∈[0,1]
    x1 = torch.rand(n_edge + (1 if r > 1 else 0), 1).to(DEVICE)
    y1 = torch.ones_like(x1)

    # Left edge: x=0, y∈[0,1]
    y2 = torch.rand(n_edge + (1 if r > 2 else 0), 1).to(DEVICE)
    x2 = torch.zeros_like(y2)

    # Right edge: x=1, y∈[0,1]
    y3 = torch.rand(n_edge, 1).to(DEVICE)
    x3 = torch.ones_like(y3)

    x = torch.cat([x0, x1, x2, x3], dim=0)
    y = torch.cat([y0, y1, y2, y3], dim=0)
    return x, y


# ──────────────────────────────────────────────────────────────────────────────
# 5. TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def exact_solution(x, y):
    """Manufactured solution: u(x,y) = sin(πx)·sin(πy)."""
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)


def train():
    model = PINN(HIDDEN_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.5)

    # Generate fixed training points
    x_int, y_int = sample_interior(N_INTERIOR)
    x_bc, y_bc   = sample_boundary(N_BOUNDARY)

    history = {"pde": [], "bc": [], "total": []}

    print(f"Training 2D Poisson PINN on {DEVICE}")
    print(f"  Interior points: {x_int.shape[0]}")
    print(f"  Boundary points: {x_bc.shape[0]}")
    print(f"  Architecture:    {HIDDEN_LAYERS}")
    print()
    print(f"{'Epoch':>6s}  {'Total Loss':>12s}  {'PDE Loss':>10s}  {'BC Loss':>10s}")
    print("-" * 50)

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
# 6. EVALUATION & VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_and_plot(model, history):
    model.eval()

    # Create a fine grid for evaluation
    N = 128
    x = torch.linspace(0, 1, N, device=DEVICE)
    y = torch.linspace(0, 1, N, device=DEVICE)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)

    with torch.no_grad():
        u_pred = model(x_flat, y_flat).reshape(N, N).cpu()

    u_exact = exact_solution(X, Y).cpu()

    # L² relative error
    error = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
    print(f"\nL2 Relative Error: {error.item():.6f} ({error.item()*100:.3f}%)\n")

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

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
    ax.set_title("|PINN - Exact|")
    plt.colorbar(im, ax=ax)

    # Cross-section at y = 0.5
    ax = axes[1, 1]
    mid = N // 2
    ax.plot(x.cpu(), u_pred[:, mid], 'b-', label="PINN", lw=2)
    ax.plot(x.cpu(), u_exact[:, mid], 'r--', label="Exact", lw=2)
    ax.set_xlabel("x"); ax.set_ylabel("u(x, 0.5)")
    ax.set_title("Cross-section at y = 0.5")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Training point distribution
    ax = axes[1, 2]
    ax.scatter(torch.rand(500).cpu(), torch.rand(500).cpu(),
              s=2, alpha=0.5, label="Interior (sample)")
    # Plot boundary sample
    bx = torch.cat([torch.linspace(0, 1, 50), torch.linspace(0, 1, 50),
                    torch.zeros(50), torch.ones(50)]).cpu()
    by = torch.cat([torch.zeros(50), torch.ones(50),
                    torch.linspace(0, 1, 50), torch.linspace(0, 1, 50)]).cpu()
    ax.scatter(bx, by, s=8, c='red', label="Boundary")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Collocation Point Layout")
    ax.legend(markerscale=3, fontsize=7); ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("pinn_poisson_result.png", dpi=150)
    plt.close()
    print("Saved: pinn_poisson_result.png")


# ──────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, history = train()
    evaluate_and_plot(model, history)
    print("\nDone. Try `pinn_2d_convection_diffusion.py` next!")
