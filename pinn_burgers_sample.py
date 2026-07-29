"""
PINN Sample: 1D Viscous Burgers' Equation
==========================================
Solves: u_t + u*u_x - (0.01/π)*u_{xx} = 0,  x ∈ [-1, 1], t ∈ [0, 1]

This is a self-contained introduction to Physics-Informed Neural Networks.
Run it with:  python pinn_burgers_sample.py

Core PINN concepts demonstrated:
  1. A neural network that approximates u(x, t)
  2. Automatic differentiation to compute PDE residuals
  3. Multi-component loss: PDE residual + initial + boundary conditions
  4. Collocation points sampled from the domain interior and boundaries
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hyperparameters ───────────────────────────────────────────────────────────
NU = 0.01 / np.pi          # viscosity
LAYERS = [2, 32, 32, 32, 32, 1]  # input(2) → 4 hidden(32) → output(1)
EPOCHS = 10000
LR = 1e-3

N_INTERIOR = 2000          # collocation points inside the domain
N_BOUNDARY = 200           # points on the boundary (x=-1, x=1)
N_INITIAL  = 200           # points on the initial condition (t=0)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NEURAL NETWORK — approximates u(x, t)
# ═══════════════════════════════════════════════════════════════════════════════
class PINN(nn.Module):
    """
    A plain feedforward network.
    Input:  (x, t)  → shape (N, 2)
    Output: u(x, t) → shape (N, 1)

    The network learns the mapping from space-time coordinates to the
    solution of the PDE.
    """
    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.append(nn.Linear(layers[i], layers[i + 1]))
            self.net.append(nn.Tanh())          # tanh is smooth → good for AD
        self.net.append(nn.Linear(layers[-2], layers[-1]))  # output layer, no activation

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def pde_residual(model, x, t):
    """
    Compute the Burgers' equation residual at collocation points.

    Residual = u_t + u * u_x - ν * u_xx

    If the model satisfies the PDE, this residual is zero everywhere.
    The PINN minimizes the squared residual to enforce the physics.
    """
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(torch.cat([x, t], dim=1))          # u(x, t)

    # First derivatives via automatic differentiation
    grads = torch.autograd.grad(u, [x, t],
                                grad_outputs=torch.ones_like(u),
                                create_graph=True)
    u_x, u_t = grads[0], grads[1]

    # Second derivative u_xx
    u_xx = torch.autograd.grad(u_x, x,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]

    # Burgers residual:  u_t + u * u_x - ν * u_xx = 0
    residual = u_t + u * u_x - NU * u_xx
    return torch.mean(residual ** 2)


def initial_loss(model, x_init, u_init):
    """MSE between model prediction and known initial condition u(x, 0)."""
    t_zero = torch.zeros_like(x_init)
    u_pred = model(torch.cat([x_init, t_zero], dim=1))
    return torch.mean((u_pred - u_init) ** 2)


def boundary_loss(model, x_boundary, t_boundary):
    """MSE for Dirichlet BC: u(-1, t) = u(1, t) = 0."""
    u_pred = model(torch.cat([x_boundary, t_boundary], dim=1))
    return torch.mean(u_pred ** 2)   # target is 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAINING DATA
# ═══════════════════════════════════════════════════════════════════════════════
def generate_data():
    """Sample collocation points from interior, boundary, and initial condition."""
    # Interior points (Latin hypercube-style random sampling)
    x_int = (torch.rand(N_INTERIOR, 1) * 2 - 1).to(DEVICE)   # x ∈ [-1, 1]
    t_int = torch.rand(N_INTERIOR, 1).to(DEVICE)              # t ∈ [0, 1]

    # Boundary points: x = -1 and x = 1
    x_bc = (torch.rand(N_BOUNDARY, 1) * 2 - 1).to(DEVICE)
    t_bc = torch.rand(N_BOUNDARY, 1).to(DEVICE)

    # Initial condition: u(x, 0) = -sin(π * x)
    x_ic = (torch.rand(N_INITIAL, 1) * 2 - 1).to(DEVICE)
    u_ic = -torch.sin(np.pi * x_ic).to(DEVICE)

    return x_int, t_int, x_bc, t_bc, x_ic, u_ic


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def train():
    model = PINN(LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

    # Generate fixed training data
    x_int, t_int, x_bc, t_bc, x_ic, u_ic = generate_data()

    history = {"pde": [], "ic": [], "bc": [], "total": []}

    print(f"{'Epoch':>6s}  {'Total Loss':>12s}  {'PDE':>10s}  {'IC':>10s}  {'BC':>10s}")
    print("-" * 56)

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()

        loss_pde  = pde_residual(model, x_int, t_int)
        loss_ic   = initial_loss(model, x_ic, u_ic)
        loss_bc   = boundary_loss(model, x_bc, t_bc)
        loss_total = loss_pde + loss_ic + loss_bc

        loss_total.backward()
        optimizer.step()
        scheduler.step()

        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())
        history["bc"].append(loss_bc.item())
        history["total"].append(loss_total.item())

        if epoch % 1000 == 0:
            print(f"{epoch:6d}  {loss_total.item():12.6e}  "
                  f"{loss_pde.item():10.2e}  {loss_ic.item():10.2e}  {loss_bc.item():10.2e}")

    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def plot_results(model, history):
    model.eval()

    # Create a fine grid for prediction
    nx, nt = 256, 100
    x = torch.linspace(-1, 1, nx)
    t = torch.linspace(0, 1, nt)
    X, T = torch.meshgrid(x, t, indexing="ij")

    x_flat = X.reshape(-1, 1).to(DEVICE)
    t_flat = T.reshape(-1, 1).to(DEVICE)

    with torch.no_grad():
        u_pred = model(torch.cat([x_flat, t_flat], dim=1)).cpu().reshape(nx, nt)

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig)

    # ── Subplot 1: Loss curves ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(history["total"], label="Total", color="black", lw=1)
    ax1.semilogy(history["pde"],   label="PDE",   alpha=0.7)
    ax1.semilogy(history["ic"],    label="IC",    alpha=0.7)
    ax1.semilogy(history["bc"],    label="BC",    alpha=0.7)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training Losses"); ax1.legend(); ax1.grid(True, alpha=0.3)

    # ── Subplot 2: Spacetime heatmap ──
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.pcolormesh(T.numpy(), X.numpy(), u_pred.numpy(), shading="auto", cmap="RdBu_r")
    ax2.set_xlabel("t"); ax2.set_ylabel("x")
    ax2.set_title("u(x, t) — PINN Prediction")
    plt.colorbar(im, ax=ax2)

    # ── Subplot 3: Snapshots at different times ──
    ax3 = fig.add_subplot(gs[0, 2])
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    for t_val in times:
        idx = int(t_val * (nt - 1))
        ax3.plot(x.numpy(), u_pred[:, idx].numpy(), label=f"t={t_val:.2f}")
    ax3.set_xlabel("x"); ax3.set_ylabel("u")
    ax3.set_title("Solution Profiles"); ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pinn_burgers_result.png", dpi=150)
    plt.show()
    print("\nSaved: pinn_burgers_result.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Training PINN for 1D Burgers' equation...\n")
    model, history = train()
    plot_results(model, history)
    print("\nDone.")
