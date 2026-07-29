"""
PINN Step 3: 2D Lid-Driven Cavity (Steady Navier-Stokes)
==========================================================
Solves the steady incompressible Navier-Stokes equations:

    Continuity:    u_x + v_y = 0
    X-Momentum:    u·u_x + v·u_y = -p_x + ν(u_xx + u_yy)
    Y-Momentum:    u·v_x + v·v_y = -p_y + ν(v_xx + v_yy)

    Domain:  [0, 1] × [0, 1]
    BCs:     Top lid (y=1):  u = 1, v = 0    ← drives the flow
             Other walls:     u = 0, v = 0    ← no-slip
    Re = 100 (ν = 0.01)

Network outputs (u, v, p) simultaneously — the three fields are coupled
through the PDEs. This is the fundamental challenge of PINNs for fluids.

Reference: Ghia et al. (1982) — the standard benchmark solution.

What you learn here:
  • Multi-output networks for coupled PDE systems
  • Nonlinear convection (u·∇u, not just linear advection)
  • Pressure-velocity coupling via the continuity equation
  • Corner singularities and why lid-driven cavity is hard
  • Why you need many more collocation points than Poisson
  • Loss balancing — momentum vs continuity vs BCs may need different weights

Run:  python pinn_2d_navier_stokes.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Physics
RE = 100.0                       # Reynolds number
NU = 1.0 / RE                    # kinematic viscosity = 0.01
RHO = 1.0                        # density (absorbed into pressure)

# Network — wider than Poisson: more physics to capture
HIDDEN = [2, 64, 64, 64, 64, 64, 3]  # (x,y) → (u, v, p)

# Training
N_INTERIOR = 2048
N_BOUNDARY = 400                 # 100 per wall
EPOCHS = 5000
LR = 1e-3

# Loss weights — continuity often needs help (it's a hard constraint)
W_CONT = 1.0                     # continuity weight
W_MOM  = 1.0                     # momentum weight
W_BC   = 10.0                    # BC weight — higher to enforce walls strongly


# ──────────────────────────────────────────────────────────────────────────────
# 1. NETWORK — 2 inputs (x,y), 3 outputs (u, v, p)
# ──────────────────────────────────────────────────────────────────────────────
# The three fields share a common trunk (they're physically coupled).
# The output is NOT activated — u, v, p can be positive or negative.

class NavierStokesNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        blocks = []
        for i in range(len(layers) - 1):
            blocks.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:              # no activation on output
                blocks.append(nn.Tanh())
        self.net = nn.Sequential(*blocks)

    def forward(self, x, y):
        """
        Returns:
            u: (N, 1) x-velocity
            v: (N, 1) y-velocity
            p: (N, 1) pressure
        """
        out = self.net(torch.cat([x, y], dim=1))
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p


# ──────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE ALL DERIVATIVES IN ONE PASS
# ──────────────────────────────────────────────────────────────────────────────
# We need u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy.
# Doing this efficiently matters — each autograd.grad call is expensive.

def compute_residuals(model, x, y):
    """
    Returns the three PDE residuals at collocation points:
      R_cont   = u_x + v_y                           (continuity)
      R_mom_x  = u·u_x + v·u_y + p_x - ν(u_xx+u_yy) (x-momentum)
      R_mom_y  = u·v_x + v·v_y + p_y - ν(v_xx+v_yy) (y-momentum)
    """
    x.requires_grad_(True)
    y.requires_grad_(True)

    u, v, p = model(x, y)

    # ── First derivatives ──
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                              create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v),
                              create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p),
                              create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p),
                              create_graph=True)[0]

    # ── Second derivatives ──
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                               create_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x),
                               create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y),
                               create_graph=True)[0]

    # ── Residuals ──
    # Continuity: u_x + v_y = 0
    R_cont = u_x + v_y

    # X-momentum: u·u_x + v·u_y + (1/ρ)·p_x - ν(u_xx + u_yy) = 0
    R_mom_x = u * u_x + v * u_y + (1.0 / RHO) * p_x - NU * (u_xx + u_yy)

    # Y-momentum: u·v_x + v·v_y + (1/ρ)·p_y - ν(v_xx + v_yy) = 0
    R_mom_y = u * v_x + v * v_y + (1.0 / RHO) * p_y - NU * (v_xx + v_yy)

    return R_cont, R_mom_x, R_mom_y


def pde_losses(model, x, y):
    """MSE of each PDE residual."""
    R_cont, R_mom_x, R_mom_y = compute_residuals(model, x, y)
    loss_cont  = torch.mean(R_cont ** 2)
    loss_mom_x = torch.mean(R_mom_x ** 2)
    loss_mom_y = torch.mean(R_mom_y ** 2)
    return loss_cont, loss_mom_x, loss_mom_y


# ──────────────────────────────────────────────────────────────────────────────
# 3. BOUNDARY CONDITIONS
# ──────────────────────────────────────────────────────────────────────────────
# Lid-driven cavity has 4 distinct BC regions:

def boundary_losses(model):
    """
    Enforce no-slip on all walls, plus lid velocity U=1 on top.
    Returns losses for each wall.
    """
    n = 100  # points per wall

    # ── Top wall (y=1):  u=1, v=0  (LID — drives the flow) ──
    x_top = torch.rand(n, 1).to(DEVICE)
    y_top = torch.ones(n, 1).to(DEVICE)
    u_top, v_top, _ = model(x_top, y_top)
    loss_top_u = torch.mean((u_top - 1.0) ** 2)  # u = 1
    loss_top_v = torch.mean(v_top ** 2)           # v = 0

    # ── Bottom wall (y=0):  u=0, v=0 ──
    x_bot = torch.rand(n, 1).to(DEVICE)
    y_bot = torch.zeros(n, 1).to(DEVICE)
    u_bot, v_bot, _ = model(x_bot, y_bot)
    loss_bot = torch.mean(u_bot ** 2) + torch.mean(v_bot ** 2)

    # ── Left wall (x=0):  u=0, v=0 ──
    y_left = torch.rand(n, 1).to(DEVICE)
    x_left = torch.zeros(n, 1).to(DEVICE)
    u_left, v_left, _ = model(x_left, y_left)
    loss_left = torch.mean(u_left ** 2) + torch.mean(v_left ** 2)

    # ── Right wall (x=1):  u=0, v=0 ──
    y_right = torch.rand(n, 1).to(DEVICE)
    x_right = torch.ones(n, 1).to(DEVICE)
    u_right, v_right, _ = model(x_right, y_right)
    loss_right = torch.mean(u_right ** 2) + torch.mean(v_right ** 2)

    # Total BC loss
    loss_bc = loss_top_u + loss_top_v + loss_bot + loss_left + loss_right
    return loss_bc


# ──────────────────────────────────────────────────────────────────────────────
# 4. TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train():
    print(f"Training 2D Lid-Driven Cavity PINN on {DEVICE}")
    print(f"  Re = {RE:.0f}  (ν = {NU:.4f})")
    print(f"  Interior: {N_INTERIOR}  Boundary: {N_BOUNDARY} (100/wall)")
    print(f"  Architecture: {HIDDEN}")
    print(f"  Loss weights: W_BC={W_BC}, W_CONT={W_CONT}, W_MOM={W_MOM}")
    print()

    model = NavierStokesNet(HIDDEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    # Fixed collocation points in interior
    x_int = torch.rand(N_INTERIOR, 1).to(DEVICE)
    y_int = torch.rand(N_INTERIOR, 1).to(DEVICE)

    history = {"cont": [], "mom_x": [], "mom_y": [], "bc": [], "total": []}

    print(f"{'Epoch':>6s}  {'Total':>12s}  {'Cont':>10s}  "
          f"{'Mom-X':>10s}  {'Mom-Y':>10s}  {'BC':>10s}")
    print("-" * 66)

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()

        loss_cont, loss_mom_x, loss_mom_y = pde_losses(model, x_int, y_int)
        loss_bc = boundary_losses(model)

        loss_total = (W_CONT * loss_cont
                      + W_MOM * (loss_mom_x + loss_mom_y)
                      + W_BC * loss_bc)

        loss_total.backward()
        optimizer.step()
        scheduler.step()

        history["cont"].append(loss_cont.item())
        history["mom_x"].append(loss_mom_x.item())
        history["mom_y"].append(loss_mom_y.item())
        history["bc"].append(loss_bc.item())
        history["total"].append(loss_total.item())

        if epoch % 500 == 0:
            print(f"{epoch:6d}  {loss_total.item():12.6e}  {loss_cont.item():10.2e}  "
                  f"{loss_mom_x.item():10.2e}  {loss_mom_y.item():10.2e}  {loss_bc.item():10.2e}")

    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# 5. GHIA REFERENCE DATA (u along vertical centerline at x=0.5)
# ──────────────────────────────────────────────────────────────────────────────
# Ghia et al. (1982), Re=100, u-velocity at x=0.5
# y from 0 (bottom) to 1 (top)

GHIA_Y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                   0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                   0.9688, 0.9766, 1.0000])
GHIA_U = np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                   -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                   0.68717, 0.73722, 0.78871, 0.84123, 1.00000])


# ──────────────────────────────────────────────────────────────────────────────
# 6. VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(model, history):
    model.eval()

    N = 128
    x = torch.linspace(0, 1, N, device=DEVICE)
    y = torch.linspace(0, 1, N, device=DEVICE)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)

    with torch.no_grad():
        u, v, p = model(x_flat, y_flat)
        u = u.reshape(N, N).cpu()
        v = v.reshape(N, N).cpu()
        p = p.reshape(N, N).cpu()

    speed = torch.sqrt(u**2 + v**2)

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"Lid-Driven Cavity — PINN Prediction (Re = {RE:.0f})",
                 fontsize=14, fontweight="bold")
    gs = GridSpec(2, 4, figure=fig)

    # ── Row 1: Losses | u-velocity | v-velocity | Streamlines ──
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(history["total"], label="Total", color="black", lw=1.2)
    ax.semilogy(history["mom_x"], label="Mom-X", alpha=0.5)
    ax.semilogy(history["mom_y"], label="Mom-Y", alpha=0.5)
    ax.semilogy(history["cont"], label="Continuity", alpha=0.5)
    ax.semilogy(history["bc"], label="BC", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    ax.set_title("Training Losses"); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.pcolormesh(X.numpy().T, Y.numpy().T, u.numpy().T,
                       shading="auto", cmap="RdBu_r")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("u-velocity"); plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.pcolormesh(X.numpy().T, Y.numpy().T, v.numpy().T,
                       shading="auto", cmap="RdBu_r")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("v-velocity"); plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[0, 3])
    # Streamlines — use 1D coords + transpose for (rows=y, cols=x) orientation
    x1d = x.cpu().numpy()
    y1d = y.cpu().numpy()
    ax.streamplot(x1d, y1d, u.numpy().T, v.numpy().T,
                  color=speed.numpy().T, cmap="viridis", density=1.5, linewidth=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Streamlines"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ── Row 2: Speed | Pressure | Centerline vs Ghia | Corner zoom ──
    ax = fig.add_subplot(gs[1, 0])
    im = ax.pcolormesh(X.numpy().T, Y.numpy().T, speed.numpy().T,
                       shading="auto", cmap="hot")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Speed |u|"); plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[1, 1])
    # Pressure (shift to ~0 mean for visualization)
    p_plot = p - p.mean()
    im = ax.pcolormesh(X.numpy().T, Y.numpy().T, p_plot.numpy().T,
                       shading="auto", cmap="RdBu_r")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Pressure (zero-mean)"); plt.colorbar(im, ax=ax)

    # Centerline u-velocity vs Ghia
    ax = fig.add_subplot(gs[1, 2])
    mid = N // 2
    u_center = u[mid, :].numpy()    # u at x=0.5, varies with y
    y_vals = y.cpu().numpy()
    ax.plot(u_center, y_vals, 'b-', lw=2, label="PINN (x=0.5)")
    ax.plot(GHIA_U, GHIA_Y, 'ro', ms=4, label="Ghia et al. (1982)")
    ax.set_xlabel("u-velocity"); ax.set_ylabel("y")
    ax.set_title("Centerline u(x=0.5, y) vs Ghia")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='gray', ls='--', alpha=0.5)

    # Bottom-left corner zoom — where secondary vortex lives
    ax = fig.add_subplot(gs[1, 3])
    n_corner = N // 4
    x1d_c = x1d[:n_corner]
    y1d_c = y1d[:n_corner]
    u_c = u[:n_corner, :n_corner].numpy().T   # transpose: rows=y, cols=x
    v_c = v[:n_corner, :n_corner].numpy().T
    spd_c = speed[:n_corner, :n_corner].numpy().T
    ax.streamplot(x1d_c, y1d_c, u_c, v_c,
                  color=spd_c, cmap="viridis", density=2, linewidth=0.8)
    ax.set_xlim(0, 0.25); ax.set_ylim(0, 0.25)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Bottom-Left Corner (secondary vortex)")

    plt.tight_layout()
    plt.savefig("pinn_navier_stokes_result.png", dpi=150)
    plt.close()
    print("Saved: pinn_navier_stokes_result.png")


# ──────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, history = train()
    plot_results(model, history)
    print("\nDone! This is the end of the PINN learning path.")
    print("Next frontiers: unsteady flows, adaptive sampling, hard BC constraints,")
    print("curriculum learning, or inverse problems (parameter discovery).")
