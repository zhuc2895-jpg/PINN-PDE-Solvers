# PINN-PDE-Solvers

Physics-Informed Neural Networks (PINNs) for solving classic partial differential equations (PDEs), implemented in PyTorch with `torch.autograd` for automatic differentiation of PDE residuals.

## What's Inside

| Script | Equation | Domain | Key Concept |
|--------|----------|--------|-------------|
| `pinn_2d_poisson.py` | 2D Poisson: $-\nabla^2 u = f$ | $[0,1]^2$ | Laplacian via autograd, Dirichlet BCs |
| `pinn_2d_convection_diffusion.py` | 2D Convection-Diffusion: $\beta\cdot\nabla u - \varepsilon\nabla^2 u = f$ | $[0,1]^2$ | First-order derivatives, Peclet number |
| `pinn_2d_navier_stokes.py` | 2D Steady Navier-Stokes (lid-driven cavity) | $[0,1]^2$ | Multi-output networks, velocity-pressure coupling |
| `pinn_burgers_sample.py` | 1D Viscous Burgers': $u_t + u u_x - \nu u_{xx} = 0$ | $[-1,1]\times[0,1]$ | Time-dependent PDE, shock formation |

Each script is self-contained and produces a result image (`*_result.png`) comparing the PINN prediction against the exact solution.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install torch numpy matplotlib
```

## Usage

Run any script independently:

```bash
python pinn_2d_poisson.py
python pinn_2d_convection_diffusion.py
python pinn_2d_navier_stokes.py
python pinn_burgers_sample.py
```

GPU is used automatically if available (`torch.cuda.is_available()`); otherwise it falls back to CPU.

## How It Works

1. **Neural network** approximates the solution $u(x, y)$ or $u(x, t)$.
2. **Automatic differentiation** (`torch.autograd.grad`) computes the required spatial/temporal derivatives.
3. **Loss function** combines:
   - PDE residual at collocation points (interior)
   - Boundary/initial condition mismatch
4. **Gradient descent** (Adam) minimizes the total loss, training the network to satisfy both the physics and the boundary data.

## Results

Each script saves a visualization comparing predicted vs. exact solutions:

- `pinn_poisson_result.png`
- `pinn_convection_diffusion_result.png`
- `pinn_navier_stokes_result.png`

## License

MIT
