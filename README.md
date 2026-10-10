# PINN-PDE-Solvers

Physics-Informed Neural Networks (PINNs) for solving classic partial differential
equations (PDEs), implemented in PyTorch with `torch.autograd` for automatic
differentiation of PDE residuals.

Each PDE is one small plugin behind a shared seam — `sample()` / `losses()` /
`loss_weights()` / `exact()`. The network, the autograd derivatives, the sampling
layouts and the training loop are all shared, so anything implemented once
(resampling, adaptive loss weights, logging, L-BFGS fine-tuning) applies to every
PDE for free. The design reasoning and the measurements behind it live in
[docs/architecture-exploration.md](docs/architecture-exploration.md).

## What's Inside

| Script | Equation | Domain | Key Concept |
|--------|----------|--------|-------------|
| `pinn_2d_poisson.py` | 2D Poisson: $-\nabla^2 u = f$ | $[0,1]^2$ | Laplacian via autograd, Dirichlet BCs |
| `pinn_2d_convection_diffusion.py` | 2D Convection-Diffusion: $\beta\cdot\nabla u - \varepsilon\nabla^2 u = f$ | $[0,1]^2$ | First-order derivatives, Peclet number |
| `pinn_2d_navier_stokes.py` | 2D Steady Navier-Stokes (lid-driven cavity) | $[0,1]^2$ | Multi-output networks, velocity-pressure coupling |
| `pinn_burgers_sample.py` | 1D Viscous Burgers': $u_t + u u_x - \nu u_{xx} = 0$ | $[-1,1]\times[0,1]$ | Time-dependent PDE, shock formation |

Each script is a thin entry point that produces a result image (`*_result.png`)
comparing the PINN prediction against the reference solution.

```
pinn/
  derivatives.py   d() / grad_all() / laplacian()   — every autograd derivative
  model.py         MLP(activation=..., n_fourier=...) + Fourier features
  sampling.py      uniform / unit square boundary / spacetime strip
  problem.py       PointSet / Data / the PDEProblem protocol
  trainer.py       the one Adam + StepLR training loop
  evaluate.py      grid evaluation, relative L2, Ghia (1982) benchmark table
  device.py        device selection that will not hang on a mismatched CUDA build
  problems/        poisson, convection_diffusion, burgers, navier_stokes
tests/             residual-integrity and training-loop tests
```

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

Each one accepts `--epochs`, `--seed`, `--out` and `--device {cpu,cuda,auto}`:

```bash
python pinn_2d_poisson.py --epochs 200 --out scratch/quick.png
```

`--device` defaults to **cpu** on purpose. `torch.cuda.is_available()` only says a
CUDA device is present — not that this torch build can run a kernel on it. A build
compiled for an older compute capability reports `True` and then hangs on the first
kernel, which is exactly what happened here (RTX 5060 / `sm_120` under
`torch 1.11.0+cu113`). See `pinn/device.py`.

## Tests

```bash
pip install pytest
python -m pytest
```

The interesting ones plug a closed-form solution into a PDE and check the residual
vanishes, or compare a residual against a hand-derived expression — so they verify
the *operator*, not just that some loss went down.

## How It Works

1. **Neural network** approximates the solution $u(x, y)$ or $u(x, t)$.
2. **Automatic differentiation** (`torch.autograd.grad`) computes the required spatial/temporal derivatives.
3. **Loss function** combines:
   - PDE residual at collocation points (interior)
   - Boundary/initial condition mismatch
4. **Gradient descent** (Adam) minimizes the total loss, training the network to satisfy both the physics and the boundary data.

A PINN's loss is a mean residual over collocation points, not an error estimate —
the two can disagree by an order of magnitude. Accuracy is therefore checked
against an analytic solution where one exists, and against the Ghia et al. (1982)
centreline table for the cavity, which has no closed form.

## Results

Each script saves a visualization:

- `pinn_poisson_result.png`
- `pinn_convection_diffusion_result.png`
- `pinn_navier_stokes_result.png`
- `pinn_burgers_result.png`

## License

MIT
