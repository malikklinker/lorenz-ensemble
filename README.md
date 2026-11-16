# Lorenz ensemble

A Python simulation of sensitivity to initial conditions in the Lorenz system (`σ = 10`, `ρ = 28`, `β = 8/3`). It integrates nearby trajectories with SciPy, computes ensemble statistics and pairwise distances, and fits early separation growth.

## Run

Requires Python, NumPy, SciPy, and Matplotlib:

```sh
python3 -m pip install -r requirements.txt
python3 lorenz.py --seed 0 --members 10 --duration 35
```

Writes `solutions.png`, `distances.png`, and `trajectories.png`, replacing previous output. For headless use, set `MPLBACKEND=Agg`. The seed makes initial conditions reproducible. At least three members are required.

The reported growth rate is a linear fit to log mean separation over the first 75% of the time interval; it depends on the fit window and is not a converged Lyapunov-exponent calculation.
