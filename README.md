# Hydrax

Sampling-based model predictive control on GPU with
[JAX](https://jax.readthedocs.io/) and
[MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html).

## About

Trajectory optimization benchmarks based on Hydrax

## Setup (conda)

Set up a conda env with cuda support (first time only):

```bash
conda env create -f environment.yml
```

Enter the conda env:

```bash
conda activate hydrax
```

Install the package and dependencies:

```bash
pip install -e .

```

## MPC Examples

Launch an interactive pendulum swingup simulation with predictive sampling:

```bash
python examples/pendulum.py ps
```

Launch an interactive humanoid standup simulation (shown above) with MPPI and
online domain randomization:

```bash
python examples/humanoid_standup.py
```
Other demos can be found in the `examples` folder.


## Trajectory optimization Benchmarks:

Run 

```bash
hydrax/simulation/plot_traj_opt.ipynb
```
