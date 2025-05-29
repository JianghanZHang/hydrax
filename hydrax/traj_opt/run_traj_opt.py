import argparse

import mujoco
import evosax.algorithms as evosax

from hydrax.algs import CEM, MPPI, PredictiveSampling, Evosax, xNES, CMAES, SVES, GaussianSmoothing
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.cart_pole import CartPole
from hydrax.tasks.cube import CubeRotation
from hydrax.tasks.walker import Walker
from hydrax.simulation.traj_opt import traj_opt_helper

"""
Run an interactive simulation of a cart-pole swingup
"""

# Define the task (cost and dynamics)

task = CartPole()

# task = CubeRotation()
# task = Walker()

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

parser = argparse.ArgumentParser(
    description="Run TO of the cart pole task."
)

parser.add_argument('-v', action='store_true', help="visualization")
arg = parser.parse_args()



# ctrl = CEM(
#     task,
#     num_samples=1024,
#     num_elites=4,
#     sigma_start=0.5,
#     sigma_min=0.5,
#     plan_horizon=1.0,
#     spline_type="zero",
#     num_knots=100,
# )

# ctrl = Evosax(
#     task,
#     evosax.Open_ES,
#     sigma = 0.3,
#     use_antithetic_sampling = False,
#     num_samples=1024,
#     plan_horizon=1.0,
#     spline_type="zero",
#     num_knots=100,
#     iterations = 1
# )

# ctrl = MPPI(
#     task,
#     noise_level = 0.3,
#     temperature = 0.01, 
#     num_samples=1024,
#     plan_horizon=0.5,
#     spline_type="cubic",
#     num_knots=10,
#     iterations = 1
# )

ctrl = GaussianSmoothing(
    task,
    sigma = 0.3,
    temperature = 0.1, 
    num_samples=1024,
    plan_horizon=1.0,
    spline_type="zero",
    num_knots=100,
    iterations = 1
)

# ctrl = xNES(
#     task,
#     num_samples=1024,
#     temperature=0.1,
#     sigma = 0.3,
#     plan_horizon=1.0,
#     spline_type="cubic",
#     num_knots=100,
#     iterations = 1
# )

# ctrl = CMAES(
#     task,
#     num_samples=1024,
#     temperature=0.1,
#     sigma = 0.3,
#     plan_horizon=1.0,
#     spline_type="zero",
#     num_knots=100,
#     iterations = 1
# )

# ctrl = SVES(
#     task,
#     num_poplutions= 1,
#     population_size=1024,
#     use_antithetic_sampling = False,
#     temperature=0.1,
#     sigma = 0.3,
#     plan_horizon=1.0,
#     spline_type="zero",
#     num_knots=100,
#     iterations = 1
# )

# Define the model used for simulation
mj_model = task.mj_model
mj_model.opt.timestep = 0.01
mj_model.opt.iterations = 5
mj_data = mujoco.MjData(mj_model)
# Generate a random quaternion

to = traj_opt_helper(ctrl, mj_model, mj_data)


if arg.v == True:
    to.load_policy()
    to.visualize_all()
else:
    to.optimize(max_iteration=100)