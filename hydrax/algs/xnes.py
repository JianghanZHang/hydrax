from typing import Any, Literal, Tuple

from evosax.algorithms.distribution_based.xnes import xNES as EvoXNES
import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task

from evosax.types import Fitness, Params, Population, State

import optax
     
# Generic types for evosax
EvoParams = Any
EvoState = Any

@jax.jit
def print_weights(w):
    jax.debug.print("🔥 weights = {}", w)

@jax.jit
def print_nan(w):
    jax.debug.print("🔥 any NaN? = {}", jnp.isnan(w).any())

@dataclass
class EvosaxParams(SamplingParams):
    """Policy parameters for evosax optimizers.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
        opt_state: The state of the evosax optimizer (covariance, etc.).
    """

    opt_state: EvoState


class xNES(SamplingBasedController):
    """
    xNES optimizer from Evosax
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        es_params: EvoParams = None,
        temperature: float = 0.1,
        sigma:float = 0.3,
        lr_sigma: float = 0.1, 
        lr_B: float = 0.1,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        **kwargs,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            optimizer: The evosax optimizer to use.
            num_samples: The number of control tapes to sample.
            es_params: The parameters for the evosax optimizer.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
            **kwargs: Additional keyword arguments for the optimizer.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
            
        )

        # Using the softmax utility as in MPPI
        def lse_fitness_shaping_fn(
            population: Population, fitness: jax.Array, state: State, params: Params
            ) -> Fitness:
            
            # fitness = fitness / jnp.std(fitness)

            fitness_shaped = jax.nn.softmax(-fitness / temperature, axis=0)

            return fitness_shaped

        self.strategy = EvoXNES(
            population_size=num_samples,
            solution=jnp.zeros(task.model.nu * self.num_knots),
            fitness_shaping_fn = lse_fitness_shaping_fn,
            optimizer = optax.sgd(learning_rate=1),  # Set learing rate = 1 here (the default was 1e-3)
            **kwargs,
        )

        if es_params is None:
            es_params = self.strategy.default_params.replace(std_init=sigma, lr_B=lr_B, lr_std_init=lr_sigma)
            # es_params = self.strategy.default_params.replace(std_init=sigma)

        self.es_params = es_params

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> EvosaxParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        rng, init_rng = jax.random.split(_params.rng)

        # jax.debug.print("Params:{}", self.es_params)
        opt_state = self.strategy.init(key = init_rng, mean=jnp.zeros(self.task.model.nu * self.num_knots), params = self.es_params)
        
        # Evosax's xNES implementation has an error when initializing B (it's det(B) is not 1), so we enforce it here!
        opt_state = opt_state.replace(B = jnp.eye(self.strategy.num_dims))

        # jax.debug.print("State:{}", opt_state)
        return EvosaxParams(
            tk=_params.tk, mean=_params.mean, opt_state=opt_state, rng=rng
        )

    def sample_knots(
        self, params: EvosaxParams
    ) -> Tuple[jax.Array, EvosaxParams]:
        """Sample control sequences from the proposal distribution."""
        rng, sample_rng = jax.random.split(params.rng)
        x, opt_state = self.strategy.ask(
            sample_rng, params.opt_state, self.es_params
        )

        # evosax works with vectors of decision variables, so we reshape U to
        # [batch_size, num_knots, nu].
        controls = jnp.reshape(
            x,
            (
                self.strategy.population_size,
                self.num_knots,
                self.task.model.nu,
            ),
        )

        controls = jnp.concatenate([controls, params.mean[None, :, :]], axis=0) # Put the current controls for recording the current cost

        return controls, params.replace(opt_state=opt_state, rng=rng)



    def update_params(
        self, params: EvosaxParams, rollouts: Trajectory
    ) -> EvosaxParams:
        """Update the policy parameters based on the rollouts."""
        costs = jnp.sum(rollouts.costs, axis=1)[:-1]  # sum over time steps (remove the current control)
        knots = rollouts.knots[:-1, :, :]

        x = jnp.reshape(knots, (self.strategy.population_size, -1))

        rng, update_rng = jax.random.split(params.rng)


        opt_state, _ = self.strategy.tell(
            key=update_rng, population=x, fitness=costs, state=params.opt_state, params=self.es_params
        )

        best_idx = jnp.argmin(costs)
        # best_knots = rollouts.knots[best_idx]

        # By default, opt_state stores the best member ever, rather than the
        # best member from the current generation. We want to just use the best
        # member from this generation, since the cost landscape is constantly
        # changing.


        # #################################### Debug #########################################
        fitness_shaped = jax.nn.softmax(-costs / 0.1, axis=0)
        
        fitness_has_nan = jnp.isnan(fitness_shaped).any()
        cost_has_nan = jnp.isnan(costs).any()
        jax.debug.print(" \n\n\n🔥fitness = \n{} \n 🔥any NaN? = {} \n 🔥weights (softmax) =\n{} \n 🔥any NaN? = {} \n mean = \n {}", \
                        costs, cost_has_nan, fitness_shaped, fitness_has_nan, opt_state.mean)

        # ####################################################################################
        
        mean = jnp.reshape(opt_state.mean,
            (
            self.num_knots,
            self.task.model.nu,
            )
        )

        # mean = jnp.clip(
        #     mean, self.task.u_min, self.task.u_max
        #     )  # (num_knots, nu)
        
        opt_mean = jnp.reshape(mean,
                            (
                            self.num_knots * self.task.model.nu
                            )
                        ) # (num_knots * nu) flat the mean for Evosax
    
        opt_state = opt_state.replace(
            best_solution=x[best_idx], best_fitness=costs[best_idx], mean = opt_mean
        )

        # jax.debug.print("x: {}", x)
        # jax.debug.print("mean: \n{}", opt_state.mean)

        
        return params.replace(mean=mean, opt_state=opt_state, rng=rng)
    

