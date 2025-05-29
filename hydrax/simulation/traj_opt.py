import time
from typing import Sequence

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx
import copy
from hydrax.alg_base import Trajectory, SamplingBasedController
import joblib
import tqdm
from functools import partial
import os


class traj_opt_helper:
    def __init__(
        self,
        controller: SamplingBasedController,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
    ):
        # logging
        print(
            f"Trajectory Optimization with {controller.num_knots} steps "
            f"over a {controller.ctrl_steps * controller.task.dt} "
            f"second horizon."
        )

        self.warm_up = False
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.controller = controller
        mjx_data = mjx.put_data(self.mj_model, self.mj_data)
        mjx_data = mjx_data.replace(mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat)
        self.mjx_data = mjx_data
        self.viewer = None

        # initialize the controller
        jit_optimize = jax.jit(partial(controller.optimize), donate_argnums=(1,))
        self.jit_optimize = jit_optimize

    def __warm_up(self):
        if self.warm_up:
            return
        # warm-up the controller
        print("Jitting the controller...")
        st = time.time()
        policy_params = self.controller.init_params()
        policy_params, _ = self.jit_optimize(self.mjx_data, policy_params)
        policy_params, _ = self.jit_optimize(self.mjx_data, policy_params)
        print(f"Time to jit: {time.time() - st:.3f} seconds")

        self.warm_up = True

    def load_policy(self):
        self.policy_params = joblib.load("policy_params_latest.pkl")
        self.cost_list = joblib.load("costs_latest.pkl")

    def trails( self,
        max_iteration: int = 100,
        num_trails: int = 6) -> None:

        self.__warm_up()

        controller_name = self.controller.__class__.__name__
        task_name = self.controller.task.__class__.__name__
        path = os.path.join("data", task_name)
        try:
            os.makedirs(path, exist_ok=True)
            print(f"path created: {path}")
        except Exception as e:
            print(f'failed to crate path: {e}')
        cost_list_list = []
        seed_list = list(np.arange(num_trails))
        for seed in seed_list:
            cost_list = self.optimize(max_iteration, seed=seed)
            cost_list_list.append(cost_list)
        
        
        cost_array = np.array(cost_list_list)
        cost_array = cost_array.mean(axis = 0)

        try:
            joblib.dump(cost_array, path + "/" + controller_name + "_costs_trails_average.pkl")
            print("Results saved")
        except Exception as e:
            print(f"Failed to save results: {e}")

    def optimize(
        self,
        max_iteration: int = 100,
        seed: int = 1
    ) -> list:

        policy_params = self.controller.init_params(seed=seed)
        cost_list = []

        for i in tqdm.tqdm(range(max_iteration)):
            policy_params, rollouts = self.jit_optimize(self.mjx_data, policy_params)
            trajectory_cost = jnp.sum(rollouts.costs[-1, :], axis=-1) # Take the current trajectory costs            
            cost_list.append(trajectory_cost) # Append the cost of the current control trajectory to the list

        print("Optimization done.")

        return cost_list
    
    def optimize_save_results(
        self,
        max_iteration: int = 100,
        seed: int = 1
    ) -> list:

        self.__warm_up()
        policy_params = self.controller.init_params(seed=seed)
        controller_name = self.controller.__class__.__name__
        task_name = self.controller.task.__class__.__name__
        path = os.path.join("data", task_name)

        os.makedirs(path, exist_ok=True)

        cost_list = []

        for i in tqdm.tqdm(range(max_iteration)):
            policy_params, rollouts = self.jit_optimize(self.mjx_data, policy_params)
            trajectory_cost = jnp.sum(rollouts.costs[-1, :], axis=-1) # Take the current trajectory costs            
            cost_list.append(trajectory_cost) # Append the cost of the current control trajectory to the list

        print("Optimization done.")

        self.policy_params = policy_params
        self.rollouts = rollouts
        try:
            joblib.dump(policy_params, path + "/" + controller_name + "_policy_params.pkl")
            joblib.dump(rollouts,  path + "/" + controller_name + "_rollouts.pkl")
            joblib.dump(cost_list, path + "/" + controller_name + "_costs.pkl")
            print("Results saved")
        except Exception as e:
            print(f"Failed to save results: {e}")

        return cost_list

    # This function will not work (the version is too old)
    # def visualize_rollout(self, idx: int, loop: bool = True):
    #     """
    #     visualize only single rollout
    #     """
    #     self.__create_temporary_viewer()
    #     i = 0
    #     while self.viewer.is_running():
    #         start_time = time.perf_counter()

    #         # Step the simulation
    #         t = i * self.mj_model.opt.timestep
    #         u = self.controller.get_action(idx, self.policy_params, t)
    #         self.tmp_mj_data.ctrl[:] = np.array(u)
    #         mujoco.mj_step(self.mj_model, self.tmp_mj_data)
    #         self.viewer.sync()

    #         # Try to run in roughly realtime
    #         elapsed_time = time.perf_counter() - start_time
    #         if elapsed_time < self.mj_model.opt.timestep:
    #             time.sleep(self.mj_model.opt.timestep - elapsed_time)
    #         i += 1
    #         if i == self.controller.task.planning_horizon * self.controller.task.sim_steps_per_control_step:
    #             self.__reset_tmp_data()
    #             if loop:
    #                 i = 0
    #             else:
    #                 return

    def __create_temporary_viewer(self):
        if self.viewer is None:
            self.tmp_mj_data = copy.copy(self.mj_data)
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.tmp_mj_data)

    def __reset_tmp_data(self):
        self.tmp_mj_data.qpos[:] = self.mj_data.qpos
        self.tmp_mj_data.qvel[:] = self.mj_data.qvel

    def visualize_all(self):
        with tqdm.tqdm(total=self.controller.num_populations, desc="Visualizing rollouts", ncols=0) as pbar:
            while True:
                for i in range(self.controller.num_populations):
                    self.visualize_rollout(i, loop=False)
                    pbar.update(1)
                pbar.reset()
            # for i in tqdm.tqdm(range(self.controller.num_populations), desc="Visualizing rollouts"):
            #     self.visualize_rollout(i, loop=False)
