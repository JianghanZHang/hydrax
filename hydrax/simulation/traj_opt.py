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

    def optimize(
        self,
        max_iteration: int = 100,
    ) -> None:

        self.__warm_up()
        policy_params = self.controller.init_params()
        cost_list = []

        for i in tqdm.tqdm(range(max_iteration)):
            policy_params, rollouts = self.jit_optimize(self.mjx_data, policy_params)
            trajectory_cost = jnp.sum(rollouts.costs[-1, :], axis=-1) # Take the current trajectory costs

            # jax.debug.print("costs.shape{}", rollouts.costs.shape)

            # jax.debug.print("trajectory costs.shape{}", trajectory_costs.shape)
            
            cost_list.append(trajectory_cost) # Append the cost of the current control trajectory to the list

        print("Optimization done.")

        self.policy_params = policy_params
        self.rollouts = rollouts
        try:
            joblib.dump(policy_params, "policy_params_latest.pkl")
            joblib.dump(rollouts, "rollouts_latest.pkl")
            joblib.dump(cost_list, "costs_latest.pkl")
            print("Results saved")
        except Exception as e:
            print(f"Failed to save results: {e}")

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
