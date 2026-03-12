# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask

class Acrobot(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 500

        self.cfg["env"]["numObservations"] = 4
        self.cfg["env"]["numActions"] = 1

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/acrobot.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        acrobot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(acrobot_asset)

        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 2.0
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        self.acrobot_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            acrobot_handle = self.gym.create_actor(env_ptr, acrobot_asset, pose, "acrobot", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, acrobot_handle)
            # Joint 0 (Shoulder) is Actuated (EFFORT)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            # Joint 1 (Elbow) is Passive (NONE)
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, acrobot_handle, dof_props)

            self.envs.append(env_ptr)
            self.acrobot_handles.append(acrobot_handle)

    def compute_reward(self):
        # retrieve environment observations from buffer
        shoulder_pos = self.obs_buf[:, 0]
        shoulder_vel = self.obs_buf[:, 1]
        elbow_pos = self.obs_buf[:, 2]
        elbow_vel = self.obs_buf[:, 3]

        self.rew_buf[:], self.reset_buf[:] = compute_acrobot_reward(
            shoulder_pos, shoulder_vel, elbow_pos, elbow_vel,
            self.reset_dist, self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)

        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze() # Shoulder Pos
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze() # Shoulder Vel
        self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze() # Elbow Pos
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze() # Elbow Vel

        return self.obs_buf

    def reset_idx(self, env_ids):
        # Randomize initial state slightly around upright (0.0)
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # Apply actions to Shoulder Joint (Index 0)
        # Actions are 1D (num_envs, 1)
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        # We only actuate joint 0. 
        # Layout in tensor: [env0_j0, env0_j1, env1_j0, env1_j1, ...]
        # We want indices 0, 2, 4, ... -> i * num_dof + 0
        actions_tensor[0::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_acrobot_reward(shoulder_pos, shoulder_vel, elbow_pos, elbow_vel,
                            reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # Goal: Keep both poles upright (0.0)
    # Penalize deviation from 0.0 for both joints
    # Penalize velocities to encourage stability
    
    # Simple L2 penalty reward
    # reward = 1.0 - shoulder_pos * shoulder_pos - elbow_pos * elbow_pos - 0.01 * torch.abs(shoulder_vel) - 0.01 * torch.abs(elbow_vel)
    
    # 优化后的奖励函数
    # 1. 存活奖励：只要不倒下，给 1.0 分
    # 2. 姿态奖励：
    #    - shoulder_pos 越接近 0 越好 (权重 2.0)
    #    - elbow_pos 越接近 0 越好 (权重 2.0)
    #    - 使用 exp(-error^2) 形式，使得奖励在目标附近更敏感，远离时衰减平滑
    # 3. 速度惩罚：
    #    - 稍微加大一点速度惩罚，防止疯狂摆动 (权重 0.1)

    rew_pos_shoulder = torch.exp(-2.0 * shoulder_pos * shoulder_pos)
    rew_pos_elbow = torch.exp(-2.0 * elbow_pos * elbow_pos)
    rew_vel_shoulder = -0.1 * torch.abs(shoulder_vel)
    rew_vel_elbow = -0.1 * torch.abs(elbow_vel)

    reward = 1.0 + rew_pos_shoulder + rew_pos_elbow + rew_vel_shoulder + rew_vel_elbow

    # Reset conditions
    # If shoulder moves too far (e.g. > reset_dist)
    # If elbow moves too far (e.g. > reset_dist)
    # reset_dist is typically around 3.0 (approx pi)
    
    reward = torch.where(torch.abs(shoulder_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(elbow_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)

    reset = torch.where(torch.abs(shoulder_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(torch.abs(elbow_pos) > reset_dist, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, reset
