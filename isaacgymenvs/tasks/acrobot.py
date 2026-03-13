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
from isaacgym.torch_utils import *
from .base.vec_task import VecTask

class Acrobot(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 500

        self.cfg["env"]["numObservations"] = 6
        self.cfg["env"]["numActions"] = 1

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # Rigid body state for Tip Height Reward
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        
        # Store actions for reward penalty
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        
        # Get rigid body indices after creating envs (and loading asset)
        # Note: asset was loaded in _create_envs, but handle is needed.
        # However, body indices are per asset.
        # But wait, gym.get_actor_rigid_body_index is what we need for the global tensor if we use indices.
        # But we are using a view of (num_envs, num_bodies, 13). So we need the index WITHIN the actor.
        # The rigid body tensor from acquire_rigid_body_state_tensor is flat: [env0_b0, env0_b1, ..., env1_b0, ...]
        # So we just need the local index of 'fixed_pole' and 'flex_1_pole'.
        
        # We can get this from the asset.
        # But we need to access the asset which is created in _create_envs.
        # Since _create_envs is called above, we can assume self.acrobot_asset is available if we store it.
        # Let's modify _create_envs to store it or get it here.
        # Actually, simpler: just get it from the first env's actor.
        
        # Better yet, let's modify _create_envs to save the body indices.


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
        
        # Get rigid body indices for reward calculation
        rigid_body_dict = self.gym.get_asset_rigid_body_dict(acrobot_asset)
        self.shoulder_body_idx = rigid_body_dict["fixed_pole"]
        self.elbow_body_idx = rigid_body_dict["flex_1_pole"]
        
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
            # Correct Configuration per User:
            # Joint 0 (Shoulder) is Actuated (EFFORT) - "带电机"
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            # Joint 1 (Elbow) is Passive (NONE) - "轴承"
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            
            # -----------------------------------------------------------
            # Joint Limits Configuration
            # -----------------------------------------------------------
            dof_props['hasLimits'][:] = True
            
            # Joint 0 (Shoulder): Limited to +/- 2*pi (One full rotation)
            # Reason: Motor wires limitation
            dof_props['lower'][0] = -2.0 * np.pi
            dof_props['upper'][0] = 2.0 * np.pi
            
            # Joint 1 (Elbow): Unlimited (Continuous Bearing)
            # Reason: Passive bearing, no wires
            dof_props['hasLimits'][1] = False
            dof_props['lower'][1] = -1.0e8 # Effectively infinite
            dof_props['upper'][1] = 1.0e8
            
            # Respect URDF dynamics (damping/friction)
            # But ensure stiffness is zero for proper passive/effort behavior
            dof_props['stiffness'][:] = 0.0 
            
            # Ensure Elbow (Joint 1) is truly frictionless/passive as requested
            dof_props['damping'][1] = 0.0001
            dof_props['friction'][1] = 0.0
            
            self.gym.set_actor_dof_properties(env_ptr, acrobot_handle, dof_props)

            self.envs.append(env_ptr)
            self.acrobot_handles.append(acrobot_handle)

    def compute_reward(self):
        # Refresh Rigid Body State for Tip Height
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        shoulder_pos = self.dof_pos[:, 0]
        elbow_pos = self.dof_pos[:, 1]
        shoulder_vel = self.dof_vel[:, 0]
        elbow_vel = self.dof_vel[:, 1]
        
        # Calculate Tip Height using Rigid Body States
        # rb_states: [num_envs, num_bodies, 13]
        # 13: pos(3), quat(4), lin_vel(3), ang_vel(3)
        rb_states = self.rigid_body_states.view(self.num_envs, -1, 13)
        
        # Shoulder Position (Z-height of fixed_pole origin)
        # Note: fixed_pole origin is at the shoulder joint.
        shoulder_height = rb_states[:, self.shoulder_body_idx, 2]
        
        # Elbow Position (Z-height of flex_1_pole origin) and Orientation
        elbow_p = rb_states[:, self.elbow_body_idx, 0:3]
        elbow_q = rb_states[:, self.elbow_body_idx, 3:7]
        
        # Tip Offset in Local Frame (Assume length 0.16m along Y based on URDF and COM)
        # We need to verify the local axis. 
        # URDF: flex_1_pole COM is at y=0.08. Joint is at 0. So length extends along Y.
        local_tip = torch.tensor([0.0, 0.16, 0.0], device=self.device).repeat(self.num_envs, 1)
        
        # Rotate local offset to global frame
        tip_offset = quat_apply(elbow_q, local_tip)
        tip_p = elbow_p + tip_offset
        tip_height = tip_p[:, 2]

        self.rew_buf[:], self.reset_buf[:] = compute_acrobot_reward(
            shoulder_pos, shoulder_vel, elbow_pos, elbow_vel,
            tip_height, shoulder_height, self.actions,
            self.reset_dist, self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)

        # ----------------------------------------------------------------------
        # Improved Observation Space (Sin/Cos Representation)
        # ----------------------------------------------------------------------
        # Raw angles are periodic [-pi, pi], which causes discontinuity issues for neural networks.
        # We transform them into continuous sin/cos components.
        #
        # Obs: [sin(theta1), cos(theta1), sin(theta2), cos(theta2), vel1, vel2]
        # Total: 6 dimensions
        
        shoulder_pos = self.dof_pos[:, 0]
        elbow_pos = self.dof_pos[:, 1]
        shoulder_vel = self.dof_vel[:, 0]
        elbow_vel = self.dof_vel[:, 1]

        self.obs_buf[:, 0] = torch.sin(shoulder_pos)
        self.obs_buf[:, 1] = torch.cos(shoulder_pos)
        self.obs_buf[:, 2] = torch.sin(elbow_pos)
        self.obs_buf[:, 3] = torch.cos(elbow_pos)
        self.obs_buf[:, 4] = shoulder_vel
        self.obs_buf[:, 5] = elbow_vel

        return self.obs_buf

    def reset_idx(self, env_ids):
        # Randomize initial state uniformly between -pi and pi
        positions = 2.0 * np.pi * torch.rand((len(env_ids), self.num_dof), device=self.device) - np.pi
        velocities = 2.0 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

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
        # Save actions for reward calculation
        self.actions = actions.clone()
        
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
def normalize_angle(x):
    # type: (Tensor) -> Tensor
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def compute_acrobot_reward(shoulder_pos, shoulder_vel, elbow_pos, elbow_vel,
                           tip_height, shoulder_height, actions,
                           reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # ----------------------------------------------------------------------
    # New Tip Height Reward Strategy (2025-03-13)
    # ----------------------------------------------------------------------
    # Based on user feedback: "末端高度" is the most direct signal.
    
    # 1. Tip Height Reward (R_height)
    #    H_max = 0.32m (0.16 + 0.16)
    #    R = (H_tip - H_shoulder) / H_max
    #    Range: [-1, 1] usually (if H_tip < H_shoulder, it's negative)
    #    User formula: (H_tip - H_shoulder) / H_max
    h_max = 0.32
    # Ensure shoulder_height is broadcastable if needed, but it should be (num_envs,)
    rew_height = (tip_height - shoulder_height) / h_max
    
    # 2. Velocity Penalty (R_vel)
    #    - (0.5 * theta1_dot^2 + 1.0 * theta2_dot^2)
    rew_vel = -1.0 * (0.5 * shoulder_vel.pow(2) + 1.0 * elbow_vel.pow(2))
    
    # 3. Action Penalty (R_action)
    #    - u^2
    #    Actions are usually [-1, 1] before scaling.
    rew_action = -1.0 * actions.squeeze().pow(2)
    
    # 4. Hover Bonus (R_bonus)
    #    If R_height > 0.95 AND |vel_sum| < 1.0 -> +10.0
    #    Otherwise 0.0
    vel_sum = torch.abs(shoulder_vel) + torch.abs(elbow_vel)
    is_hover = (rew_height > 0.95) & (vel_sum < 1.0)
    rew_bonus = torch.where(is_hover, torch.ones_like(rew_height) * 10.0, torch.zeros_like(rew_height))
    
    # 5. Anti-Stall Penalty (防止摆烂)
    #    User feedback: "肩部上抬，肘部重合，抖动摆烂" -> Tip Height approx 0.
    #    If Tip is near Shoulder (abs(rew_height) < 0.3) AND velocity is low,
    #    it means the agent is stuck in a local optimum (folded state).
    #    We punish this heavily to force it to move out of this zone.
    #    Range: [-0.3, 0.3] -> covering the folded state (0.0).
    #    Threshold: vel_sum < 3.0 (relaxed to catch jittering too).
    is_stalled = (torch.abs(rew_height) < 0.3) & (vel_sum < 3.0)
    rew_stall = torch.where(is_stalled, torch.ones_like(rew_height) * -2.0, torch.zeros_like(rew_height))
    
    # Weights
    w1 = 10.0 # Increase Height weight to overpower penalties
    w2 = 0.1
    w3 = 0.05
    
    reward = w1 * rew_height + w2 * rew_vel + w3 * rew_action + rew_bonus + rew_stall
    
    # ----------------------------------------------------------------------
    # Reset Logic
    # ----------------------------------------------------------------------
    # 1. Reset if spinning too much (Propeller protection)
    limit = 6.28
    reset_spin = torch.where(torch.abs(shoulder_pos) > limit, torch.ones_like(reset_buf), reset_buf)
    
    # 2. Timeout Reset
    reset_time = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    
    reset = torch.where(reset_spin > 0, reset_spin, reset_time)
    
    return reward, reset

