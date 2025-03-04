# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_tasks.manager_based.manipulation.martin_lift.mdp.observations import get_target_object

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(env: ManagerBasedRLEnv, minimal_height: float) -> torch.Tensor:
    """Reward the agent for lifting the selected object above the minimal height."""
    
    # ✅ Get the target cube dynamically
    target_object = get_target_object(env)

    # ✅ Ensure object exists in scene
    if target_object not in env.scene:
        print(f"[ERROR] `{target_object}` not found in scene! Defaulting to `red_cube`.")
        target_object = "red_cube"

    object: RigidObject = env.scene[target_object]
    cube_height = object.data.root_pos_w[:, 2]

    reward = torch.where(cube_height > minimal_height, 1.0, 0.0)

    print(f"[DEBUG] Cube Height: {cube_height}, Minimal Height: {minimal_height}, Reward: {reward}")

    return reward


def object_ee_distance(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Reward the agent for reaching the selected object using tanh-kernel."""
    
    # ✅ Get the target cube dynamically
    target_object = get_target_object(env)

    # ✅ Ensure object exists in scene
    if target_object not in env.scene:
        print(f"[ERROR] `{target_object}` not found in scene! Defaulting to `red_cube`.")
        target_object = "red_cube"

    object: RigidObject = env.scene[target_object]
    ee_frame: FrameTransformer = env.scene["ee_frame"]

    # Extract positions
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]  # End-effector position

    # Compute distance
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    print(f"[DEBUG] Object-EE Distance (meters) - Mean: {object_ee_distance.mean().item():.4f}, "
          f"Min: {object_ee_distance.min().item():.4f}, Max: {object_ee_distance.max().item():.4f}")

    return 1 - torch.tanh(object_ee_distance / std)

# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("red_cube"),
# ) -> torch.Tensor:
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
#     distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
#     return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))