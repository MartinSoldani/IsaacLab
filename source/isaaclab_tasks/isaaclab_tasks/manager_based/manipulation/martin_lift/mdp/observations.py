# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Define global variable
is_testing = False  # Default to training mode

def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    red_cube_cfg: SceneEntityCfg = SceneEntityCfg("red_cube")
) -> torch.Tensor:
    """Returns the cube position differently for training vs. testing."""

    if is_testing:  #During testing, use externally detected coordinates
        target_pos_w = env.command_manager.get_command("object_pose")  # Injected coordinates
    else:  # During training, use the real cube's position
        red_cube: RigidObject = env.scene[red_cube_cfg.name]
        target_pos_w = red_cube.data.root_pos_w[:, :3]  

    # Convert from world coordinates to the robot's root frame
    robot: RigidObject = env.scene[robot_cfg.name]
    target_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], target_pos_w
    )

    return target_pos_b