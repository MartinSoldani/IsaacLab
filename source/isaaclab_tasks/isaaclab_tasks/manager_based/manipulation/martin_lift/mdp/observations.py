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

# Define global variable #TODO CHANGE BACK TO FALSE
is_testing = True  # Default to training mode

def get_target_object(env) -> str:
    """Returns the dynamically chosen target object (red_cube or green_cube)."""
    if hasattr(env.scene, "object") and isinstance(env.scene.object, str):
        target_object = env.scene.object.lower().replace(" ", "_")
        if target_object in ["red_cube", "green_cube"]:
            return target_object
    print("[ERROR] `env.scene.object` is not set or invalid! Defaulting to `red_cube`.")
    return "red_cube"  # Default cube if invalid


def object_position_in_robot_root_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the target object's position in the robot's root frame, using target_coords or real position."""
    
    # ✅ Get the correct target cube dynamically
    target_object = get_target_object(env)

    # ✅ Ensure `target_object` exists before using it
    if not hasattr(env.scene, target_object):
        print(f"[ERROR] `{target_object}` not found in scene! Defaulting to `red_cube`.")
        target_object = "red_cube"

    if is_testing:
        if hasattr(env, "target_cube_coords"):
            target_pos_w = env.target_cube_coords
            print(f"[DEBUG] Testing Mode - Injected Coordinates for {target_object}: {target_pos_w}")
        else:
            print("[WARNING] No target_cube_coords found in testing mode. Using default.")
            target_pos_w = torch.tensor([[0.0, 0.0, 0.05]], device=env.sim.device)  # ✅ Add batch dimension
    else:
        # ✅ Use actual object position in training mode
        object: RigidObject = getattr(env.scene, target_object)  # ✅ Use `getattr()` instead of `env.scene[target_object]`
        target_pos_w = object.data.root_pos_w[:, :3]  # ✅ Ensure shape is (num_envs, 3)
        print(f"[DEBUG] Training Mode - Real Position for {target_object}: {target_pos_w}")

    # ✅ Ensure target_pos_w has the same shape as robot.data.root_state_w[:, :3]
    if target_pos_w.dim() == 1:
        target_pos_w = target_pos_w.unsqueeze(0)  # ✅ Expand dims if needed

    # Compute position relative to the robot’s root frame
    robot: RigidObject = env.scene["robot"]  # ✅ Access robot correctly
    target_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],  # (num_envs, 3)
        robot.data.root_state_w[:, 3:7],  # (num_envs, 4) - quaternion
        target_pos_w  # (num_envs, 3) ✅ Ensure shape matches
    )
    
    return target_pos_b