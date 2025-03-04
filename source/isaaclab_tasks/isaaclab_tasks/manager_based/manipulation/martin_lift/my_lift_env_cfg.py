# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import os

from isaaclab_tasks.manager_based.manipulation.martin_lift.mdp.observations import get_target_object  # ✅ Import function

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and cubes with different colors.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """
    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    red_cube: RigidObjectCfg | DeformableObjectCfg = MISSING
    green_cube: RigidObjectCfg | DeformableObjectCfg = MISSING
    
    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # sensors
    overhead_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverheadCamera",
        update_period=0.1,  # Keep update rate the same
        height=480,  # Keep resolution the same
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=35.0,
            focus_distance=200.0,
            horizontal_aperture=15,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 2.55),
            rot=(0, 0.0, 0.0, 0.1),
            convention="opengl",
        ),
    )


##
# MDP settings
##


# @configclass FOR GOAL TRACKING APPARENTLY
# class CommandsCfg:
#     """Command terms for the MDP."""

#     object_pose = mdp.UniformPoseCommandCfg(
#         asset_name="robot",
#         body_name=MISSING,  # will be set by agent env cfg
#         resampling_time_range=(5.0, 5.0),
#         debug_vis=True,
#         ranges=mdp.UniformPoseCommandCfg.Ranges(
#             pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
#         ),
#     )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # target_object_position = BRUH SKIP
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events, targeting only the red cube."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.1, 0.1), "z": (0.05, 0.05)},  # Adjusted for red cube on table
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("red_cube", body_names="Cube"),  # Target only red_cube
        },
    )

    def reset(self, env):
        print("[DEBUG] Starting episode reset...")
        self.reset_all.func(env)
        self.reset_object_position.func(env)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        coords_path = os.path.join(base_dir, "real_coordinates.json")
        target_object = env.scene.object.lower().replace(" ", "_") if hasattr(env.scene, "object") else "red_cube"
        if os.path.exists(coords_path):
            with open(coords_path, "r") as f:
                real_coords = json.load(f)
            if target_object in real_coords:
                env.target_cube_coords = torch.tensor(real_coords[target_object]["position"], device=env.sim.device)
                print(f"[INFO] Target cube ({target_object}) coordinates set: {env.target_cube_coords}")
            else:
                env.target_cube_coords = torch.tensor([0.0, 0.0, 0.05], device=env.sim.device)  # Default for red_cube, adjust for green_cube
                print(f"[WARNING] No {target_object} found in real coordinates. Using default.")
        else:
            env.target_cube_coords = torch.tensor([0.0, 0.0, 0.05], device=env.sim.device)  # Default for red_cube, adjust for green_cube
            print(f"[WARNING] Real coordinates file not found. Using default.")
        target_cube = env.scene[target_object]
        target_cube.set_root_state(pos=env.target_cube_coords, rot=torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.sim.device))
        print(f"[DEBUG] {target_object} position set to: {env.target_cube_coords}")
        return None

@configclass
class RewardsCfg:
    """Reward terms for the MDP, targeting the specified object dynamically via target_coords."""
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.2},
        weight=10.0
    )
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.05},
        weight=30.0
    )

# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""

#     reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=10.0)

#     lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    # object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=16.0,
    # )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=2.0,
    # )

    # # action penalty
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-1e-4,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("red_cube")}
    )


# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     # action_rate = CurrTerm(
#     #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
#     # )

#     # joint_vel = CurrTerm(
#     #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
#     # )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""
    target_cube_coords: torch.Tensor = torch.tensor([0.0, 0.0, 0.05])  # ✅ Initialize the attribute
    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        target_cube_coords: torch.Tensor = torch.tensor([0.0, 0.0, 0.05])
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
