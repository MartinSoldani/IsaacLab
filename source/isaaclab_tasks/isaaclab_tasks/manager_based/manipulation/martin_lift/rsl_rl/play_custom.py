# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import time
import torch
import json
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)

parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--object", type=str, default=None, help="Specify the object to pick up (e.g., 'red' or 'green')")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import numpy as np
import cv2

from isaaclab_tasks.manager_based.manipulation.martin_lift.mdp.observations import is_testing

is_testing = True


import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def main():
    """Play with RSL-RL agent and inject object coordinates."""

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    


    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)



    # Take initial photo and process with OWL-ViT
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "images", "initial_training_view.jpg")
    save_initial_photo(env, image_path)
    cube_coords = process_with_owlvit(image_path)
    
    # Save coordinates to a file for use in the reward function
    coords_path = os.path.join(base_dir, "coordinates.json")
    with open(coords_path, "w") as f:
        json.dump(cube_coords, f)
    print(f"[INFO] Cube coordinates saved to {coords_path}")

    # Load coordinates into the underlying environment
    underlying_env = env.unwrapped  # Access the unwrapped Isaac Lab environment
    underlying_env.target_cube_coords = None
    if "red cube" in cube_coords:
        underlying_env.target_cube_coords = torch.tensor(cube_coords["red cube"]["center"], device=underlying_env.sim.device)
        print(f"[INFO] Target cube (red) coordinates: {underlying_env.target_cube_coords}")
    elif "green cube" in cube_coords:
        underlying_env.target_cube_coords = torch.tensor(cube_coords["green cube"]["center"], device=underlying_env.sim.device)
        print(f"[INFO] Target cube (green) coordinates: {underlying_env.target_cube_coords}")
    else:
        print("[WARNING] No cubes detected in the initial image!")


    # TRANSFORM DETECTED IMAGE COORDS INTO REAL WORLD CORDS
    # Path to detected image coordinates

    # Fixed table height
    table_height = 0.05  # Cube Z offset for being placed on the table with respect to the robot arm, also placed on the table

    # Compute real-world positions dynamically
    real_world_positions = image_to_world_fixed_height(env, coords_path, table_height)

    # Save real-world positions to a JSON file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    real_coords_path = os.path.join(base_dir, "real_coordinates.json")
    os.makedirs(os.path.dirname(real_coords_path) or ".", exist_ok=True)
    with open(real_coords_path, "w") as f:
        json.dump(real_world_positions, f, indent=4)  # Use indent=4 for readable JSON
    print(f"[INFO] Real-world cube coordinates saved to {real_coords_path}")


    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Inject object coordinates before reset
    target_coords = get_real_coordinates(args_cli.object)
    print(f"[INFO] Setting object position to {target_coords.cpu().numpy()}")

    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[:, :3]  # Get robot's base position
    robot_rot = env.unwrapped.scene["robot"].data.root_state_w[:, 3:7]  # Get robot's rotation
    print(f"[DEBUG] Robot Base Position: {robot_pos}, Rotation: {robot_rot}")
    # OVERRIDE CUBE POSITION WITH THE ONE DETECTED IN IMAGE!!!
    env.unwrapped.command_manager._commands["object_pose"] = target_coords  # Inject position before reset
    print(env.unwrapped.command_manager._commands["object_pose"])

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.physics_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()



def save_initial_photo(env, file_path):
    """Captures and saves a single RGB image from the overhead camera at the start of training."""
    # Unwrap the RslRlVecEnvWrapper to access the underlying Isaac Lab environment
    underlying_env = env.unwrapped
    if not hasattr(underlying_env, "scene"):
        print("[ERROR] Underlying environment has no scene attribute!")
        return

    camera: Camera = underlying_env.scene["overhead_camera"]
    if camera is None:
        print("[ERROR] Overhead camera not found in scene! Scene contents:", underlying_env.scene.keys())
        return

    # Update camera with simulation timestep
    camera.update(dt=underlying_env.sim.get_physics_dt())

    # Access RGB data directly from output dict
    rgb_data = camera.data.output.get("rgb")
    if rgb_data is None or rgb_data.shape[0] == 0:
        print("[ERROR] No RGB data available from camera!")
        return

    # Select first environment's image, convert to numpy
    rgb_numpy = rgb_data[0].cpu().numpy()  # [height, width, channels]

    # Normalize to 0-255 if in 0-1 range
    if rgb_numpy.max() <= 1.0:
        rgb_numpy = (rgb_numpy * 255).astype(np.uint8)
    else:
        rgb_numpy = rgb_numpy.astype(np.uint8)

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    # Save image
    img = Image.fromarray(rgb_numpy)
    img.save(file_path, quality=95)
    print(f"[INFO] Initial training image saved at {file_path}")

def process_with_owlvit(image_path):
    """Process the image with OWL-ViT to detect cube coordinates."""
    # Load the processor and model
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)  # Convert to NumPy for visualization
    target_labels = ["red cube", "blue cube"]
    inputs = processor(images=image, text=target_labels, return_tensors="pt")

    # Perform detection
    outputs = model(**inputs)

    # Post-process to get bounding boxes and scores
    target_sizes = torch.Tensor([image.size[::-1]])  # [height, width]
    results = processor.post_process_object_detection(outputs, threshold=0.005, target_sizes=target_sizes)[0]

    object_locations = {}
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > 0.005:  # Confidence threshold
            label_name = target_labels[label.item()]
            box_coords = box.tolist()  # [x_min, y_min, x_max, y_max]
            object_locations[label_name] = {
                "bbox": box_coords,
                "center": [
                    (box_coords[0] + box_coords[2]) / 2,  # x_center
                    (box_coords[1] + box_coords[3]) / 2   # y_center
                ],
                "confidence": score.item()
            }

    # ✅ Save image with bounding boxes
    base_dir = os.path.dirname(os.path.abspath(__file__))
    boxed_image_path = os.path.join(base_dir, "images", "initial_boxed_cubes.jpg")
    save_boxed_image(image_np, object_locations, boxed_image_path)

    return object_locations

def image_to_world_fixed_height(env, image_coords_path, table_height):
    """
    Convert image pixel coordinates to real-world 3D positions assuming a fixed table height.
    
    Parameters:
    - env: The wrapped IsaacLab simulation environment.
    - image_coords_path (str): Path to JSON file with image coordinates.
    - table_height (float): Fixed height of the table where cubes are placed.

    Returns:
    - real_world_positions (dict): 3D positions of cubes in IsaacLab.
    """

    # Unwrap the RslRlVecEnvWrapper to access the underlying Isaac Lab environment
    underlying_env = env.unwrapped
    if not hasattr(underlying_env, "scene"):
        print("[ERROR] Underlying environment has no scene attribute!")
        return {}

    # Retrieve the overhead camera from the scene
    camera: Camera = underlying_env.scene["overhead_camera"]
    if camera is None:
        print("[ERROR] Overhead camera not found in scene!")
        return {}

    # Get camera intrinsics dynamically
    intrinsic_matrix = camera.data.intrinsic_matrices.cpu().numpy()[0]  # Extract 3x3 intrinsics matrix
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]  # Focal lengths
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]  # Principal point

    # Get camera extrinsics (position & rotation)
    camera_pos = camera.data.pos_w.cpu().numpy()[0]  # Camera world position (x, y, z)
    camera_quat = camera.data.quat_w_ros.cpu().numpy()[0]  # Camera rotation (x, y, z, w)

    # Load detected image coordinates from JSON file
    with open(image_coords_path, "r") as f:
        image_coords = json.load(f)

    # Set fixed Z coordinate (table height)
    Z_fixed = table_height

    # Compute real-world positions
    real_world_positions = {}

    for cube_name, data in image_coords.items():
        # Extract image pixel center
        x_img, y_img = data["center"]

        # Normalize pixel coordinates
        x_norm = (x_img - cx) / fx
        y_norm = (y_img - cy) / fy

        # Compute real-world X, Y coordinates (assuming Z = table height)
        X_world = camera_pos[0] + x_norm * (camera_pos[2] - Z_fixed)
        Y_world = camera_pos[1] + y_norm * (camera_pos[2] - Z_fixed)

        # Store the computed world coordinates
        real_world_positions[cube_name] = {
            "position": [X_world, Y_world, Z_fixed],  # Fixed Z-coordinate
            "confidence": data["confidence"],
        }

    return real_world_positions


def save_boxed_image(image_np, object_locations, save_path):
    """Draws bounding boxes on the image and saves it."""
    image_with_boxes = image_np.copy()
    
    for label, info in object_locations.items():
        bbox = info["bbox"]
        confidence = info["confidence"]

        # Convert to integer values
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Draw rectangle (bounding box)
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

        # Label text
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(
            image_with_boxes, label_text, (x_min, y_min - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Save the modified image
    cv2.imwrite(save_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Boxed cubes image saved at {save_path}")


def get_real_coordinates(object_name):
    """Retrieve real-world coordinates for the specified object from real_coordinates.json."""
    if not object_name:
        raise ValueError("No object specified! Use --object to specify 'red' or 'green'.")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    coords_path = os.path.join(base_dir, "real_coordinates.json")
    
    if not os.path.exists(coords_path):
        raise FileNotFoundError(f"Real-world coordinates file not found at {coords_path}")
    
    with open(coords_path, "r") as f:
        real_coords = json.load(f)
    
    if object_name.lower() == "red cube" and "red cube" in real_coords:
        return torch.tensor(real_coords["red cube"]["position"], device="cuda" if torch.cuda.is_available() else "cpu")
    elif object_name.lower() == "green cube" and "blue cube" in real_coords:
        return torch.tensor(real_coords["blue cube"]["position"], device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError(f"Unknown or undetected object: {object_name}")
    
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()