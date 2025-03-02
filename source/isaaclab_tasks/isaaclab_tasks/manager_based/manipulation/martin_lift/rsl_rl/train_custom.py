# train_custom.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL, modified to capture an initial photo and process with OWL-ViT."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import json
import torch
from datetime import datetime
from PIL import Image
import numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.sensors import Camera  # Import Camera for type hinting
import cv2

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent, capturing an initial photo and processing with OWL-ViT."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    wrapped_env = RslRlVecEnvWrapper(env)  # Keep the wrapped environment for RSL-RL

    # Take initial photo and process with OWL-ViT
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "images", "initial_training_view.jpg")
    save_initial_photo(wrapped_env, image_path)
    cube_coords = process_with_owlvit(image_path)
    
    # Save coordinates to a file for use in the reward function
    coords_path = os.path.join(base_dir, "coordinates.json")
    with open(coords_path, "w") as f:
        json.dump(cube_coords, f)
    print(f"[INFO] Cube coordinates saved to {coords_path}")

    # Load coordinates into the underlying environment
    underlying_env = wrapped_env.unwrapped  # Access the unwrapped Isaac Lab environment
    underlying_env.target_cube_coords = None
    if "red cube" in cube_coords:
        underlying_env.target_cube_coords = torch.tensor(cube_coords["red cube"]["center"], device=underlying_env.sim.device)
        print(f"[INFO] Target cube (red) coordinates: {underlying_env.target_cube_coords}")
    elif "green cube" in cube_coords:
        underlying_env.target_cube_coords = torch.tensor(cube_coords["green cube"]["center"], device=underlying_env.sim.device)
        print(f"[INFO] Target cube (green) coordinates: {underlying_env.target_cube_coords}")
    else:
        print("[WARNING] No cubes detected in the initial image!")

    # create runner from rsl-rl
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    wrapped_env.close()

def save_initial_photo(wrapped_env, file_path):
    """Captures and saves a single RGB image from the overhead camera at the start of training."""
    # Unwrap the RslRlVecEnvWrapper to access the underlying Isaac Lab environment
    underlying_env = wrapped_env.unwrapped
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


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()