# run_simulation.py

import argparse
import torch
from isaaclab.app import AppLauncher
import isaaclab.sim as sim_utils
from isaaclab.envs import gym
from isaaclab.scene import InteractiveScene

# 1️⃣ Launch Omniverse simulator
parser = argparse.ArgumentParser(description="Standalone simulation for cube picking environment")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2️⃣ Load the environment
env = gym.make("Isaac-Lift-Cube-Franka-v0", render_mode="rgb_array")
scene = env.scene  # Access the scene (robot, cubes, camera, etc.)
sim = env.sim  # Get simulation context

# 3️⃣ Simulation Loop
def run_simulator(sim, scene):
    """Runs the Isaac Sim environment interactively."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # Reset every 500 steps
        if count % 500 == 0:
            count = 0
            print("[INFO]: Resetting environment...")
            scene.reset()

        # Move the robot using default joint positions
        targets = scene["robot"].data.default_joint_pos
        scene["robot"].set_joint_position_target(targets)
        scene.write_data_to_sim()

        # Step simulation
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # Print sensor data for debugging
        print("\n-------------------------------")
        print("Camera Data:")
        print("RGB Image Shape: ", scene["camera"].data.output["rgb"].shape)
        print("Depth Image Shape: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        print("-------------------------------")

        print("Height Scanner:")
        print("Max Height Value: ", torch.max(scene["height_scanner"].data.ray_hits_w[..., -1]).item())
        print("-------------------------------")

        print("Contact Forces:")
        print("Max Contact Force: ", torch.max(scene["contact_forces"].data.net_forces_w).item())
        print("-------------------------------")

# 4️⃣ Run Simulation
run_simulator(sim, scene)

# 5️⃣ Cleanup
env.close()
