"""
Main file for running the simulator.
"""

import csv, yaml, argparse
import pandas as pd
from pathlib import Path
from environment import Environment
from robot import Robot
from utils import Position, Pose, Landmark, Bounds
from viz import Visualizer

if __name__ == "__main__":
    # set up pathing and argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y",
        "--config",
        type=Path,
        default=Path("../input/config.yaml"),
        help="Path to config.yaml file.",
    )
    parser.add_argument(
        "-c",
        "--cmds",
        type=Path,
        default=Path("../input/vel_cmd_example.csv"),
        help="Path to motor commands file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("../output/"),
        help="Indicate the desired output folder.",
    )
    args = parser.parse_args()
    CONFIG_PATH: Path = args.config
    assert CONFIG_PATH.exists()
    CMD_PATH: Path = args.cmds
    assert CMD_PATH.exists()
    OUTPUT_PATH: Path = args.output
    assert OUTPUT_PATH.exists()

    # pull config info
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        env_info = config["environment"]
        robot_info = config["robot"]
        sensor_info = config["sensors"]
    
    # set up the environment
    dimensions = Bounds(0, env_info["width"], 0, env_info["height"])
    dt = env_info["timestep"]
    obstacles = [Bounds(dims[0], dims[1], dims[2], dims[3],) for dims in env_info["obstacles"]]
    landmarks = [Landmark(Position(env_info["landmarks"][lm_idx][0], env_info["landmarks"][lm_idx][1]), lm_idx) for lm_idx in range(len(env_info["landmarks"]))]
    initial_robot_pose = Pose(Position(env_info["robot_start"][0], env_info["robot_start"][1]), env_info["robot_start"][2])
    pinger_range = env_info["pinger_range"]

    env = Environment(
        dimensions,
        dt,
        obstacles,
        landmarks,
        initial_robot_pose,
        pinger_range,
    )

    robot = Robot(env, robot_info, sensor_info)

    # set up timekeeping
    total_seconds = env_info["runtime"]
    total_timesteps = total_seconds / env.timestep

    # set up logging
    ground_truth_history = []
    sensor_data_history = []

    # set up input filepath and output filepaths
    output_ground_truth_filepath = OUTPUT_PATH / "gt_data.csv"
    output_sensor_data_filepath = OUTPUT_PATH / "sensor_data.csv"
    output_env_data_filepath = OUTPUT_PATH / "env_data.csv"

    # open up the instructions, pop the first
    with open(CMD_PATH, "r") as cmd:
        vel_cmds = csv.reader(cmd)
        next_cmd = next(vel_cmds)  # skip column names
        next_cmd = next(vel_cmds)
        current_x_vel = float(next_cmd[1])
        current_y_vel = float(next_cmd[2])
        current_ang_vel = float(next_cmd[3])
        # iterate through each timestep
        for step in range(int(total_timesteps) + 1):
            # Take a ground truth snapshot and add it to the history
            ground_truth_history.append(env.take_state_snapshot())

            # Take sensor measurements and add it to the history
            sensor_data_history.append(robot.take_sensor_measurements())

            # Retrieve the next motor command from the input file
            if round(float(next_cmd[0]), 3) <= env.timestep * step:
                current_x_vel = float(next_cmd[1])
                current_y_vel = float(next_cmd[2])
                current_ang_vel = float(next_cmd[3])
                try:
                    next_cmd = next(vel_cmds)
                except StopIteration:
                    pass

            # Execute the motor command
            robot.robot_step_translational(current_x_vel, current_y_vel, current_ang_vel)

    # at the end, write the histories into output files
    # Write ground_truth_history to a file
    ground_truth_df = pd.concat(ground_truth_history, ignore_index=True)
    ground_truth_df.to_csv(output_ground_truth_filepath, index=False)

    # Write sensor_data_history to a file
    sensor_data_df = pd.concat(sensor_data_history, ignore_index=True)
    sensor_data_df.to_csv(output_sensor_data_filepath, index=False)

    # Write environment data to a file
    env_data_df = env.get_environment_info()
    env_data_df.to_csv(output_env_data_filepath, index=False)

    # Make visualizer
    visualizer = Visualizer(OUTPUT_PATH)
    visualizer.draw_all()
