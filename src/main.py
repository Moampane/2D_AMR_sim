"""
Main file for running the simulator.
"""

import csv
import pandas as pd
from pathlib import Path
from environment import Environment
from robot import Robot
from utils import Position, Pose, Landmark, Bounds
from viz import Visualizer

if __name__ == "__main__":
    # set up the environment
    dimensions = Bounds(0, 10, 0, 10)
    dt = 0.1
    obstacles = [Bounds(6, 8, 2, 7), Bounds(1, 2, 8, 9)]
    landmarks = [Landmark(Position(1, 1), 0), Landmark(Position(3, 3), 1), Landmark(Position(5, 5), 2)]
    initial_robot_pose = Pose(Position(1, 1), 0.0)

    env = Environment(
        dimensions,
        dt,
        obstacles,
        landmarks,
        initial_robot_pose,
    )

    # set up the robot
    robot_info = {
        "Motor": {
            "linear_noise": 0.03,
            "angular_noise": 0.05
        }
    }
    sensor_info = {
        "LandmarkPinger": {
            "interval": 0.5,
            "max_range": 3.0,
            "range_noise": 0.1,
            "range_noise_ratio": 0.05,
            "bearing_noise": 0.08,
            "bearing_noise_ratio": 0.0,
        },
        "Odometry": {
            "interval": 0.1,
            "x_noise": 0.05,
            "y_noise": 0.05,
            "ang_noise": 0.01,
            "x_noise_ratio": 0.01,
            "y_noise_ratio": 0.01,
            "ang_noise_ratio": 0.1,
        }
    }

    robot = Robot(env, robot_info, sensor_info)

    # set up timekeeping
    total_seconds = 20
    total_timesteps = total_seconds / env.timestep

    # set up logging
    ground_truth_history = []
    sensor_data_history = []

    # set up input filepath and output filepaths
    input_commands_filepath = "input/vel_cmd_example.csv"
    output_ground_truth_filepath = "output/gt_data.csv"
    output_sensor_data_filepath = "output/sensor_data.csv"
    output_env_data_filepath = "output/env_data.csv"

    # open up the instructions, pop the first
    with open(input_commands_filepath, "r") as cmd:
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
    OUTPUT_PATH = Path("./output")
    visualizer = Visualizer(OUTPUT_PATH)
    visualizer.draw_all()
