"""
Main file for running the simulator.
"""

import csv, yaml, argparse
import pandas as pd
import numpy as np
from pathlib import Path
from environment import Environment
from robot import Robot
from utils import Position, Pose, Landmark, Bounds
from viz import Visualizer
from kalman_filter import KalmanFilter
from extended_kalman_filter import ExtendedKalmanFilter

if __name__ == "__main__":
    # setup pathing and argparse
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
        default=Path("../input/diff_cmd_example.csv"),
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

    # setup the environment
    dimensions = Bounds(0, env_info["width"], 0, env_info["height"])
    dt = env_info["timestep"]
    obstacles = [
        Bounds(
            dims[0],
            dims[1],
            dims[2],
            dims[3],
        )
        for dims in env_info["obstacles"]
    ]
    landmarks = [
        Landmark(
            Position(
                env_info["landmarks"][lm_idx][0], env_info["landmarks"][lm_idx][1]
            ),
            lm_idx,
        )
        for lm_idx in range(len(env_info["landmarks"]))
    ]
    initial_robot_pose = Pose(
        Position(env_info["robot_start"][0], env_info["robot_start"][1]),
        env_info["robot_start"][2],
    )
    pinger_range = env_info["pinger_range"]

    env = Environment(
        dimensions,
        dt,
        obstacles,
        landmarks,
        initial_robot_pose,
        pinger_range,
    )

    # setup the robot
    robot = Robot(env, robot_info, sensor_info)

    if not robot_info["diff_drive"]:
        # setup the Kalman filter
        kf = KalmanFilter(dt, initial_robot_pose.to_array())
    else:
        # setup extended Kalman filter
        ekf = ExtendedKalmanFilter(dt, initial_robot_pose.to_array())

    # setup timekeeping
    total_seconds = env_info["runtime"]
    total_timesteps = total_seconds / env.timestep

    # setup logging
    ground_truth_history = []
    sensor_data_history = []
    kf_history = []
    ekf_history = []

    # setup input filepath and output filepaths
    output_ground_truth_filepath = OUTPUT_PATH / "gt_data.csv"
    output_sensor_data_filepath = OUTPUT_PATH / "sensor_data.csv"
    output_env_data_filepath = OUTPUT_PATH / "env_data.csv"
    output_kf_data_filepath = OUTPUT_PATH / "kf_data.csv"
    output_ekf_data_filepath = OUTPUT_PATH / "ekf_data.csv"

    # open up the instructions, pop the first
    with open(CMD_PATH, "r") as cmd:
        vel_cmds = csv.reader(cmd)
        next_cmd = next(vel_cmds)  # Skip column names
        next_cmd = next(vel_cmds)

        if not robot_info["diff_drive"]:  # If translational drive
            current_x_vel = float(next_cmd[1])
            current_y_vel = float(next_cmd[2])
            current_ang_vel = float(next_cmd[3])
            # iterate through each timestep
            for step in range(int(total_timesteps) + 1):
                # Take a ground truth snapshot and add it to the history
                ground_truth_history.append(env.take_state_snapshot())

                # Take sensor measurements and add it to the history
                curr_sensor_df = robot.take_sensor_measurements()
                sensor_data_history.append(curr_sensor_df)

                # Iterate Kalman Filter
                odom_x_vel = curr_sensor_df.at[0, "Odometry_x_vel"]
                odom_y_vel = curr_sensor_df.at[0, "Odometry_y_vel"]
                odom_ang_vel = curr_sensor_df.at[0, "Odometry_ang_vel"]

                kf.predict(np.array([[odom_x_vel], [odom_y_vel], [odom_ang_vel]]))

                if "GPS_x" in curr_sensor_df.columns:
                    gps_x = curr_sensor_df.at[0, "GPS_x"]
                    gps_y = curr_sensor_df.at[0, "GPS_y"]
                    gps = robot.sensors["GPS"]

                    kf.update(np.array([[gps_x], [gps_y]]), gps.H, gps.R)

                kf_history.append(
                    pd.DataFrame(
                        {
                            "x": [kf.x[0][0]],
                            "y": [kf.x[1][0]],
                            "theta": [kf.x[2][0]],
                        }
                    )
                )

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
                robot.robot_step_translational(
                    current_x_vel, current_y_vel, current_ang_vel
                )
        else:
            current_lin_vel = float(next_cmd[1])
            current_ang_vel = float(next_cmd[2])
            # iterate through each timestep
            for step in range(int(total_timesteps) + 1):
                # Take a ground truth snapshot and add it to the history
                ground_truth_history.append(env.take_state_snapshot())

                # Take sensor measurements and add it to the history
                curr_sensor_df = robot.take_sensor_measurements()
                sensor_data_history.append(curr_sensor_df)

                # Iterate EKF
                odom_lin_vel = curr_sensor_df.at[0, "Odometry_lin_vel"]
                odom_ang_vel = curr_sensor_df.at[0, "Odometry_ang_vel"]

                ekf.predict(np.array([[odom_lin_vel], [odom_ang_vel]]))

                if "lmp_bearing_0" in curr_sensor_df.columns:
                    lmp = robot.sensors["LandmarkPinger"]
                    for _, row in curr_sensor_df.iterrows():
                        for lm_id in range(len(env_info["landmarks"])):
                            bearing = row.get(f"lmp_bearing_{lm_id}")
                            lm_range = row.get(f"lmp_range_{lm_id}")

                            if bearing != float("inf"):
                                z = np.array([[lm_range], [bearing]])

                                residual = lmp.y(z, ekf.x_state, lm_id)
                                H = lmp.H_eval(ekf.x_state, lm_id)
                                S = H @ ekf.P @ H.T + lmp.R(z)
                                mahal = (residual.T @ np.linalg.inv(S) @ residual)[0, 0]
                                if mahal > 3.0:
                                    continue

                                ekf.update(
                                    H,
                                    lmp.R(z),
                                    None,
                                    residual,
                                )

                if "GPS_x" in curr_sensor_df.columns:
                    gps_x = curr_sensor_df.at[0, "GPS_x"]
                    gps_y = curr_sensor_df.at[0, "GPS_y"]
                    gps = robot.sensors["GPS"]

                    ekf.update(gps.H, gps.R, np.array([[gps_x], [gps_y]]), None)

                ekf_history.append(
                    pd.DataFrame(
                        {
                            "x": [ekf.x_state[0][0]],
                            "y": [ekf.x_state[1][0]],
                            "theta": [ekf.x_state[2][0]],
                        }
                    )
                )

                # Retrieve the next motor command from the input file
                if round(float(next_cmd[0]), 3) <= env.timestep * step:
                    current_lin_vel = float(next_cmd[1])
                    current_ang_vel = float(next_cmd[2])
                    try:
                        next_cmd = next(vel_cmds)
                    except StopIteration:
                        pass

                # Execute the motor command
                robot.robot_step_differential(current_lin_vel, current_ang_vel)

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

    # Write predictions to a file
    if not robot_info["diff_drive"]:
        kf_df = pd.concat(kf_history, ignore_index=True)
        kf_df.to_csv(output_kf_data_filepath, index=False)
    else:
        ekf_df = pd.concat(ekf_history, ignore_index=True)
        ekf_df.to_csv(output_ekf_data_filepath, index=False)

    # Make visualizer
    visualizer = Visualizer(OUTPUT_PATH, robot_info["diff_drive"])
    visualizer.draw_all()
