"""
A simulated robotic agent with teleoperation and sensing capabilities.

The Robot class models the robotic agent that explores the world. The robot is remote-controlled by angular and linear velocity commands read from an external file. The robot can execute motor commands to move, and can sense both externally (GPS, landmarks, obstacles) and internally (odometry, IMU).
"""

import random
import pandas as pd
from environment import Environment
from sensors import WheelEncoder, LandmarkPinger, GPS
from utils import floating_mod_zero


class Robot:
    """
    A class that models a simulated robotic agent.

    Attributes:
        env: the environment this robot is operating in
        sensors: list of all robot sensors
    """

    def __init__(self, env: Environment, robot_info: dict, sensor_info: dict):
        """
        Initialize an instance of the Robot class.

        Args:
            env: the environment this robot is operating in
        """
        self.env = env

        # Desired commands
        self.cmd_x_vel = 0.0    # m/s
        self.cmd_y_vel = 0.0    # m/s
        self.cmd_ang_vel = 0.0    # rad/s

        # Noisy / actual commands
        self.act_x_vel = 0.0    # m/s
        self.act_y_vel = 0.0    # m/s
        self.act_ang_vel = 0.0    # rad/s

        # Physical properties
        mtr_info = robot_info["MotorCommands"]
        self.lin_noise = mtr_info["linear_noise"]
        self.ang_noise = mtr_info["angular_noise"]

        # Setup sensors
        lmp_info = sensor_info["LandmarkPinger"]
        odom_info = sensor_info["Odometry"]
        gps_info = sensor_info["GPS"]
        self.sensors = {
            "LandmarkPinger": LandmarkPinger(
                robot = self,
                name = "LandmarkPinger",
                interval = lmp_info["interval"],
                init_max_range = env.pinger_range,
                init_range_noise = lmp_info["range_noise_const"],
                init_range_noise_ratio = lmp_info["range_noise_prop"],
                init_bearing_noise = lmp_info["bearing_noise_const"],
                init_bearing_noise_ratio = lmp_info["bearing_noise_prop"],
            ),
            "Odometry": WheelEncoder(
                robot = self,
                name = "Odometry",
                interval = odom_info["interval"],
                init_x_noise = odom_info["x_noise"],
                init_y_noise = odom_info["y_noise"],
                init_ang_noise = odom_info["ang_noise"],
                x_noise_ratio = odom_info["x_noise_ratio"],
                y_noise_ratio = odom_info["y_noise_ratio"],
                angular_noise_ratio = odom_info["ang_noise_ratio"],
            ),
            "GPS": GPS(
                robot = self,
                name = "GPS",
                interval = gps_info["interval"],
                init_x_noise = gps_info["x_noise"],
                init_y_noise = gps_info["y_noise"],
            ),
        }

    def robot_step_differential(self, lin_vel: float, ang_vel: float):
        """
        Differential-drive mode. Given forward linear and angular velocities, determine the robot's change in x, y, and heading and apply those changes in the environment.

        Args:
            lin_vel: input linear velocity command
            ang_vel: input angular velocity command

        Returns:
            dx: change in x position
            dy: change in y position
            d-theta: change in heading
        """
        # TODO: fill in the function
        pass

    def robot_step_translational(self, x_vel: float, y_vel: float, ang_vel: float):
        """
        Swerve-drive mode. Given x, y, and angular velocities, determine the robot's change in x, y, and heading and apply those changes in the environment.

        Args:
            x_vel: input x velocity command
            y_vel: input y velocity command
            ang_vel: input angular velocity command

        Returns:
            dx: change in x position
            dy: change in y position
            dtheta: change in heading
        """
        # Track commanded velocities
        self.cmd_x_vel = x_vel   # m/s
        self.cmd_y_vel = y_vel   # m/s
        self.cmd_ang_vel = ang_vel   # rad/s

        # Make noisy commands
        x_vel = x_vel * (1 + random.gauss(0, self.lin_noise))
        y_vel = y_vel * (1 + random.gauss(0, self.lin_noise))
        ang_vel = ang_vel * (1 + random.gauss(0, self.ang_noise))

        # Track actual velocities
        self.act_x_vel = x_vel   # m/s
        self.act_y_vel = y_vel   # m/s
        self.act_ang_vel = ang_vel   # rad/s

        # Calculate change in robot state variables
        timestep = self.env.timestep
        dx = x_vel * timestep
        dy = y_vel * timestep
        dtheta = ang_vel * timestep

        # Apply change in robot state variables
        self.env.robot_step(dx, dy, dtheta)

        return dx, dy, dtheta

    def take_sensor_measurements(self):
        """
        Return noisy sensor readings of the environment at this timestep, including data from all sensors, in a table format.
        """
        measurements = pd.DataFrame({"Time": [self.env.time]})

        for sensor in self.sensors.values():
            if floating_mod_zero(self.env.time, sensor.interval):
                sensor.last_meas_t = self.env.time
                measurements = pd.merge(
                    measurements,
                    sensor.sample(),
                    left_index=True,
                    right_index=True,
                )

        # Get the command velocities
        measurements["cmd_x_vel"] = [self.cmd_x_vel]
        measurements["cmd_y_vel"] = [self.cmd_y_vel]
        measurements["cmd_ang_vel"] = [self.cmd_ang_vel]

        return measurements
