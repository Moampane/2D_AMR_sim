"""
An abstract base class that all sensor classes must inherit from. This structure guarantees that all sensors have certain traits, including a name, sampling interval, and sampling function.

In addition to basic features, all sensors should have noise constants. Different sensors may use different distributions to model noise, and may take in different parameters to shape that noise. For example, one sensor might have a constant noise mean, while another might have noise that grows proportionally with distance or time.

Exteroceptive sensors measure the robot's relationship to the world. This includes GPS, cameras, LiDAR, and anything else that takes a measurement that can relate the robot's state to things beyond the robot.

Proprioceptive sensors measure the robot's relationship to its past states. This includes IMUs, wheel encoders, and anything else that measures how the robot's state is relatively changing, without relating the robot to the world.
"""

import random
from abc import ABC, abstractmethod
from math import pi
from utils import BearingRange


class SensorInterface(ABC):
    """
    A basic Interface to standardize all sensors.

    Attributes:
        name: string identifier
        robot: reference robot. required for observing the environment
        interval: period between measurements
        last_meas_t: time of last sensor measurement
    """

    def __init__(self, name: str, robot, interval: float):
        """
        Initialize a sensor class instace.

        Args:
            name: reference identifier
            robot: reference robot
            interval: period between measurements
        """
        self._name = name
        self.robot = robot
        self._interval = interval
        self.last_meas_t = robot.env.time

    @property
    def name(self) -> str:
        """
        Getter for the name property.
        """
        return self._name

    @property
    def interval(self) -> float:
        """
        Getter for the interval property.
        """
        return self._interval

    @property
    def last_meas_t(self) -> float:
        """
        Getter for the time of last measurement property.
        """
        return self._last_meas_t

    @last_meas_t.setter
    def last_meas_t(self, value: float):
        """
        Setter for the time of last measurement property.
        """
        self._last_meas_t = value

    @abstractmethod
    def sample(self):
        """
        Sample the environment and return the noisy measurement(s).
        """
        pass


class WheelEncoder(SensorInterface):
    """
    This class represents a wheel encoder set that measures the robot's motor speeds.
    Reports noisy estimates of linear and angular velocities.

    Attributes:
        name: string identifier
        robot: reference robot
        interval: period between measurements
        last_meas_t: time of last measurement
        x_noise: absolute noise for x velocity stdev
        y_noise: absolute noise for y velocity stdev
        ang_noise: absolute noise for angular velocity stdev
        prop_x_noise: proportional noise for x velocity
        prop_y_noise: proportional noise for y velocity
        prop_ang_noise: proportional noise for angular
    """

    def __init__(
        self,
        robot,
        name="wheel_encoder",
        interval=0.1,
        init_x_noise=0.05,
        init_y_noise=0.05,
        init_ang_noise=0.03,
        x_noise_ratio=0.05,
        y_noise_ratio=0.05,
        angular_noise_ratio=0.03,
    ):
        """
        Initialize an instance of the WheelEncoder class.

        Args:
            robot: reference robot
            name: reference identifier
            interval: period between measurements
            init_x_noise: absolute noise for x velocity stdev
            init_y_noise: absolute noise for y velocity stdev
            init_ang_noise: absolute noise for angular velocity stdev
            x_noise_ratio: proportional noise for x velocity
            y_noise_ratio: proportional noise for y velocity
            angular_noise_ratio: proportional noise for angular
        """
        super().__init__(name, robot, interval)
        self.x_noise = init_x_noise  # m/s
        self.y_noise = init_y_noise  # m/s
        self.ang_noise = init_ang_noise  # rad/s
        self.prop_x_noise = x_noise_ratio
        self.prop_y_noise = y_noise_ratio
        self.prop_ang_noise = angular_noise_ratio

    def sample(self):
        """
        Sample the robot's linear and angular velocity.
        """
        commanded_x_vel = self.robot.cmd_x_vel
        commanded_y_vel = self.robot.cmd_y_vel
        commanded_ang_vel = self.robot.cmd_ang_vel

        measured_x_vel = random.gauss(commanded_x_vel, self.x_noise + abs(commanded_x_vel) * self.prop_x_noise)
        measured_y_vel = random.gauss(commanded_y_vel, self.y_noise + abs(commanded_y_vel) * self.prop_y_noise)
        measured_ang_vel = random.gauss(commanded_ang_vel, self.ang_noise + abs(commanded_ang_vel) * self.prop_ang_noise)

        return measured_x_vel, measured_y_vel, measured_ang_vel


class LandmarkPinger(SensorInterface):
    """
    This class represents a sensor that measures the range and bearing between the robot and the floating-point landmarks on the map. In practice, this sensor could be a ToF sensor, a node in a network of beacons, or even a camera.

    Attributes:
        name: reference identifier
        robot (Robot): reference robot
        interval (float): period between measurements
        max_range (int): maximum distance from a beacon for it to be visible
        range_noise (float): absolute noise for range stdev
        range_noise_ratio (float): porportional noise for range stdev
        bearing_noise (float): absolute noise for bearing stdev
    """

    def __init__(
        self,
        robot,
        name="landmark_pinger",
        interval=1.0,
        init_range_noise=0.5,
        init_range_noise_ratio=0.05,
        init_bearing_noise=pi / 6,
        init_bearing_noise_ratio=0.03,
        init_max_range=10.0,
    ):
        """
        Initialize an instance of the LandmarkPinger class.

        Args:
            name (str): reference identifier
            robot (Robot): reference robot
            interval (float): period between measurements
            init_max_range (int): maximum distance from a beacon for it to be visible
            init_range_noise (float): absolute noise for range stdev
            init_range_noise_ratio (float): proportional noise for range stdev
            init_bearing_noise (float): absolute noise for bearing stdev
            init_range_noise_ratio (float): proportional noise for range stdev
        """
        super().__init__(name, robot, interval)
        self.max_range = init_max_range  # meters
        self.range_noise = init_range_noise  # meters
        self.range_noise_ratio = init_range_noise_ratio
        self.bearing_noise = init_bearing_noise  # radians
        self.bearing_noise_ratio = init_bearing_noise_ratio

    def sample(self):
        """
        Reports noisy measurements of the bearing and range between the robot and all nearby landmarks.
        """
        gt_bearing_ranges = self.robot.env.get_proximity_to_landmarks()
        noisy_bearing_ranges = []

        for lm in gt_bearing_ranges:
            if lm.range <= self.max_range:
                noisy_bearing = random.gauss(lm.bearing, self.bearing_noise + lm.bearing * self.bearing_noise_ratio)
                noisy_range = random.gauss(lm.range, self.range_noise + lm.range * self.range_noise_ratio)
            else:
                noisy_bearing = float("inf")
                noisy_range = float("inf")

            noisy_br = BearingRange(lm.landmark_id, noisy_bearing, noisy_range)
            noisy_bearing_ranges.append(noisy_br)

        return noisy_bearing_ranges
