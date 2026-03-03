import numpy as np
import random

class KalmanFilter():
    """
    A class that implements the basic Kalman Filter algorithm, which assumes linear system dynamics and Gaussian noise.

    Attributes:
        dt: the length of each timestep, in seconds
        x: the state vector for the system we are estimating
        P: the process model, describing the uncertainty in our estimate
        F: the state transition matrix, describing how our state naturally changes from timestep to timestep
        B: the control input model, describing how control inputs affect each state variable in the state vector
        Q: the process noise, modeling unexpected disturbance in state transitions
    """
    def __init__(self, dt, prior):
        """
        Initialize an instance of the KalmanFilter class.

        Args:
            dt: the length of each timestep, in seconds
            prior: the initial estimates for each state variable
        """
        self.timestep = dt
        self.x = prior
        self.P = np.eye(3) * 50
        self.F = np.eye(3)
        self.B = np.eye(3) * dt
        self.Q = self.get_Q()

    def predict(self, u):
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def get_Q(self):
        """
        Generate white noise to apply to the process model after each prediction.
        """
        stdev = 0.1
        return np.array(
            [
                [
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                ],
                [
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                ],
                [
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                ],
            ]
        )