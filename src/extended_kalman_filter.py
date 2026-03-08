"""
Extended Kalman Filter implementation for the simulator. Tracks the following states:

x = [x, y, theta]

We expect the following control inputs:

u = [v, w]
"""

import numpy as np
import sympy
from sympy.abc import x, y, v, w, R, theta
from sympy import Matrix, Symbol
import random

from utils import wrap_angle


class ExtendedKalmanFilter:
    """
    This class implements the Extended Kalman Filter algorithm.
    """

    def __init__(self, dt: float, prior: np.ndarray):
        """
        Initialize an Extended Kalman Filter.

        A state vector includes the following:
            x position
            y position
            heading


        Args:
            dt: the length of each timestep, in seconds
            prior: the initial estimates for each state variable-
        """
        self.timestep: float = dt

        self.x_state: np.ndarray = prior

        self.P: np.ndarray = np.eye(len(self.x_state)) * 50

        self.f_xu: Matrix = Matrix(
            [
                [x + v * sympy.cos(theta) * self.timestep],  # calculation of x
                [y + v * sympy.sin(theta) * self.timestep],  # calculation of y
                [theta + w * self.timestep],  # calculation of theta
            ]
        )

        self.F: Matrix = self.f_xu.jacobian(Matrix([x, y, theta]))

        # dictionary that maps Sympy symbols to numerical values. we will use these to substitute values into our symbolic matrices!
        self.subs: dict[Symbol, float] = {
            x: self.x_state[0],
            y: self.x_state[1],
            theta: self.x_state[2],
            v: 0,
            w: 0,
        }

    def predict(self, u: np.ndarray):
        """
        Predicts the next state vector and its covariance matrix using the state transition matrix and an input control vector. The Kalman Filter uses the following predict equations:

        x_t+1 = f(x,u)
        P_t+1 = F * P * F.T + Q

        where F is the Jacobian of f(x,u)

        Args:
            u: the input control vector
        """
        self.subs[x] = self.x_state[0, 0]
        self.subs[y] = self.x_state[1, 0]
        self.subs[theta] = self.x_state[2, 0]
        self.subs[v] = u[0, 0]
        self.subs[w] = u[1, 0]

        # evaluate the nonlinear motion model f(x,u) at the subsitution values
        fxu_eval = np.array(self.f_xu.subs(self.subs).evalf(), dtype=float)

        # evaluate the Jacobian matrix F at the substitution values
        F_eval = np.array(self.F.subs(self.subs).evalf(), dtype=float)

        # calculate the next state prediction
        self.x_state = fxu_eval
        self.x_state[2, 0] = wrap_angle(self.x_state[2, 0])

        # calculate the next covariance prediction
        self.P = F_eval @ self.P @ F_eval.T + self.get_Q()

        # return state vector and state covariance
        return self.x_state, self.P

    def update(
        self,
        H: np.ndarray,
        R: np.ndarray,
        z: np.ndarray | None,
        y: np.ndarray | None,
    ):
        """
        Updates the current state prediction using observations from the environment. The Extended Kalman Filter uses the following update equations:

        x = x + K * y
        P = P - K * H * P

        Where K and y are given by the following:
        y = z - h(x) (residual: error between observation and expected observation given estimated state vector)
        K = P * H.T * inv(S) (Kalman Gain: portion of total uncertainty that is from the prediction)
        S = H * P * H.T + R (total uncertainty in the system)

        where H is the Jacobian of h(x)

        Args:
            H: the Jacobian of the nonlinear measurement model, which relates the state space to the measurement space
            R: the measurement noise model (covariance)
            y: the residual, which is the error between the measured observation and the observation expected by the predicted state
        """
        # calculate the total uncertainty in the system
        S = H @ self.P @ H.T + R

        # calculate the Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)

        if y is None:
            y = z - H @ self.x_state

        # update state vector
        self.x_state = self.x_state + K @ y
        self.x_state[2, 0] = wrap_angle(self.x_state[2, 0])

        # update process model
        self.P = self.P - K @ H @ self.P

        # return state vector and process model
        return self.x_state, self.P

    def get_Q(self):
        """
        Generate white noise to apply to the process model after each prediction.
        """
        return np.diag([0.01, 0.01, 0.01])
