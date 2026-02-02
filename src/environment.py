"""
A simulation environment for a mobile robot operating in two dimensions.

The Environment class models the world that the robots navigate in. The world is continuous and two-dimensional. The world possesses an outer border, internal obstacles, and identifiable landmarks. The world also manages the passage of time and the motion of robotic agents within the world over time.

Critically, the environment tracks the robot's state. In this case, the robot's state is a vector that includes three state variables: x position, y position, and heading.
"""

from utils import Position, Pose, Bounds, Landmark, BearingRange
import math


class Environment:
    """
    A class that models the world simulation environment and the robot's state.

    Attributes:
        dimensions: the horizontal and vertical size of the world
        dt: the length of each timestep, in seconds
        obstacles: a list of obstacles
        landmarks: a list of landmarks
        robot_pose: the position and heading of the robot in the world
    """

    def __init__(
        self,
        dimensions: Bounds,
        dt: float,
        init_obstacles: list[Bounds],
        init_landmarks: list[Landmark],
        robot_starting_pose: Pose,
    ):
        """
        Initialize an instance of the Environment class.

        Args:
            dimensions: the horizontal and vertical size of the world
            dt: the length of each timestep, in seconds
            init_obstacles: a list of obstacles
            init_landmarks: a list of landmarks
            robot_starting_pose: the initial position and heading of the robot
        """
        self.world_bounds = dimensions
        self.timestep = dt
        self.time = 0.0

        # Test robot_starting_pose is within the world
        assert dimensions.within_bounds(robot_starting_pose.pos)

        # Test obstacles are in the world and robot is not in them
        for obs in init_obstacles:
            assert dimensions.within_x(obs.x_min)
            assert dimensions.within_x(obs.x_max)
            assert dimensions.within_y(obs.y_min)
            assert dimensions.within_y(obs.y_max)

            assert not obs.within_bounds(robot_starting_pose.pos)

        # Test landmarks are in the world and are unique
        for mark in init_landmarks:
            assert dimensions.within_bounds(mark.pos)
        assert len(set(l.id for l in init_landmarks)) == len(init_landmarks)

        self.robot_pose = robot_starting_pose   # Theta is [0, 2pi]
        self.obstacles = init_obstacles
        self.landmarks = init_landmarks

    def robot_step(self, dx: float, dy: float, dtheta: float):
        """
        Update the robot's position and heading in the world. The robot should not be able to pass through obstacles or outside of the world bounds.

        Args:
            dx: change in x position
            dy: change in y position
            dtheta: change in heading

        Returns:
            Nothing, but update the robot_pose property at the end
        """
        # Get valid dx and dy
        dx, dy = self.is_valid_motion(dx, dy)

        # Set new robot pose
        curr_x, curr_y, curr_theta = self.robot_pose.pos.x, self.robot_pose.pos.y, self.robot_pose.theta
        self.robot_pose = Pose(Position(curr_x + dx, curr_y + dy), (curr_theta + dtheta) % 2 * math.pi)

        # Take timestep
        self.time += self.timestep

    def is_valid_motion(self, dx: float, dy: float):
        """
        Given attempted x and y motion by the robot, determine what motion is physically possible (i.e. doesn't go through any obstacles or barriers). Return the actual motion that will be executed.

        Args:
            dx: attempted change in x position
            dy: attempted change in y position

        Returns:
            dx: change in x position that should be executed
            dy: change in y position that should be executed
        """
        # TODO: could make it so robot goes to the bound (world or obstacle wall), but would require knowing what is invalidating the position
        new_x = self.robot_pose.pos.x
        new_y = self.robot_pose.pos.y

        # Check new x
        if self.is_valid_position(Position(new_x + dx, new_y)):
            new_x += dx

        # Check new y
        if self.is_valid_position(Position(new_x, new_y + dy)):
            new_y += dy

        return new_x, new_y

    def is_valid_position(self, position: Position):
        """
        Check if a given robot position is valid; i.e. not out-of-bounds or within an obstacle. Return a boolean representing whether or not this condition is true.

        Args:
            position: the robot position

        Returns:
            true if the position is valid and false otherwise
        """
        for ob in self.obstacles:
            if ob.within_bounds(position):
                return False

        return self.world_bounds.within_bounds(position)

    def get_robot_pose(self):
        """
        Return the true robot pose.
        """
        return self.robot_pose

    def get_proximity_to_landmarks(self):
        """
        Return a list of the robot's true range and bearing to all landmarks.
        """
        # TODO: fill in the function
        pass

    def take_state_snapshot(self):
        """
        Return true state information about this timestep, including time, robot position, and the robot's bearing/range to landmarks, in a table format.
        """
        # TODO: fill in the function
        pass

    def get_environment_info(self):
        """
        Return static information about the environment, including dimensions, timestep size, locations and dimensions of obstacles, and locations of landmarks.
        """
        # TODO: fill in the function
        pass
