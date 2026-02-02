import pytest
import math
from src.utils import Position, Pose, Bounds, Landmark, BearingRange
from src.environment import Environment

class TestEnvironmentInitialization:
    """
    Tests for Environment initialization.
    """

    @pytest.fixture
    def basic_env(self):
        """
        Fixture providing a basic environment.
        """
        return Environment(
            dimensions=Bounds(0, 10, 0, 10),
            robot_starting_pose=Pose(Position(5, 5), 0),
            init_obstacles=[Bounds(2, 3, 2, 3)],
            init_landmarks=[Landmark(Position(7, 7), 0), Landmark(Position(3, 8), 1)],
            dt=0.1,
        )

    def test_initialization_basic(self, basic_env: Environment):
        """
        Test the environment properly instantiates all attributes.
        """
        assert basic_env.world_bounds == Bounds(0, 10, 0, 10)
        assert basic_env.timestep == 0.1
        assert basic_env.time == 0.0
        assert basic_env.robot_pose == Pose(Position(5, 5), 0)
        assert len(basic_env.obstacles) == 1
        assert len(basic_env.landmarks) == 2

    def test_initialization_rejects_robot_outside_bounds(self):
        """
        Test initialization fails if robot starts outside environment.
        """
        with pytest.raises(AssertionError):
            Environment(
                dimensions=Bounds(0, 10, 0, 10),
                robot_starting_pose=Pose(Position(15, 5), 0),  # Outside bounds
                init_obstacles=[],
                init_landmarks=[],
                dt=0.1,
            )

    def test_initialization_rejects_obstacles_outside_bounds(self):
        """
        Test initialization fails if any part of any obstacle is outside environment.
        """
        with pytest.raises(AssertionError):
            Environment(
                dimensions=Bounds(0, 10, 0, 10),
                robot_starting_pose=Pose(),
                init_obstacles=[Bounds(-1, 9, -1, 3)],  # Outside bounds
                init_landmarks=[],
                dt=0.1,
            )

    def test_initialization_rejects_robot_inside_obstacle(self):
        """
        Test initialization fails if robot is in any obstacle.
        """
        with pytest.raises(AssertionError):
            Environment(
                dimensions=Bounds(0, 10, 0, 10),
                robot_starting_pose=Pose(Position(1.5, 1.5), 0),  # Inside obstacle
                init_obstacles=[Bounds(1, 2, 1, 2)],
                init_landmarks=[],
                dt=0.1,
            )

    def test_initialization_rejects_landmarks_outside_bounds(self):
        """
        Test initialization fails if any landmark is outside environment.
        """
        with pytest.raises(AssertionError):
            Environment(
                dimensions=Bounds(0, 10, 0, 10),
                robot_starting_pose=Pose(),
                init_obstacles=[],
                init_landmarks=[Landmark(Position(11, 7), 0)],  # Outside bounds
                dt=0.1,
            )

    def test_initialization_rejects_duplicate_landmarks(self):
        """
        Test initialization fails if any landmark ids are repeated
        """
        with pytest.raises(AssertionError):
            Environment(
                dimensions=Bounds(0, 10, 0, 10),
                robot_starting_pose=Pose(),
                init_obstacles=[],
                init_landmarks=[Landmark(Position(7, 7), 0), Landmark(Position(1, 1), 0)],  # Repeated landmark id
                dt=0.1,
            )
