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
            obstacles=[Bounds(2, 3, 2, 3)],
            landmarks=[Landmark(Position(7, 7), 0), Landmark(Position(3, 8), 1)],
            dt=0.1,
        )

    def test_initialization_basic(self, basic_env: Environment):
        """
        Test the environment properly instantiates all attributes.
        """
        assert basic_env.DIMENSIONS == Bounds(0, 10, 0, 10)
        assert basic_env.DT == 0.1
        assert basic_env.time == 0.0
        assert basic_env.robot_pose == Pose(Position(5, 5), 0)
        assert len(basic_env.OBSTACLES) == 1
        assert len(basic_env.LANDMARKS) == 2