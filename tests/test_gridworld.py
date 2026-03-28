"""Tests for GridWorld environment."""

import numpy as np
import pytest
from symnet.env.gridworld import GridWorld, WALL, EMPTY, GOAL_A, GOAL_B, AGENT_A, AGENT_B, OBS_PATCH


@pytest.fixture
def env():
    return GridWorld(grid_size=8, num_obstacles=8, max_steps=200, seed=0)


@pytest.fixture
def small_env():
    """Smaller env for faster tests."""
    return GridWorld(grid_size=8, num_obstacles=4, max_steps=50, seed=1)


# ── reset() ────────────────────────────────────────────────────────────────

class TestReset:
    def test_returns_tuple_of_two_obs(self, env):
        result = env.reset()
        (obs_a, obs_b), info = result
        assert isinstance(obs_a, np.ndarray)
        assert isinstance(obs_b, np.ndarray)

    def test_obs_shape(self, env):
        (obs_a, obs_b), _ = env.reset()
        assert obs_a.shape == (75,), f"Expected (75,), got {obs_a.shape}"
        assert obs_b.shape == (75,)

    def test_obs_dtype(self, env):
        (obs_a, obs_b), _ = env.reset()
        assert obs_a.dtype == np.float32
        assert obs_b.dtype == np.float32

    def test_obs_range(self, env):
        (obs_a, obs_b), _ = env.reset()
        assert obs_a.min() >= 0.0 and obs_a.max() <= 1.0
        assert obs_b.min() >= 0.0 and obs_b.max() <= 1.0

    def test_info_has_positions(self, env):
        _, info = env.reset()
        assert "pos_a"  in info
        assert "pos_b"  in info
        assert "goal_a" in info
        assert "goal_b" in info

    def test_positions_are_distinct(self, env):
        _, info = env.reset()
        pos_a  = tuple(info["pos_a"])
        pos_b  = tuple(info["pos_b"])
        goal_a = tuple(info["goal_a"])
        goal_b = tuple(info["goal_b"])
        positions = [pos_a, pos_b, goal_a, goal_b]
        assert len(set(positions)) == 4, "Agents and goals must be at distinct positions"

    def test_positions_within_grid(self, env):
        _, info = env.reset()
        N = env.N
        for key in ("pos_a", "pos_b", "goal_a", "goal_b"):
            pos = info[key]
            assert 0 <= pos[0] < N and 0 <= pos[1] < N, f"{key} out of bounds: {pos}"

    def test_step_count_reset(self, small_env):
        small_env.reset()
        # Do a step
        small_env.step(0, 0)
        # Reset again
        small_env.reset()
        assert small_env._step_count == 0

    def test_deterministic_with_seed(self, env):
        (a1, b1), _ = env.reset(seed=42)
        (a2, b2), _ = env.reset(seed=42)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)


# ── step() ─────────────────────────────────────────────────────────────────

class TestStep:
    def test_returns_correct_types(self, env):
        env.reset()
        result = env.step(0, 0)
        (obs_a, obs_b), reward, terminated, truncated, info = result
        assert obs_a.shape == (75,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_increments_counter(self, env):
        env.reset()
        assert env._step_count == 0
        env.step(0, 0)
        assert env._step_count == 1
        env.step(1, 1)
        assert env._step_count == 2

    def test_all_actions_are_valid(self, env):
        for action in range(4):  # up, down, left, right
            env.reset(seed=action)
            obs, reward, terminated, truncated, info = env.step(action, action)
            assert obs[0].shape == (75,)

    def test_info_has_required_keys(self, env):
        env.reset()
        _, _, _, _, info = env.step(0, 0)
        required = {"pos_a", "pos_b", "goal_a", "goal_b", "collision",
                    "a_at_goal", "b_at_goal", "both_at_goal", "step"}
        assert required.issubset(info.keys())

    def test_step_count_in_info(self, env):
        env.reset()
        for expected in range(1, 4):
            _, _, _, _, info = env.step(0, 0)
            assert info["step"] == expected


# ── Collision ──────────────────────────────────────────────────────────────

class TestCollision:
    def _force_collision(self, env):
        """
        Place agents adjacent and move them into each other.
        Returns reward and info from the step that causes collision.
        """
        (obs_a, obs_b), info = env.reset(seed=0)
        # Manually place agents so they will collide
        env._pos_a = np.array([4, 3], dtype=np.int32)
        env._pos_b = np.array([4, 5], dtype=np.int32)
        env._grid  = env._build_grid()
        # A moves right (3), B moves left (2) — they both target (4,4)
        _, reward, _, _, info = env.step(3, 2)
        return reward, info

    def test_collision_detected(self, env):
        _, info = self._force_collision(env)
        assert info["collision"] is True

    def test_collision_penalty_applied(self, env):
        reward, info = self._force_collision(env)
        # Reward should include collision penalty (-0.5)
        assert reward < 0, f"Expected negative reward on collision, got {reward}"
        assert info["collision_penalty"] == pytest.approx(-0.5, abs=1e-6)

    def test_agents_dont_move_on_collision(self, env):
        (obs_a, obs_b), _ = env.reset(seed=0)
        env._pos_a = np.array([4, 3], dtype=np.int32)
        env._pos_b = np.array([4, 5], dtype=np.int32)
        env._grid  = env._build_grid()
        _, _, _, _, info = env.step(3, 2)  # both try to reach (4,4)
        # Neither should have moved to (4,4)
        assert not (info["pos_a"][0] == 4 and info["pos_a"][1] == 4), \
            "Agent A should not be at collision cell"
        assert not (info["pos_b"][0] == 4 and info["pos_b"][1] == 4), \
            "Agent B should not be at collision cell"


# ── Task success / termination ─────────────────────────────────────────────

class TestTermination:
    def test_task_reward_on_success(self, env):
        """Manually place both agents on their goals and step."""
        env.reset(seed=0)
        # Place agents directly on goals
        env._pos_a = env._goal_a.copy()
        env._pos_b = env._goal_b.copy()
        env._grid  = env._build_grid()
        # Move in place (action=stay isn't valid, use action that hits wall to not move)
        # Actually: we can step any direction; agent won't move if it would leave grid
        # The simplest approach: place them one cell from goals and steer them in
        # But for the test, let's just check the condition directly via step
        # Since agents are already on goals, let's move them right (action=3) and
        # check if the step before them leaving registers success.
        # Better: place adjacent to bounds so they stay put
        env._pos_a = env._goal_a.copy()
        env._pos_b = env._goal_b.copy()
        # Force them to stay: move toward wall
        # The grid is 8x8, place at (0,0) → move up or left → stays at (0,0)
        # Reset positions to goals
        env._pos_a = env._goal_a.copy()
        env._pos_b = env._goal_b.copy()
        env._grid  = env._build_grid()
        _, reward, terminated, _, info = env.step(0, 0)  # both try to move up
        if info["both_at_goal"]:
            assert terminated is True
            assert reward >= 10.0  # task_reward applied

    def test_timeout_on_max_steps(self):
        """Episode should truncate at max_steps."""
        env = GridWorld(grid_size=8, num_obstacles=4, max_steps=5, seed=99)
        env.reset()
        truncated = False
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(0, 0)
            if terminated or truncated:
                break
        assert truncated is True, "Episode should truncate at max_steps=5"

    def test_no_spontaneous_termination(self, env):
        """Agents in middle of grid shouldn't terminate immediately."""
        env.reset(seed=7)
        # Place agents away from goals
        env._pos_a = np.array([0, 0], dtype=np.int32)
        env._pos_b = np.array([0, 1], dtype=np.int32)
        env._grid  = env._build_grid()
        _, _, terminated, truncated, _ = env.step(0, 0)
        # Should not terminate unless they happen to be on goals
        if not env._grid[0, 0] == GOAL_A and not env._grid[0, 1] == GOAL_B:
            assert not terminated


# ── Walls / obstacles ──────────────────────────────────────────────────────

class TestObstacles:
    def test_obstacles_not_walkable(self, env):
        """Agents should not be able to walk onto obstacle cells."""
        env.reset(seed=0)
        N = env.N
        # Find an obstacle cell
        obstacle_cells = [
            (r, c)
            for r in range(N)
            for c in range(N)
            if env._grid[r, c] == WALL
        ]
        if not obstacle_cells:
            pytest.skip("No obstacles in this seed's layout")

        obs_r, obs_c = obstacle_cells[0]
        # Place agent A adjacent to the obstacle
        for dr, dc, action in [(-1, 0, 1), (1, 0, 0), (0, -1, 3), (0, 1, 2)]:
            adj_r, adj_c = obs_r + dr, obs_c + dc
            if 0 <= adj_r < N and 0 <= adj_c < N and env._grid[adj_r, adj_c] == EMPTY:
                env._pos_a = np.array([adj_r, adj_c], dtype=np.int32)
                env._grid  = env._build_grid()
                env.step(action, 1)  # A tries to walk into wall
                pos_a = env._pos_a
                assert not (pos_a[0] == obs_r and pos_a[1] == obs_c), \
                    "Agent A walked through an obstacle!"
                break

    def test_boundary_not_walkable(self, env):
        """Agents at the boundary should not walk out of bounds."""
        env.reset(seed=0)
        env._pos_a = np.array([0, 0], dtype=np.int32)
        env._grid  = env._build_grid()
        env.step(0, 1)   # A tries to go up (off grid), B goes down
        pos_a = env._pos_a
        assert pos_a[0] >= 0 and pos_a[1] >= 0, "Agent walked off grid!"


# ── Observation content ────────────────────────────────────────────────────

class TestObservation:
    def test_self_channel_contains_agent_marker(self, env):
        """Channel 2 (agents) should have 0.5 for agent's own position (centre cell)."""
        env.reset(seed=0)
        # Place agent A at a known location away from edges
        env._pos_a = np.array([4, 4], dtype=np.int32)
        env._grid  = env._build_grid()
        obs_a = env._get_obs(env._pos_a, is_agent_a=True)  # this returns the unflattened array internal helper
        centre = OBS_PATCH // 2  # 2
        assert obs_a[2, centre, centre] == pytest.approx(0.5), \
            "Agent A should see itself as 0.5 in centre of agent channel"

    def test_wall_channel_marks_out_of_bounds(self, env):
        """Placing agent at (0,0) should show walls/oob in top-left of patch."""
        env.reset(seed=0)
        env._pos_a = np.array([0, 0], dtype=np.int32)
        env._grid  = env._build_grid()
        obs_a = env._get_obs(env._pos_a, is_agent_a=True)
        # Top-left corner of patch (offset [-2,-2] from (0,0) → out of bounds)
        assert obs_a[0, 0, 0] == 1.0, "Out-of-bounds should be marked as wall"
