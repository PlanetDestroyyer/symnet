"""
GridWorld — custom gymnasium environment for SymNet.

An NxN grid with two agents, two goals, and random obstacles.
Both agents share reward (cooperative setting).
Observations are local 5×5 patches around each agent.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Tile constants (used in the internal grid state)
# ---------------------------------------------------------------------------
EMPTY    = 0
WALL     = 1
GOAL_A   = 2   # goal assigned to agent A
GOAL_B   = 3   # goal assigned to agent B
AGENT_A  = 4
AGENT_B  = 5

# ---------------------------------------------------------------------------
# Observation channel indices
# ---------------------------------------------------------------------------
CH_WALLS  = 0   # 1 where wall or out-of-bounds
CH_GOALS  = 1   # 0.5 for goal_A, 1.0 for goal_B (from each agent's POV)
CH_AGENTS = 2   # 0.5 for self, 1.0 for other agent

OBS_PATCH = 5   # 5×5 local view


class GridWorld(gym.Env):
    """
    Symmetric dual-agent cooperative grid world.

    Parameters
    ----------
    grid_size : int
        Side length of the square grid (default 8).
    num_obstacles : int
        Number of static obstacles placed randomly each episode (default 8).
    max_steps : int
        Maximum episode length before timeout (default 200).
    seed : int | None
        Optional RNG seed.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        grid_size: int = 8,
        num_obstacles: int = 8,
        max_steps: int = 200,
        seed: int | None = None,
        phase: int = 3,
    ) -> None:
        super().__init__()
        self.N = grid_size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.phase = phase

        # Observation space: (75,) float32 in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3 * OBS_PATCH * OBS_PATCH,),
            dtype=np.float32,
        )
        # Action space per agent: 0=up 1=down 2=left 3=right
        self.action_space = spaces.Discrete(4)

        self._rng = np.random.default_rng(seed)

        # State (initialised on reset)
        self._grid: np.ndarray | None = None
        self._pos_a: np.ndarray | None = None  # (row, col)
        self._pos_b: np.ndarray | None = None
        self._goal_a: np.ndarray | None = None
        self._goal_b: np.ndarray | None = None
        self._step_count: int = 0
        self._episode_start_step: int = 0

        # Tracking for efficiency bonus
        self._solved_step: int | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_distinct_positions(self, n: int) -> list[np.ndarray]:
        """Sample n distinct free (row, col) positions on the grid."""
        positions: list[np.ndarray] = []
        occupied: set[tuple[int, int]] = set()
        while len(positions) < n:
            r = int(self._rng.integers(0, self.N))
            c = int(self._rng.integers(0, self.N))
            if (r, c) not in occupied:
                occupied.add((r, c))
                positions.append(np.array([r, c], dtype=np.int32))
        return positions

    def _build_grid(self) -> np.ndarray:
        """Build the internal grid array from current state."""
        grid = np.zeros((self.N, self.N), dtype=np.int32)
        grid[tuple(self._goal_a)]  = GOAL_A
        grid[tuple(self._goal_b)]  = GOAL_B
        grid[tuple(self._pos_a)]   = AGENT_A
        grid[tuple(self._pos_b)]   = AGENT_B
        return grid

    def _get_obs(self, pos: np.ndarray, is_agent_a: bool) -> np.ndarray:
        """
        Extract a 5×5 patch centred on `pos`.

        Channels:
          0 – walls / out-of-bounds
          1 – goals (goal_A=0.5, goal_B=1.0 from this agent's POV)
          2 – agents (self=0.5, other=1.0)
        """
        half = OBS_PATCH // 2   # 2
        obs = np.zeros((3, OBS_PATCH, OBS_PATCH), dtype=np.float32)

        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                pi = pos[0] + di
                pj = pos[1] + dj
                oi = di + half
                oj = dj + half

                # Out-of-bounds → wall
                if pi < 0 or pi >= self.N or pj < 0 or pj >= self.N:
                    obs[CH_WALLS, oi, oj] = 1.0
                    continue

                tile = self._grid[pi, pj]

                if tile == WALL:
                    obs[CH_WALLS, oi, oj] = 1.0
                elif tile == GOAL_A:
                    # For agent A, goal_A is "my goal" (0.5); for B, it's "other goal"
                    obs[CH_GOALS, oi, oj] = 0.5 if is_agent_a else 1.0
                elif tile == GOAL_B:
                    obs[CH_GOALS, oi, oj] = 1.0 if is_agent_a else 0.5
                elif tile == AGENT_A:
                    obs[CH_AGENTS, oi, oj] = 0.5 if is_agent_a else 1.0
                elif tile == AGENT_B:
                    obs[CH_AGENTS, oi, oj] = 1.0 if is_agent_a else 0.5

        return obs

    def _try_move(self, pos: np.ndarray, action: int) -> np.ndarray:
        """
        Return the new position after applying `action`.
        Returns the same position if the move is blocked by wall or boundary.
        """
        delta = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[action]
        new_pos = pos + np.array(delta, dtype=np.int32)
        # Boundary check
        if new_pos[0] < 0 or new_pos[0] >= self.N or new_pos[1] < 0 or new_pos[1] >= self.N:
            return pos.copy()
        # Wall check
        if self._grid[tuple(new_pos)] == WALL:
            return pos.copy()
        return new_pos

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._solved_step = None

        # Sample 2 agents + 2 goals + obstacles (all distinct)
        n_special = 4 + self.num_obstacles  # agents, goals, obstacles
        all_positions = self._sample_distinct_positions(n_special)

        self._pos_a   = all_positions[0]
        self._pos_b   = all_positions[1]
        self._goal_a  = all_positions[2]
        self._goal_b  = all_positions[3]
        obstacles      = all_positions[4:]

        # Build obstacle layer in grid
        self._grid = np.zeros((self.N, self.N), dtype=np.int32)
        for obs_pos in obstacles:
            self._grid[tuple(obs_pos)] = WALL

        # Stamp goals and agents
        self._grid = self._build_grid()

        obs_a = self._get_obs(self._pos_a, is_agent_a=True).flatten()
        obs_b = self._get_obs(self._pos_b, is_agent_a=False).flatten()

        info = {
            "pos_a": self._pos_a.copy(),
            "pos_b": self._pos_b.copy(),
            "goal_a": self._goal_a.copy(),
            "goal_b": self._goal_b.copy(),
            "step": 0,
        }
        return (obs_a, obs_b), info

    def step(
        self,
        action_a: int,
        action_b: int,
    ) -> tuple[tuple[np.ndarray, np.ndarray], float, bool, bool, dict]:
        """
        Apply actions and return (obs_a, obs_b), shared_reward, terminated, truncated, info.

        Note: the env computes base collision/timeout reward components.
        The caller can pass comm_steps for comm_penalty (see reward.py).
        Here we compute environment-observable reward signals only and return
        all signal components in `info` for the trainer to assemble.
        """
        assert self._grid is not None, "Call reset() before step()."
        self._step_count += 1

        # ---- Move agents (ignore walls / boundaries) ----
        # Temporarily clear agent tiles for collision logic
        self._grid[tuple(self._pos_a)] = EMPTY if not (
            np.array_equal(self._pos_a, self._goal_a) or
            np.array_equal(self._pos_a, self._goal_b)
        ) else (GOAL_A if np.array_equal(self._pos_a, self._goal_a) else GOAL_B)
        self._grid[tuple(self._pos_b)] = EMPTY if not (
            np.array_equal(self._pos_b, self._goal_a) or
            np.array_equal(self._pos_b, self._goal_b)
        ) else (GOAL_A if np.array_equal(self._pos_b, self._goal_a) else GOAL_B)

        new_pos_a = self._try_move(self._pos_a, action_a)
        new_pos_b = self._try_move(self._pos_b, action_b)

        # Collision: both agents try to occupy the same cell → neither moves
        collision = np.array_equal(new_pos_a, new_pos_b)
        if collision:
            new_pos_a = self._pos_a.copy()
            new_pos_b = self._pos_b.copy()

        self._pos_a = new_pos_a
        self._pos_b = new_pos_b

        # Re-stamp grid
        self._grid = self._build_grid()

        # ---- Check goal condition ----
        a_at_goal = np.array_equal(self._pos_a, self._goal_a)
        b_at_goal = np.array_equal(self._pos_b, self._goal_b)
        both_at_goal = a_at_goal and b_at_goal

        # ---- Compute reward signals (based on phase) ----
        if self.phase == 1:
            # Single agent (Agent A must reach goal, B is ignored for task success)
            task_reward = 50.0 if a_at_goal else 0.0
            coord_bonus = 0.0
            efficiency_bonus = 5.0 if a_at_goal and self._step_count <= self.max_steps * 0.5 else 0.0
            terminated = bool(a_at_goal)
        elif self.phase == 2:
            # Independent (Each gets reward for reaching their own goal)
            task_reward = (25.0 if a_at_goal else 0.0) + (25.0 if b_at_goal else 0.0)
            coord_bonus = 0.0
            efficiency_bonus = 5.0 if both_at_goal and self._step_count <= self.max_steps * 0.5 else 0.0
            terminated = bool(both_at_goal)
        else:
            # Cooperative (Phase 3)
            task_reward       = 50.0  if both_at_goal else 0.0
            coord_bonus       = 10.0  if both_at_goal and not np.array_equal(self._goal_a, self._goal_b) else 0.0
            efficiency_bonus  = 0.0
            if both_at_goal:
                efficiency_bonus = 5.0 if self._step_count <= self.max_steps * 0.5 else 0.0
            terminated = bool(both_at_goal)

        # Dense shaped reward (BUG 2)
        dist_a = np.linalg.norm(self._pos_a - self._goal_a)
        dist_b = np.linalg.norm(self._pos_b - self._goal_b)
        dense_reward = -0.1 * (dist_a + dist_b)

        collision_penalty = -0.5  if collision else 0.0
        # comm_penalty applied per step in trainer (outside env)
        # timeout_penalty applied at termination in trainer

        total_reward = task_reward + coord_bonus + efficiency_bonus + collision_penalty + dense_reward

        truncated  = bool(self._step_count >= self.max_steps)

        # ---- Observations ----
        obs_a = self._get_obs(self._pos_a, is_agent_a=True).flatten()
        obs_b = self._get_obs(self._pos_b, is_agent_a=False).flatten()

        info = {
            "pos_a": self._pos_a.copy(),
            "pos_b": self._pos_b.copy(),
            "goal_a": self._goal_a.copy(),
            "goal_b": self._goal_b.copy(),
            "collision": collision,
            "a_at_goal": a_at_goal,
            "b_at_goal": b_at_goal,
            "both_at_goal": both_at_goal,
            "step": self._step_count,
            # Individual reward components for logging
            "task_reward": task_reward,
            "coord_bonus": coord_bonus,
            "efficiency_bonus": efficiency_bonus,
            "collision_penalty": collision_penalty,
        }

        return (obs_a, obs_b), total_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering (text-mode for debugging)
    # ------------------------------------------------------------------

    def render(self) -> str:
        assert self._grid is not None, "Call reset() first."
        tile_chars = {
            EMPTY:   ".",
            WALL:    "#",
            GOAL_A:  "G",
            GOAL_B:  "H",
            AGENT_A: "A",
            AGENT_B: "B",
        }
        rows = []
        for r in range(self.N):
            row = ""
            for c in range(self.N):
                tile = self._grid[r, c]
                ch = tile_chars.get(tile, "?")
                # Mark agents on top of goals
                if np.array_equal(self._pos_a, [r, c]):
                    ch = "A"
                elif np.array_equal(self._pos_b, [r, c]):
                    ch = "B"
                row += ch + " "
            rows.append(row)
        return "\n".join(rows)
