"""
Reward module for SymNet.

All reward constants live here so they can be imported by both the environment
and the trainer without circular dependencies.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reward constants (spec §05)
# ---------------------------------------------------------------------------
TASK_REWARD       =  50.0   # both agents reach their assigned goals
COORD_BONUS       =   3.0   # awarded alongside TASK_REWARD (goals distinct)
EFFICIENCY_BONUS  =   1.0   # solved in < 50% of max_steps
COMM_PENALTY      =  -0.01  # per communication step (applied by trainer)
COLLISION_PENALTY =  -0.5   # agents occupy same cell
TIMEOUT_PENALTY   =  -1.0   # episode terminated by truncation


def compute_step_reward(
    both_at_goal: bool,
    collision: bool,
    step: int,
    max_steps: int,
    comm_steps: int = 1,
) -> tuple[float, dict[str, float]]:
    """
    Compute the full shared reward for one step.

    Parameters
    ----------
    both_at_goal : bool  – True if both agents are at their respective goals.
    collision    : bool  – True if agents tried to occupy the same cell.
    step         : int   – Current episode step count.
    max_steps    : int   – Episode length limit.
    comm_steps   : int   – Number of comm vectors exchanged this step (usually 1).

    Returns
    -------
    total : float
    breakdown : dict   – Individual component values (for logging).
    """
    task        = TASK_REWARD      if both_at_goal else 0.0
    coord       = COORD_BONUS      if both_at_goal else 0.0
    efficiency  = EFFICIENCY_BONUS if both_at_goal and step <= max_steps * 0.5 else 0.0
    comm        = COMM_PENALTY * comm_steps
    collision   = COLLISION_PENALTY if collision else 0.0
    # timeout penalty is added by the trainer when truncated
    total = task + coord + efficiency + comm + collision
    return total, {
        "task":       task,
        "coord":      coord,
        "efficiency": efficiency,
        "comm":       comm,
        "collision":  collision,
    }
