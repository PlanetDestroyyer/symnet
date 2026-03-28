import numpy as np
import gymnasium as gym

class MiniGridWrapper:
    def __init__(self):
        try:
            import minigrid
        except ImportError:
            pass
        self.env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array")
        self.obs_shape = (3, 64, 64)
        self.max_steps = 1000

    def reset(self):
        obs, info = self.env.reset()
        obs_A = self._process(self.env.render())
        obs_B = self._process(self.env.render())
        new_info = {
            "pos_a": np.zeros(2), "pos_b": np.zeros(2),
            "goal_a": np.zeros(2), "goal_b": np.zeros(2),
            "collision": False, "both_at_goal": False, "step": 0
        }
        return (obs_A, obs_B), new_info

    def step(self, action_A, action_B):
        # Minigrid actions: 0=left, 1=right, 2=forward
        obs, reward, terminated, truncated, info = self.env.step(action_A % 3)
        obs_A = self._process(self.env.render())
        obs_B = self._process(self.env.render())
        new_info = {
            "pos_a": np.zeros(2), "pos_b": np.zeros(2),
            "goal_a": np.zeros(2), "goal_b": np.zeros(2),
            "collision": False, "both_at_goal": terminated, "step": 0
        }
        return (obs_A, obs_B), float(reward), terminated, truncated, new_info

    def _process(self, obs):
        import cv2
        obs = cv2.resize(obs, (64, 64))
        return obs.transpose(2, 0, 1) / 255.0
