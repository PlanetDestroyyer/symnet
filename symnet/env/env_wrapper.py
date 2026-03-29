import numpy as np
import os
import sys

def auto_patch_minedojo():
    """
    Automatically patches MineDojo's broken Gradle dependencies,
    sets up Java 8 environment, and starts a virtual display for Kaggle/Colab.
    """
    # 0. Start Virtual Display (required for Minecraft)
    try:
        from pyvirtualdisplay import Display
        if not any(isinstance(v, Display) for v in globals().values()):
            display = Display(visible=0, size=(640, 480))
            display.start()
            print("[SymNet] Started virtual display for MineDojo")
    except Exception as e:
        print(f"[SymNet] Could not start virtual display: {e}")

    try:
        import minedojo
    except ImportError:
        return

    # 1. Patch build.gradle for the MixinGradle error
    minedojo_path = minedojo.__path__[0]
    gradle_path = os.path.join(minedojo_path, "sim/Malmo/Minecraft/build.gradle")

    if os.path.exists(gradle_path):
        with open(gradle_path, "r") as f:
            content = f.read()

        repo_str = "maven { url 'https://repo.spongepowered.org/repository/maven-public/' }"
        if repo_str not in content:
            content = content.replace("repositories {", f"repositories {{\n        {repo_str}")

        old_dep = "com.github.SpongePowered:MixinGradle:dcfaf61"
        new_dep = "org.spongepowered:mixingradle:0.6-SNAPSHOT"
        
        if old_dep in content:
            content = content.replace(old_dep, new_dep)
            with open(gradle_path, "w") as f:
                f.write(content)
            print(f"[SymNet] Patched MineDojo build.gradle at {gradle_path}")

    # 2. Set JAVA_HOME if Java 8 is found (standard Kaggle/Colab path)
    java8_path = "/usr/lib/jvm/java-8-openjdk-amd64"
    if os.path.exists(java8_path):
        os.environ["JAVA_HOME"] = java8_path
        print(f"[SymNet] Set JAVA_HOME to {java8_path}")

class MineDojoWrapper:
    """
    Drop-in replacement for GridWorld.
    Returns same interface: obs_A, obs_B, reward, done
    """
    def __init__(self, task="survival"):
        auto_patch_minedojo()
        try:
            import minedojo
        except ImportError:
            raise ImportError("Please install minedojo: pip install minedojo")
            
        # Use simulation directly to avoid 'make()' wrapper assertions
        from minedojo.sim import MineDojoSim
        self.env = MineDojoSim(
            image_size=(64, 64),
            event_level_control=True,
            generate_world_type="default"
        )
        self.obs_shape = (3, 64, 64)
        self.max_steps = 1000

    def reset(self):
        obs = self.env.reset()
        # Split single agent obs into A and B views
        obs_A = self._process(obs)
        obs_B = self._process(obs)  # TODO: second agent
        
        info = {
            "pos_a": np.zeros(2), "pos_b": np.zeros(2),
            "goal_a": np.zeros(2), "goal_b": np.zeros(2),
            "collision": False, "both_at_goal": False, "step": 0
        }
        return (obs_A, obs_B), info

    def step(self, action_A, action_B):
        # Minedojo expects specific action format.
        md_action = self.env.action_space.no_op()
        if action_A == 0: md_action[0] = 1 # forward
        elif action_A == 1: md_action[0] = 2 # back
        elif action_A == 2: md_action[1] = 1 # left
        elif action_A == 3: md_action[1] = 2 # right
        
        obs, reward, done, info = self.env.step(md_action)
        
        obs_A = self._process(obs)
        obs_B = self._process(obs)
        
        new_info = {
            "pos_a": np.zeros(2), "pos_b": np.zeros(2),
            "goal_a": np.zeros(2), "goal_b": np.zeros(2),
            "collision": False, "both_at_goal": done, "step": 0
        }
        
        return (obs_A, obs_B), float(reward), done, False, new_info

    def _process(self, obs):
        # normalize pixels to [0,1]
        return obs['rgb'].transpose(2,0,1) / 255.0
