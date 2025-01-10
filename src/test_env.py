# test_env.py
import gymnasium as gym
from env.grid_world import GridWorldEnv
import time

# Register the environment
gym.register(id="GridWorld-v0", entry_point="env.grid_world:GridWorldEnv")


def main():
    # Create the environment
    env = gym.make("GridWorld-v0", render_mode="human")

    # Reset the environment
    observation, info = env.reset()

    for _ in range(1000):
        # Take a random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(0.1)  # Add delay to make visualization easier to follow

    env.close()


if __name__ == "__main__":
    main()
