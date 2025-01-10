# src/train.py
import gymnasium as gym
import numpy as np
from agents.ppo_agent import PPOAgent
import torch
import time
import json
from pathlib import Path

gym.register(id="GridWorld-v0", entry_point="env.grid_world:GridWorldEnv")


def train():
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Initialize log file
    log_file = log_dir / f'training_log_{time.strftime("%Y%m%d-%H%M%S")}.json'
    training_log = []

    # Initialize environment
    env = gym.make("GridWorld-v0")

    # Get dimensions
    state_example = env.reset()[0]
    input_dims = len(np.array(state_example["agent"])) + len(
        np.array(state_example["target"])
    )
    n_actions = env.action_space.n

    # Initialize agent
    agent = PPOAgent(
        input_dims=input_dims, n_actions=n_actions, batch_size=64, lr=0.0003
    )

    n_episodes = 1000
    best_reward = float("-inf")

    for episode in range(n_episodes):
        observation, _ = env.reset()
        done = False
        score = 0

        while not done:
            state = np.concatenate([observation["agent"], observation["target"]])
            action, prob, val = agent.choose_action(state)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.store(state, action, prob, val, reward, done)

            observation = next_observation
            score += reward

            if done:
                agent.learn()

        # Logging
        log_entry = {"episode": episode, "score": score, "timestamp": time.time()}
        training_log.append(log_entry)

        # Save log periodically
        if episode % 10 == 0:
            with open(log_file, "w") as f:
                json.dump(training_log, f)
            print(f"Episode {episode} Score: {score}")

        if score > best_reward:
            best_reward = score
            # Save model
            torch.save(agent.actor.state_dict(), "best_actor.pth")
            torch.save(agent.critic.state_dict(), "best_critic.pth")

    env.close()


if __name__ == "__main__":
    train()
