# src/agents/ppo_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]
        return batches


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs


class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class PPOAgent:
    def __init__(
        self,
        input_dims,
        n_actions,
        lr=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(input_dims, n_actions)
        self.critic = CriticNetwork(input_dims)
        self.memory = PPOMemory(batch_size)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def choose_action(self, observation):
        state = torch.FloatTensor(observation).unsqueeze(0)

        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()

        value = self.critic(state)

        return action.item(), probs[0][action.item()].item(), value.item()

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr = torch.FloatTensor(np.array(self.memory.states))
            action_arr = torch.LongTensor(np.array(self.memory.actions))
            old_prob_arr = torch.FloatTensor(np.array(self.memory.probs))
            vals_arr = torch.FloatTensor(np.array(self.memory.vals))

            # Compute advantages
            rewards = np.array(self.memory.rewards)
            dones = np.array(self.memory.dones)
            advantages = np.zeros(len(rewards))

            for t in range(len(rewards) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards) - 1):
                    a_t += discount * (
                        rewards[k]
                        + self.gamma * vals_arr[k + 1] * (1 - dones[k])
                        - vals_arr[k]
                    )
                    discount *= self.gamma * self.gae_lambda
                advantages[t] = a_t

            advantages = torch.FloatTensor(advantages)

            # Get new action probabilities
            probs = self.actor(state_arr)
            dist = Categorical(probs)
            new_probs = dist.log_prob(action_arr)

            # Compute ratio
            prob_ratio = torch.exp(new_probs - torch.log(old_prob_arr))

            # Compute PPO loss
            weighted_probs = advantages * prob_ratio
            weighted_clipped_probs = (
                torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                * advantages
            )
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

            # Compute value loss
            returns = advantages + vals_arr
            critic_value = self.critic(state_arr)
            critic_loss = F.mse_loss(critic_value.squeeze(), returns)

            # Update networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        self.memory.clear()
