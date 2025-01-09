

# Guide to PettingZoo + Gymnasium Implementation

## 1. Basic PettingZoo Structure

PettingZoo extends Gymnasium for multi-agent environments. Here's the core structure:

```python
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np

class GridWorldEnv(ParallelEnv):
    def __init__(self, config):
        super().__init__()
        # List of agent ids
        self.agents = [f"agent_{i}" for i in range(config['n_agents'])]
        self.grid_size = config['grid_size']
        
        # Define spaces for each agent
        self.action_spaces = {
            agent: spaces.Discrete(4)  # Up, Down, Left, Right
            for agent in self.agents
        }
        
        self.observation_spaces = {
            agent: spaces.Dict({
                "position": spaces.Box(0, self.grid_size, shape=(2,), dtype=int),
                "local_view": spaces.Box(0, 1, shape=(config['view_size'], config['view_size'], 3)),
            })
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        self.agents = self.possible_agents[:]
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        """Execute one step for all agents"""
        # Process all agents' actions simultaneously
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        for agent in self.agents:
            # Update state based on actions
            self._process_action(agent, actions[agent])
            
            # Collect results for each agent
            observations[agent] = self._get_obs(agent)
            rewards[agent] = self._get_reward(agent)
            terminations[agent] = self._get_termination(agent)
            truncations[agent] = self._get_truncation(agent)
            infos[agent] = {}
            
        return observations, rewards, terminations, truncations, infos
```

## 2. Key Components for Our Grid World

### State Management
```python
class GridState:
    def __init__(self, size):
        self.grid = np.zeros((size, size, 3))  # Channels: agents, obstacles, goals
        self.agent_positions = {}
        self.goals = []
        self.obstacles = []

    def update_agent_position(self, agent_id, new_pos):
        old_pos = self.agent_positions.get(agent_id)
        if old_pos is not None:
            self.grid[old_pos[0], old_pos[1], 0] = 0
        self.grid[new_pos[0], new_pos[1], 0] = 1
        self.agent_positions[agent_id] = new_pos
```

### Observation Generation
```python
def _get_obs(self, agent_id):
    pos = self.state.agent_positions[agent_id]
    view_size = self.config['view_size']
    
    # Get local view around agent
    local_view = self._get_local_view(pos, view_size)
    
    return {
        "position": np.array(pos),
        "local_view": local_view,
    }

def _get_local_view(self, pos, size):
    # Extract a square view around the agent's position
    x, y = pos
    pad = size // 2
    padded_grid = np.pad(self.state.grid, ((pad, pad), (pad, pad), (0, 0)))
    return padded_grid[x:x+size, y:y+size]
```

## 3. Modular Agent Architecture with PPO

The modular architecture breaks down the agent into distinct components:

```python
class ModularPPOAgent(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        
        # Visual processing
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # State processing
        self.state_encoder = nn.Sequential(
            nn.Linear(2, 64),  # Position encoding
            nn.ReLU()
        )
        
        # Combined processing
        self.shared = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU()
        )
        
        # Policy (action) head
        self.policy = nn.Sequential(
            nn.Linear(128, action_space.n),
            nn.Softmax(dim=-1)
        )
        
        # Value head
        self.value = nn.Sequential(
            nn.Linear(128, 1)
        )
```

## 4. Communication Module

The communication module enables agents to share information:

```python
class AgentCommunication(nn.Module):
    def __init__(self, state_size, message_size):
        super().__init__()
        
        # Message generation
        self.message_encoder = nn.Sequential(
            nn.Linear(state_size, message_size),
            nn.Tanh()  # Bounded messages
        )
        
        # Message processing
        self.attention = nn.MultiheadAttention(
            embed_dim=message_size,
            num_heads=4
        )
        
        # Message interpretation
        self.message_decoder = nn.Sequential(
            nn.Linear(message_size, state_size),
            nn.ReLU()
        )
    
    def forward(self, agent_states, other_messages):
        # Generate message from current state
        my_message = self.message_encoder(agent_states)
        
        # Process all messages using attention
        processed_messages, _ = self.attention(
            my_message, other_messages, other_messages
        )
        
        # Interpret messages
        message_effect = self.message_decoder(processed_messages)
        return message_effect
```

## 5. Curriculum Learning

Curriculum learning is a training strategy where you start with easier tasks and gradually increase difficulty. Think of it like teaching a child math - you start with addition before moving to multiplication.

Example implementation:

```python
class CurriculumManager:
    def __init__(self):
        self.current_level = 0
        self.success_threshold = 0.8
        self.evaluation_window = 100
        self.recent_successes = []
        
        self.curricula = [
            {"grid_size": 5, "n_obstacles": 2, "n_goals": 1},    # Level 0
            {"grid_size": 8, "n_obstacles": 4, "n_goals": 1},    # Level 1
            {"grid_size": 10, "n_obstacles": 6, "n_goals": 2},   # Level 2
            {"grid_size": 10, "n_obstacles": 8, "n_goals": 2, "moving_obstacles": True},  # Level 3
        ]
    
    def get_current_config(self):
        return self.curricula[self.current_level]
    
    def update(self, episode_success):
        # Track recent performance
        self.recent_successes.append(episode_success)
        if len(self.recent_successes) > self.evaluation_window:
            self.recent_successes.pop(0)
            
        # Check if ready to advance
        if len(self.recent_successes) == self.evaluation_window:
            success_rate = sum(self.recent_successes) / self.evaluation_window
            if success_rate >= self.success_threshold:
                self.current_level = min(self.current_level + 1, len(self.curricula) - 1)
                self.recent_successes = []
```

Benefits of curriculum learning:
1. Faster initial learning
2. Better final performance
3. More stable training
4. Reduced likelihood of getting stuck in local optima

Would you like me to elaborate on any of these components or provide more specific implementation details?
