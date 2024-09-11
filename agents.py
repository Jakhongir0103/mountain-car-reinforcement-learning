import torch
import torch.nn as nn
import gymnasium as gym

import random
from collections import deque, namedtuple

import matplotlib.pyplot as plt

import numpy as np

# Random Agent defition
class RandomAgent():
    def __init__(self) -> None:
        self.env = gym.make('MountainCar-v0', render_mode="rgb_array")
        self.state = None
        self.done = None

    def render(self):
        return self.env.render()

    def reset(self):
        self.state, _ = self.env.reset(seed=np.random.randint(0,100))
        self.done = False

    def select_action(self):
        return self.env.action_space.sample()

    def observe(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self.done = terminated or truncated
        return next_state, reward
    
# DQN Network
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_hidden=64):
        super(DQN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_observations, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_actions)
        )

    def forward(self, x):
        return self.layer(x)

# Experience Replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminated'))
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, state, action, next_state, reward, terminated):
        self.buffer.append(self.transition(state, action, next_state, reward, terminated))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        state = torch.cat([samples[i].state.unsqueeze(0) for i in range(batch_size)], dim=0).to(self.device)
        action = torch.tensor([samples[i].action.item() for i in range(batch_size)]).unsqueeze(1).to(self.device)
        next = torch.cat([samples[i].next_state.unsqueeze(0) for i in range(batch_size)], dim=0).to(self.device)
        reward = torch.tensor([samples[i].reward.item() for i in range(batch_size)]).to(self.device)
        terminated = torch.tensor([samples[i].terminated for i in range(batch_size)]).to(self.device)
        return state, action, next, reward, terminated

    def __len__(self):
        return len(self.buffer)

# DQN Agent Definition
class DQNAgent():
    def __init__(self, n_episodes, aux_reward=False, rnd_reward=False) -> None:
        self.gamma=0.99
        self.batch_size=64
        self.buffer_size=10000
        self.buffer_size_start=100
        self.epsilon_start=0.9
        self.epsilon_end=0.05
        self.target_update_frequency=20
        self.n_episodes=n_episodes
        self.epsilon_decay = (self.epsilon_end / self.epsilon_start) ** (1 / self.n_episodes)
        self.epsilon=self.epsilon_start
        self.rnd_reward=rnd_reward
        self.rnd_store_size=100
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the buffer
        self.replay_buffer = ReplayMemory(capacity=self.buffer_size)

        # Initialize the environment
        # self.env = gym.make('MountainCar-v0')
        self.env = gym.make('MountainCar-v0', render_mode="rgb_array")
        # self.env = gym.make('MountainCar-v0', render_mode="human")
        self.done = False
        self.terminated = False

        # Get the number of state observations
        state, _ = self.env.reset()
        n_observations = len(state)

        # Initialize the networks
        self.policy_net = DQN(n_observations, self.env.action_space.n).to(self.device)
        self.target_net = DQN(n_observations, self.env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Set loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-4)

        # RND Reward
        if self.rnd_reward:
            # Initialize the RND networks
            torch.manual_seed(0)
            self.rnd_predict_net = DQN(n_observations, 1).to(self.device)
            torch.manual_seed(1000)
            self.rnd_target_net = DQN(n_observations, 1).to(self.device)
            # Store states ans intrinsic reward
            self.rnd_states = deque([], maxlen=self.rnd_store_size)
            self.rnd_rewards = []
            # Reward Factor
            self.reward_factor=0.2 # <= 1 / 5

        # Auxiliary Reward
        if aux_reward:
            self.position_weight = 1 #
            self.velocity_weight = 7 #
        else:
            self.position_weight = 0
            self.velocity_weight = 0

    def render(self, title):
        plt.imshow(self.env.render())
        plt.axis('off')
        plt.title(title)
        plt.show()

    def reset(self):
        state, _ = self.env.reset(seed=np.random.randint(0,100))
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        self.done = False
        self.terminated = False
        return state

    def select_action(self, state, explore=True):
        sample = random.random()
        if sample > self.epsilon or not explore:
            # max action
            with torch.no_grad():
                action =  self.policy_net(state).argmax(dim=-1)
        else:
            # random action
            action = torch.tensor(self.env.action_space.sample(), device=self.device, dtype=torch.long)

        return action

    def update(self):
        if len(self.replay_buffer) < self.buffer_size_start:
            return None

        # Sample transitions
        state, action, next_state, reward, terminated = self.replay_buffer.sample(self.batch_size)

        ### DQN ###
        # Target expected Q values
        with torch.no_grad():
          target_q_values = self.target_net(next_state).max(dim=-1).values.detach()
        expected_q_values = reward + self.gamma * target_q_values * (1 - terminated)
        # Policy Q values
        policy_q_values = self.policy_net(state).gather(dim=1, index=action).squeeze()

        ### RND ###
        if self.rnd_reward:
            # Target and prediction
            rnd_target_pred = self.rnd_target_net(next_state)
            rnd_predict_pred = self.rnd_predict_net(next_state)
            # Compute Loss
            loss = self.criterion(policy_q_values, expected_q_values) + self.criterion(rnd_predict_pred, rnd_target_pred)
            loss_item = loss.item()
        else:
            # Compute Loss
            loss = self.criterion(policy_q_values, expected_q_values)
            loss_item = loss.item()

        # Gradient
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_item

    def calculate_rnd_reward(self, next_state):
        # normalize the next state
        if len(self.rnd_states) == self.rnd_store_size:
          rnd_states_mean = np.array(self.rnd_states).mean(axis=0)
          rnd_states_std = np.array(self.rnd_states).std(axis=0)
          next_state = (next_state - rnd_states_mean) / rnd_states_std

        # calculate the intrinsic reward
        rnd_target_pred = self.rnd_target_net(next_state).detach()
        rnd_predict_pred = self.rnd_predict_net(next_state).detach()
        reward = torch.pow(rnd_predict_pred - rnd_target_pred,2).sum()

        # store if not filled
        if len(self.rnd_rewards) < self.rnd_store_size:
            self.rnd_rewards.append(reward.item())

        # normalize the intrinsic reward
        if len(self.rnd_rewards) >= self.rnd_store_size:
          reward = (reward - np.mean(self.rnd_rewards)) / np.std(self.rnd_rewards)

        reward = reward.clamp(-5.0, 5.0).item()

        reward = np.array(reward, dtype=np.float32)
        return reward

    def observe(self, action):
        next_state, env_reward, terminated, truncated, _ = self.env.step(action.item())
        next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
        env_reward = np.array(env_reward, dtype=np.float32)
        self.done = terminated or truncated
        self.terminated = terminated

        # Auxiliary reward calculation - encourage higher position on the right and positive velocity
        # `position_weight`=0 and `velocity_weight`=0 means no Auxiliary reward
        position, velocity = next_state.cpu().numpy()
        aux_reward = self.position_weight * abs(position + 0.5) + self.velocity_weight * abs(velocity)

        # RND reward
        if self.rnd_reward:
            rnd_reward = self.reward_factor * self.calculate_rnd_reward(next_state)
            return next_state, env_reward, rnd_reward

        return next_state, env_reward, aux_reward
    
class DynaAgent:
    def __init__(self, discr_step=[0.025, 0.005], gamma=0.99, epsilon=0.9, epsilon_min=0.05, decay_rate=0.99, k=5):
        self.discr_step = discr_step
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.k = k
        self.n_states = [int((0.6 - (-1.2)) / self.discr_step[0]) + 1, int((0.07 - (-0.07)) / self.discr_step[1]) + 1]
        self.n_actions = 3  # left, right, do nothing
        self.env = gym.make('MountainCar-v0')

        # Initialize model
        self.P = np.ones((self.n_states[0], self.n_states[1], self.n_actions, self.n_states[0], self.n_states[1])) / (self.n_states[0] * self.n_states[1])
        self.R = np.zeros((self.n_states[0], self.n_states[1], self.n_actions))
        self.Q = np.zeros((self.n_states[0], self.n_states[1], self.n_actions))
        self.visited_states = set()
        self.q_updates = []

    def discretize_state(self, state):
        discrete_state = [min(int((state[0] - (-1.2)) / self.discr_step[0]), self.n_states[0] - 1),
                          min(int((state[1] - (-0.07)) / self.discr_step[1]), self.n_states[1] - 1)]
        return tuple(discrete_state)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        discrete_state = self.discretize_state(state)
        return np.argmax(self.Q[discrete_state])

    def update(self, state, action, reward, next_state):
        s = self.discretize_state(state)
        s_prime = self.discretize_state(next_state)
        self.visited_states.add((s, action))

        # Update transition probabilities
        self.P[s][action][s_prime] += 1
        total_transitions = np.sum(self.P[s][action])
        self.P[s][action] = self.P[s][action] / total_transitions

        # Update rewards
        self.R[s][action] = reward

    def update_q_values(self, state, action):
        s = self.discretize_state(state)
        q_old = self.Q[s][action]
        q_update = self.R[s][action] + self.gamma * np.sum(self.P[s][action] * np.max(self.Q, axis=2))
        self.Q[s][action] = q_update
        return abs(q_old - q_update)

    def planning_step(self):
        for _ in range(self.k):
            s, a = list(self.visited_states)[np.random.choice(len(self.visited_states))]
            self.update_q_values(s, a)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay_rate

    def observe(self, state, action, next_state, reward):
        self.update(state, action, reward, next_state)
        self.q_updates.append(self.update_q_values(state, action))
        self.planning_step()

