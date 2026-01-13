import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super().__init__()
        self.action_bound = action_bound
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.mu = nn.Linear(16, action_dim)
        self.std = nn.Linear(16, action_dim)

    def forward(self, state):  # 마지막 층 
        x = self.net(state)
        mu = torch.tanh(self.mu(x)) * self.action_bound
        std = F.softplus(self.std(x)) + 1e-3
        return mu, std

"""
                Network
                   │
     ┌──────┴──────┐
     │                          │
 μ head                        σ head

"""

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, state):
        return self.net(state)


class A2CAgent:
    def __init__(self, env, lr=1e-3, gamma=0.99, device="cpu"):
        self.env = env
        self.gamma = gamma
        self.device = device
        self.batch_size = 32

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.actor = Actor(
            self.state_dim, self.action_dim, self.action_bound
        ).to(device)

        self.critic = Critic(self.state_dim).to(device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.episode_rewards = []


    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, std = self.actor(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        return action.cpu().numpy()[0]

    def train(self, max_episodes=500):
        for ep in range(max_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            states, actions, rewards, next_states, dones = [], [], [], [], []

            while not done:
                action = self.get_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append([reward])
                next_states.append(next_state)
                dones.append(done)

                state = next_state
                episode_reward += reward

                if len(states) < self.batch_size:
                    continue

                self.update(states, actions, rewards, next_states, dones)
                states, actions, rewards, next_states, dones = [], [], [], [], []

            self.episode_rewards.append(episode_reward)
            print(f"Episode {ep+1}, Reward {episode_reward:.2f}")

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)

        values = self.critic(states)
        targets = torch.zeros_like(values)

        for i in range(len(rewards)):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + self.gamma * self.critic(next_states[i])

        critic_loss = nn.MSELoss()(values, targets.detach())
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        mu, std = self.actor(states)
        dist = torch.distributions.Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)

        advantages = (targets - values).detach()
        actor_loss = -(log_probs * advantages).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

    def plot(self):
        plt.plot(self.episode_rewards)
        plt.show()