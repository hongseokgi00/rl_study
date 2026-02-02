import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(s).to(device),
            torch.FloatTensor(a).to(device),
            torch.FloatTensor(r).unsqueeze(1).to(device),
            torch.FloatTensor(s2).to(device),
            torch.FloatTensor(d).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)


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
            nn.ReLU(),
            nn.Linear(16, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state) * self.action_bound


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


class OUNoise:
    def __init__(self, dim, mu=0.0, theta=0.15, sigma=0.2):
        self.dim = dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(dim)

    def reset(self):
        self.state = np.zeros(self.dim)

    def sample(self):
        dx = self.theta * (self.mu - self.state)
        dx += self.sigma * np.random.randn(self.dim)
        self.state += dx
        return self.state


class DDPGAgent:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.action_bound).to(device)

        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.buffer = ReplayBuffer(20000)
        self.noise = OUNoise(self.action_dim)

        self.save_dir = "results/ddpg"
        os.makedirs(self.save_dir, exist_ok=True)
        self.rewards = []

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        s, a, r, s2, d = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            a2 = self.actor_target(s2)
            q_target = self.critic_target(s2, a2)
            y = r + (1 - d) * self.gamma * q_target

        q = self.critic(s, a)
        critic_loss = torch.mean((q - y) ** 2)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for t, s in zip(self.actor_target.parameters(), self.actor.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def train(self, max_episodes=500):
        for ep in range(max_episodes):
            state, _ = self.env.reset()
            self.noise.reset()
            done = False
            ep_reward = 0.0

            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = self.actor(state_t).cpu().numpy()[0]

                action += self.noise.sample()
                action = np.clip(action, -self.action_bound, self.action_bound)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                train_reward = float(np.clip(reward, -1.0, 1.0))
                self.buffer.add(state, action, train_reward, next_state, done)

                self.update()

                state = next_state
                ep_reward += reward

            self.rewards.append(ep_reward)
            np.save(f"{self.save_dir}/rewards.npy", np.array(self.rewards))
            torch.save(self.actor.state_dict(), f"{self.save_dir}/actor.pt")
            torch.save(self.critic.state_dict(), f"{self.save_dir}/critic.pt")

            print(f"[EP {ep+1}] Reward: {ep_reward:.2f}")
