import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        )

        self.mu = nn.Linear(16, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        mu, std = self(state)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        action = action * self.action_bound
        return action, log_prob


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


class SACAgent:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.batch_size = 64

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound).to(device)
        self.q1 = Critic(self.state_dim, self.action_dim).to(device)
        self.q2 = Critic(self.state_dim, self.action_dim).to(device)
        self.q1_target = Critic(self.state_dim, self.action_dim).to(device)
        self.q2_target = Critic(self.state_dim, self.action_dim).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=1e-3)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=1e-3)

        self.buffer = ReplayBuffer(20000)

        self.save_dir = "results/sac"
        os.makedirs(self.save_dir, exist_ok=True)
        self.rewards = []
        

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        s, a, r, s2, d = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            q1_t = self.q1_target(s2, a2)
            q2_t = self.q2_target(s2, a2)
            q_t = torch.min(q1_t, q2_t) - self.alpha * logp2
            target = r + (1 - d) * self.gamma * q_t

        q1_loss = F.mse_loss(self.q1(s, a), target)
        q2_loss = F.mse_loss(self.q2(s, a), target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        a_new, logp = self.actor.sample(s)
        q1_pi = self.q1(s, a_new)
        q2_pi = self.q2(s, a_new)

        actor_loss = (self.alpha * logp - torch.min(q1_pi, q2_pi)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for t, s in zip(self.q1_target.parameters(), self.q1.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)
        for t, s in zip(self.q2_target.parameters(), self.q2.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def train(self, max_episodes=500):
        for ep in range(max_episodes):
            state, _ = self.env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _ = self.actor.sample(state_t)

                action_np = action.cpu().numpy()[0]
                next_state, reward, terminated, truncated, _ = self.env.step(action_np)
                done = terminated or truncated

                train_reward = float(np.clip(reward, -1.0, 1.0))
                self.buffer.add(state, action_np, train_reward, next_state, done)

                self.update()

                state = next_state
                ep_reward += reward

            self.rewards.append(ep_reward)
            np.save(f"{self.save_dir}/rewards.npy", np.array(self.rewards))
            torch.save(self.actor.state_dict(), f"{self.save_dir}/actor.pt")
            torch.save(self.q1.state_dict(), f"{self.save_dir}/q1.pt")
            torch.save(self.q2.state_dict(), f"{self.save_dir}/q2.pt")

            print(f"[EP {ep+1}] Reward: {ep_reward:.2f}")
