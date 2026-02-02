import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


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
            nn.ReLU(),
        )

        self.mu = nn.Linear(16, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.net(state)
        mu = torch.tanh(self.mu(x)) * self.action_bound
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std).expand_as(mu)
        return mu, std


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
            nn.Linear(16, 1),
        )

    def forward(self, state):
        return self.net(state)


class PPOAgent:
    def __init__(self, env):
        self.env = env

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.actor_lr = 3e-4
        self.critic_lr = 1e-3
        self.batch_size = 64
        self.epochs = 10

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound).to(device)
        self.critic = Critic(self.state_dim).to(device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.result_dir = "results/ppo"
        os.makedirs(self.result_dir, exist_ok=True)

        self.actor_path = os.path.join(self.result_dir, "actor.pt")
        self.critic_path = os.path.join(self.result_dir, "critic.pt")
        self.reward_path = os.path.join(self.result_dir, "rewards.npy")

        if os.path.exists(self.reward_path):
            self.episode_rewards = list(np.load(self.reward_path))
            print(f"[INFO] Loaded {len(self.episode_rewards)} previous rewards")
        else:
            self.episode_rewards = []

        if os.path.exists(self.actor_path):
            self.actor.load_state_dict(
                torch.load(self.actor_path, map_location=device, weights_only=True)
            )
            print("[INFO] Loaded actor weights")

        if os.path.exists(self.critic_path):
            self.critic.load_state_dict(
                torch.load(self.critic_path, map_location=device, weights_only=True)
            )
            print("[INFO] Loaded critic weights")
            
    def get_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            mu, std = self.actor(state_t)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)

        action = torch.clamp(action, -self.action_bound, self.action_bound)
        return action.cpu().numpy()[0], log_prob.cpu().item()

    def compute_gae(self, rewards, values, dones, next_value):
        gae = 0
        advantages = []
        targets = []

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            targets.insert(0, gae + values[t])
            next_value = values[t]

        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        targets = torch.tensor(targets, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, targets

    def update(self, states, actions, old_log_probs, advantages, targets):
        for _ in range(self.epochs):
            mu, std = self.actor(states)
            dist = torch.distributions.Normal(mu, std)
            log_probs = dist.log_prob(actions).sum(dim=1)

            ratio = torch.exp(log_probs - old_log_probs)
            clipped = torch.clamp(
                ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
            )

            actor_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()

            values = self.critic(states).squeeze(1)
            critic_loss = nn.MSELoss()(values, targets)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_opt.step()

    def save_all(self):
        np.save(self.reward_path, np.array(self.episode_rewards))
        torch.save(self.actor.state_dict(), self.actor_path)
        torch.save(self.critic.state_dict(), self.critic_path)

    def train(self, max_episodes=500):
        for ep in range(max_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

            while not done:
                action, log_prob = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                value = self.critic(
                    torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                ).item()

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(float(done))
                log_probs.append(log_prob)
                values.append(value)

                state = next_state
                episode_reward += reward

            with torch.no_grad():
                next_value = self.critic(
                    torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                ).item()

            states_t = torch.tensor(states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
            old_log_probs_t = torch.tensor(log_probs, dtype=torch.float32, device=device)
            values_t = torch.tensor(values, dtype=torch.float32, device=device)

            advantages, targets = self.compute_gae(
                rewards, values_t, dones, next_value
            )

            self.update(states_t, actions_t, old_log_probs_t, advantages, targets)

            self.episode_rewards.append(episode_reward)
            self.save_all()

            print(f"[Episode {len(self.episode_rewards)}] Reward: {episode_reward:.2f}")

    def plot(self):
        plt.plot(self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()
