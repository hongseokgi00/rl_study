import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


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

    def forward(self, state):
        x = self.net(state)
        mu = torch.tanh(self.mu(x)) * self.action_bound
        std = torch.nn.functional.softplus(self.std(x))
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
            nn.Linear(16, 1)
        )

    def forward(self, state):
        return self.net(state)


class PPOAgent:
    def __init__(self, env):
        self.env = env

        self.GAMMA = 0.95
        self.GAE_LAMBDA = 0.9
        self.BATCH_SIZE = 32
        self.ACTOR_LR = 1e-4
        self.CRITIC_LR = 1e-3
        self.RATIO_CLIP = 0.05
        self.EPOCHS = 5
        self.std_bound = (1e-2, 1.0)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        self.critic = Critic(self.state_dim)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.ACTOR_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.CRITIC_LR)

        self.save_epi_reward = []

    def log_pdf(self, mu, std, action):
        std = torch.clamp(std, self.std_bound[0], self.std_bound[1])
        var = std.pow(2)
        log_prob = -0.5 * (action - mu).pow(2) / var \
                   - 0.5 * torch.log(var * 2 * np.pi)
        return log_prob.sum(dim=1, keepdim=True)

    def get_action(self, state):
        with torch.no_grad():
            mu, std = self.actor(state)
            std = torch.clamp(std, *self.std_bound)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
        return mu, std, action

    def gae_target(self, rewards, values, next_value, done):
        gae = 0
        gaes = []
        targets = []

        if not done:
            forward_val = next_value
        else:
            forward_val = 0

        for r, v in zip(reversed(rewards), reversed(values)):
            delta = r + self.GAMMA * forward_val - v
            gae = delta + self.GAMMA * self.GAE_LAMBDA * gae
            gaes.insert(0, gae)
            targets.insert(0, gae + v)
            forward_val = v

        return torch.tensor(gaes, dtype=torch.float32), \
               torch.tensor(targets, dtype=torch.float32)

    def actor_learn(self, states, actions, old_log_probs, gaes):
        mu, std = self.actor(states)
        log_probs = self.log_pdf(mu, std, actions)

        ratio = torch.exp(log_probs - old_log_probs)
        clipped = torch.clamp(
            ratio, 1.0 - self.RATIO_CLIP, 1.0 + self.RATIO_CLIP
        )

        loss = -torch.min(ratio * gaes, clipped * gaes).mean()

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

    def critic_learn(self, states, targets):
        values = self.critic(states)
        loss = nn.MSELoss()(values, targets)

        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()

    def train(self, max_episodes):
        for ep in range(max_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            batch_s, batch_a, batch_r, batch_logp = [], [], [], []

            while not done:
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                mu, std, action = self.get_action(state_t)
                action_np = action.numpy()[0]
                action_np = np.clip(action_np, -self.action_bound, self.action_bound)

                next_state, reward, done, _, _ = self.env.step(action_np)
                episode_reward += reward

                logp = self.log_pdf(mu, std, action).detach()

                batch_s.append(state)
                batch_a.append(action_np)
                batch_r.append((reward + 8) / 8)
                batch_logp.append(logp.numpy())

                if len(batch_s) >= self.BATCH_SIZE:
                    states = torch.tensor(batch_s, dtype=torch.float32)
                    actions = torch.tensor(batch_a, dtype=torch.float32)
                    rewards = torch.tensor(batch_r, dtype=torch.float32).unsqueeze(1)
                    old_logp = torch.tensor(batch_logp, dtype=torch.float32).squeeze(1)

                    values = self.critic(states).detach()
                    next_value = self.critic(
                        torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                    ).detach()

                    gaes, targets = self.gae_target(
                        rewards, values, next_value, done
                    )

                    for _ in range(self.EPOCHS):
                        self.actor_learn(states, actions, old_logp, gaes)
                        self.critic_learn(states, targets)

                    batch_s, batch_a, batch_r, batch_logp = [], [], [], []

                state = next_state

            print(f"Episode {ep+1}, Reward: {episode_reward:.2f}")
            self.save_epi_reward.append(episode_reward)

    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
