import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


class A2CAgent:
    def __init__(self, env, lr=3e-4, gamma=0.99, device="cpu"):
        self.env = env
        self.gamma = gamma
        self.device = device
        self.batch_size = 64

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound).to(device)
        self.critic = Critic(self.state_dim).to(device)

        self.actor_opt = optim.Adam(
            [
                {"params": self.actor.net.parameters()},
                {"params": self.actor.mu.parameters()},
                {"params": [self.actor.log_std], "lr": lr * 0.1},
            ],
            lr=lr,
        )
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        ############################################################################# 

        self.result_dir = "results/a2c"
        os.makedirs(self.result_dir, exist_ok=True)

        self.reward_path = os.path.join(self.result_dir, "rewards.npy")
        self.actor_path = os.path.join(self.result_dir, "actor.pt")
        self.critic_path = os.path.join(self.result_dir, "critic.pt")

        if os.path.exists(self.actor_path):
            self.actor.load_state_dict(
                torch.load(self.actor_path, map_location=self.device, weights_only=True)
            )
            if self._has_nonfinite(self.actor):
                self.actor.apply(self._init_weights)
                with torch.no_grad():
                    self.actor.log_std.fill_(0.0)
            else:
                print("[INFO] Loaded actor weights")

        if os.path.exists(self.critic_path):
            self.critic.load_state_dict(
                torch.load(self.critic_path, map_location=self.device, weights_only=True)
            )
            if self._has_nonfinite(self.critic):
                self.critic.apply(self._init_weights)
            else:
                print("[INFO] Loaded critic weights")
        #####################################################################################


    def _has_nonfinite(self, model):
        for p in model.parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                return True
        return False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    def get_action(self, state):
        if not np.isfinite(state).all():
            state, _ = self.env.reset()

        state_t = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            mu, std = self.actor(state_t)

        if not torch.isfinite(mu).all() or not torch.isfinite(std).all():
            raise RuntimeError("Actor produced NaN/Inf")

        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        action = torch.clamp(action, -self.action_bound, self.action_bound)
        return action.cpu().numpy()[0]

    def train(self, max_episodes=500):
        for _ in range(max_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            cur_step = 0

            states, actions, rewards, next_states, dones = [], [], [], [], []

            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                cur_step += 1

                if not np.isfinite(next_state).all():
                    done = True
                    break

                reward = float(np.clip(reward, -1.0, 1.0))

                if getattr(self.env, "render_mode", None) == "human":
                    time.sleep(0.03)

                states.append(state)
                actions.append(action)
                rewards.append([reward])
                next_states.append(next_state)
                dones.append(done)

                state = next_state
                episode_reward += reward

                if len(states) >= self.batch_size:
                    self.update(states, actions, rewards, next_states, dones)
                    states, actions, rewards, next_states, dones = [], [], [], [], []

            if len(states) > 0:
                self.update(states, actions, rewards, next_states, dones)

            self.episode_rewards.append(episode_reward)
            self.save_all()

            print(f"[Episode {len(self.episode_rewards)}] Reward: {episode_reward:.2f} | Step: {cur_step}")

    def update(self, states, actions, rewards, next_states, dones):
        if (
            not np.isfinite(np.asarray(states)).all()
            or not np.isfinite(np.asarray(actions)).all()
            or not np.isfinite(np.asarray(rewards)).all()
            or not np.isfinite(np.asarray(next_states)).all()
        ):
            return

        states_t = torch.from_numpy(np.asarray(states)).float().to(self.device)
        actions_t = torch.from_numpy(np.asarray(actions)).float().to(self.device)
        rewards_t = torch.from_numpy(np.asarray(rewards)).float().to(self.device)
        next_states_t = torch.from_numpy(np.asarray(next_states)).float().to(self.device)
        dones_t = torch.from_numpy(np.asarray(dones)).float().unsqueeze(1).to(self.device)

        values = self.critic(states_t)

        with torch.no_grad():
            next_values = self.critic(next_states_t)
            targets = rewards_t + self.gamma * next_values * (1.0 - dones_t)
            targets = torch.clamp(targets, -10.0, 10.0)

        critic_loss = F.mse_loss(values, targets)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_opt.step()

        mu, std = self.actor(states_t)
        if not torch.isfinite(mu).all() or not torch.isfinite(std).all():
            return

        dist = torch.distributions.Normal(mu, std)
        log_probs = dist.log_prob(actions_t).sum(dim=1, keepdim=True)

        advantages = targets - values
        if advantages.numel() < 2:
            return

        adv_std = advantages.std(unbiased=False)
        if not torch.isfinite(adv_std) or adv_std < 1e-6:
            adv_std = torch.tensor(1.0, device=self.device)

        advantages = (advantages - advantages.mean()) / adv_std
        advantages = advantages.detach()

        actor_loss = -(log_probs * advantages).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_opt.step()

        with torch.no_grad():
            self.actor.log_std.clamp_(-5.0, 2.0)

    def save_all(self):
        np.save(self.reward_path, np.array(self.episode_rewards))
        torch.save(self.actor.state_dict(), self.actor_path)
        torch.save(self.critic.state_dict(), self.critic_path)
