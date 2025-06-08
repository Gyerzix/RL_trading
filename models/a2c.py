import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape), 256),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

class A2CAgent:
    def __init__(self, env, gamma=0.99, lr=1e-3):
        self.env = env
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(env.observation_space.shape, env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.tensor(state[np.newaxis, ...], dtype=torch.float32).to(self.device)
        probs, _ = self.model(state)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    def train(self, episodes=100):
        rewards_all = []
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            log_probs = []
            values = []
            rewards = []

            while not done:
                state_tensor = torch.tensor(state[np.newaxis, ...], dtype=torch.float32).to(self.device)
                probs, value = self.model(state_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

                next_state, reward, done, _, _ = self.env.step(action.item())

                log_probs.append(dist.log_prob(action))
                values.append(value)
                rewards.append(reward)

                state = next_state

            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            values = torch.cat(values).squeeze()
            log_probs = torch.stack(log_probs)

            advantage = returns - values
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            loss = actor_loss + critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            rewards_all.append(sum(rewards))
        return rewards_all
