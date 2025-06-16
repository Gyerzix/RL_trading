import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        # Общая часть сети для актера и критика
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # Актер: вероятности действий
        self.actor = nn.Sequential(
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
        # Критик: оценка ценности состояния
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        shared_features = self.shared(x)
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_probs, value


class A2CAgent:
    def __init__(self, env, gamma=0.99, lr=1e-3, entropy_coef=0.01):
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.model = ActorCritic(obs_shape, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Для совместимости с save_model/load_model
        self.epsilon = None  # Заглушка, так как A2C не использует epsilon

    def select_action(self, state):
        state = torch.tensor(state[np.newaxis, ...], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.model(state)
        action_probs = action_probs.cpu().numpy()[0]
        action = np.random.choice(self.n_actions, p=action_probs)
        return action

    def train(self, episodes=100):
        rewards_history = []
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            states, actions, rewards, next_states, dones = [], [], [], [], []

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state
                total_reward += reward

                # Обновление модели после каждого шага (online A2C)
                self._learn(states, actions, rewards, next_states, dones)
                states, actions, rewards, next_states, dones = [], [], [], [], []

            rewards_history.append(total_reward)
            print(f"Episode: {episode}, Total Reward: {total_reward:.2f}")

        return rewards_history

    def _learn(self, states, actions, rewards, next_states, dones):
        if not states:
            return

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Получаем текущие вероятности действий и ценности
        action_probs, values = self.model(states)
        _, next_values = self.model(next_states)

        # Вычисляем преимущества (advantage)
        advantages = rewards + self.gamma * next_values.squeeze() * (1 - dones) - values.squeeze()

        # Потеря актера
        log_probs = torch.log(action_probs + 1e-10)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        actor_loss = -(action_log_probs * advantages.detach()).mean()

        # Потеря критика
        critic_loss = advantages.pow(2).mean()

        # Энтропия для исследования
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=1).mean()

        # Общая потеря
        loss = actor_loss + critic_loss - self.entropy_coef * entropy

        # Обновление модели
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
