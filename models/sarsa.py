import numpy as np
from collections import defaultdict


class SARSAAgent:
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, bins=10):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.bins = bins
        self.n_actions = env.action_space.n

        # Инициализация Q-таблицы
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

        # Параметры дискретизации
        self.obs_shape = env.observation_space.shape[0]
        self.bin_edges = [np.linspace(-1, 1, bins + 1) for _ in range(self.obs_shape)]

    def _discretize_state(self, state):
        """Дискретизация непрерывного состояния"""
        state = np.clip(state, -1, 1)  # Нормализация для простоты
        discrete_state = []
        for i in range(self.obs_shape):
            bin_idx = np.digitize(state[i], self.bin_edges[i], right=True) - 1
            bin_idx = np.clip(bin_idx, 0, self.bins - 1)
            discrete_state.append(bin_idx)
        return tuple(discrete_state)

    def select_action(self, state):
        discrete_state = self._discretize_state(state)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[discrete_state]))

    def train(self, episodes=100):
        rewards_history = []
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            # Выбираем первое действие
            action = self.select_action(state)

            while not done:
                # Выполняем действие
                next_state, reward, done, _, _ = self.env.step(action)

                # Выбираем следующее действие (on-policy)
                next_action = self.select_action(next_state)

                # Обновляем Q-таблицу
                self._learn(state, action, reward, next_state, next_action, done)

                state = next_state
                action = next_action
                total_reward += reward

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            rewards_history.append(total_reward)
            print(f"Episode: {episode}, Total Reward: {total_reward:.2f}")

        return rewards_history

    def _learn(self, state, action, reward, next_state, next_action, done):
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)

        # Текущая Q-ценность
        q_current = self.q_table[discrete_state][action]

        # Целевая Q-ценность
        q_next = self.q_table[discrete_next_state][next_action] if not done else 0
        q_target = reward + self.gamma * q_next

        # Обновление Q-таблицы
        self.q_table[discrete_state][action] += self.alpha * (q_target - q_current)
