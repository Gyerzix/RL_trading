import numpy as np
import random
import pickle

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}

    def _get_state_key(self, state):
        return tuple(state[-1].round(2))  # последние 7 признаков окна

    def select_action(self, state):
        key = self._get_state_key(state)
        if random.random() < self.epsilon or key not in self.q_table:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[key])

    def train(self, episodes=100):
        rewards = []
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                key = self._get_state_key(state)
                if key not in self.q_table:
                    self.q_table[key] = np.zeros(self.env.action_space.n)

                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_key = self._get_state_key(next_state)

                if next_key not in self.q_table:
                    self.q_table[next_key] = np.zeros(self.env.action_space.n)

                # Q-Learning update
                self.q_table[key][action] += self.alpha * (
                    reward + self.gamma * np.max(self.q_table[next_key]) - self.q_table[key][action]
                )

                state = next_state
                total_reward += reward

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            rewards.append(total_reward)
        return rewards
