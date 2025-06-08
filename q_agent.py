import numpy as np
import pickle
from collections import defaultdict
import random


class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filename="q_table.pkl"):
        with open(filename, "rb") as f:
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), pickle.load(f))
