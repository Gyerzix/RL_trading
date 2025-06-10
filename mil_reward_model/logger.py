import pickle
import os

class TrajectoryLogger:
    def __init__(self, window_size=5):
        self.bags = []
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.window_size = window_size

    def log_step(self, state, action, reward):
        self.buffer_states.append(state)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)

        if len(self.buffer_states) == self.window_size:
            bag = {
                "states": self.buffer_states.copy(),
                "actions": self.buffer_actions.copy(),
                "return": sum(self.buffer_rewards)
            }
            self.bags.append(bag)
            self.buffer_states.pop(0)
            self.buffer_actions.pop(0)
            self.buffer_rewards.pop(0)

    def save(self, path="trajectories.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.bags, f)
        print(f"Saved {len(self.bags)} bags to {path}")
