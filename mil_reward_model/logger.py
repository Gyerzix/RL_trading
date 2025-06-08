import pickle
import os

class TrajectoryLogger:
    def __init__(self):
        self.trajectories = []
        self.current_states = []
        self.current_actions = []
        self.current_rewards = []

    def log_step(self, state, action, reward):
        self.current_states.append(state)
        self.current_actions.append(action)
        self.current_rewards.append(reward)

    def end_episode(self):
        trajectory_return = sum(self.current_rewards)
        self.trajectories.append({
            "states": self.current_states.copy(),
            "actions": self.current_actions.copy(),
            "return": trajectory_return
        })
        self.current_states.clear()
        self.current_actions.clear()
        self.current_rewards.clear()

    def save(self, path="trajectories.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.trajectories, f)
        print(f"Saved {len(self.trajectories)} trajectories to {path}")
