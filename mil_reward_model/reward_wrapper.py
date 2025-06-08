import torch
import numpy as np
from mil_reward_model.csc_lstm_model import CSCInstanceSpaceLSTM

class MILRewardModel:
    def __init__(self, model_path, state_dim, action_dim=3, device="cpu"):
        self.device = torch.device(device)
        self.model = CSCInstanceSpaceLSTM(
            state_dim=state_dim,
            action_dim=action_dim,
            feature_dim=64,
            lstm_hidden_dim=32
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.hidden = None

    def reset(self):
        self.hidden = None  # Reset hidden state at start of episode

    def get_reward(self, state, action):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, state_dim)
        action_one_hot = torch.zeros(1, 3).to(self.device)
        action_one_hot[0, action] = 1
        input_vec = torch.cat([state, action_one_hot], dim=1).unsqueeze(1)  # (1, 1, state_dim + 3)

        with torch.no_grad():
            feature = self.model.feature_extractor(input_vec[:, 0])  # (1, F)
            lstm_out, self.hidden = self.model.lstm(feature.unsqueeze(1), self.hidden)  # (1, 1, H)
            concat = torch.cat([lstm_out.squeeze(1), feature], dim=1)  # (1, H+F)
            reward = self.model.head(concat).squeeze().item()  # scalar

        return reward
