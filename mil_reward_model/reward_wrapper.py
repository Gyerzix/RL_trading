import torch
import numpy as np
from mil_reward_model.csc_lstm_model import CSCInstanceSpaceLSTM

class MILRewardModel:
    def __init__(self, model_path, state_dim, action_dim=3, device="cpu", reward_window=5):
        self.device = torch.device(device)
        self.model = CSCInstanceSpaceLSTM(
            state_dim=state_dim,
            action_dim=action_dim,
            feature_dim=64,
            lstm_hidden_dim=32
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.window = reward_window
        self.buffer = []
        self.hidden_state = np.zeros(32)  # размер скрытого состояния

    def reset(self):
        self.buffer = []
        self.hidden_state = np.zeros(32)

    def get_reward(self, state, action):
        self.buffer.append((state, action))
        if len(self.buffer) < self.window:
            return 0.0

        state_actions = []
        for s, a in self.buffer:
            a_onehot = np.zeros(3)
            a_onehot[a] = 1
            state_actions.append(np.concatenate([s, a_onehot]))

        x = torch.tensor(np.array([state_actions]), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, return_pred, lstm_out = self.model(x)
        self.hidden_state = lstm_out[:, -1, :].squeeze(0).cpu().numpy()
        self.buffer.pop(0)
        return return_pred.item()

    def get_hidden_state(self):
        return self.hidden_state

