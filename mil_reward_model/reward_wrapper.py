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
        self.buffer = []  # хранит (state, action)

    def reset(self):
        self.buffer = []

    def get_reward(self, state, action):
        self.buffer.append((state, action))
        if len(self.buffer) < self.window:
            return 0.0  # Нет награды до тех пор, пока не заполнилось окно

        # Подготовим batch из одного bag
        state_actions = []
        for s, a in self.buffer:
            a_onehot = np.zeros(3)
            a_onehot[a] = 1
            state_actions.append(np.concatenate([s, a_onehot]))

        x = torch.tensor(np.array([state_actions]), dtype=torch.float32).to(self.device)  # (1, window, D)
        with torch.no_grad():
            _, return_pred = self.model(x)
        reward = return_pred.item()
        self.buffer.pop(0)  # сдвигаем окно
        return reward

