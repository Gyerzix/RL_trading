import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.net(x)


class CSCInstanceSpaceLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim=64, lstm_hidden_dim=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # Feature extractor for state-action pairs
        self.feature_extractor = FeatureExtractor(state_dim + action_dim, feature_dim)

        # LSTM for temporal hidden state
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=lstm_hidden_dim, batch_first=True)

        self.lstm_norm = nn.LayerNorm(lstm_hidden_dim)

        # Head network for reward prediction
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_dim + feature_dim, 128),  # Увеличен размер
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state_action_seq):
        """
        state_action_seq: tensor of shape (batch_size, seq_len, state_dim + action_dim)
        """
        B, T, D = state_action_seq.shape

        # Extract features
        features = self.feature_extractor(state_action_seq.view(B * T, D))  # (B*T, F)
        features = features.view(B, T, self.feature_dim)  # (B, T, F)

        # LSTM hidden states
        lstm_out, _ = self.lstm(features)  # (B, T, H)

        # Concatenate skip connections
        concat = torch.cat([lstm_out, features], dim=-1)  # (B, T, H+F)

        # Predict rewards
        rewards = self.head(concat).squeeze(-1)  # (B, T)

        # Return both timestep rewards and their sum (total return)
        return rewards, rewards.sum(dim=1), lstm_out  # (B, T), (B,)
