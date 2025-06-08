import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from csc_lstm_model import CSCInstanceSpaceLSTM
import os
import numpy as np
from tqdm import tqdm

# === 1. Dataset ===

class TrajectoryDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            self.trajectories = pickle.load(f)

        self.max_len = max(len(traj['states']) for traj in self.trajectories)
        self.state_dim = len(self.trajectories[0]['states'][0])
        self.action_dim = 1  # discrete action index, we one-hot encode it

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        states = np.array(traj['states'])  # shape (T, state_dim)
        actions = np.array(traj['actions'])  # shape (T,)

        # One-hot encode actions
        actions_one_hot = np.zeros((len(actions), 3))
        actions_one_hot[np.arange(len(actions)), actions] = 1

        state_action = np.concatenate([states, actions_one_hot], axis=1)  # shape (T, state_dim + 3)

        # Pad to max_len
        pad_len = self.max_len - state_action.shape[0]
        if pad_len > 0:
            pad = np.zeros((pad_len, self.state_dim + 3))
            state_action = np.vstack([state_action, pad])

        return torch.tensor(state_action, dtype=torch.float32), torch.tensor(traj['return'], dtype=torch.float32)


# === 2. Training function ===

def train_mil_model(data_path, save_path="mil_model.pt", epochs=30, batch_size=32, lr=1e-3):
    dataset = TrajectoryDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSCInstanceSpaceLSTM(
        state_dim=dataset.state_dim,
        action_dim=3,  # One-hot for Discrete(3)
        feature_dim=64,
        lstm_hidden_dim=32
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = batch
            x, y = x.to(device), y.to(device)

            rewards, returns = model(x)  # returns: shape (B,)
            loss = criterion(returns, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        print(f"[Epoch {epoch+1}] Loss: {epoch_loss / len(dataset):.4f}")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to: {save_path}")


# === 3. CLI ===

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=r"C:\Users\shakh\Trading_with_RL\trajectories.pkl")
    parser.add_argument("--save_path", type=str, default="mil_model.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    train_mil_model(
        data_path=args.data_path,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
