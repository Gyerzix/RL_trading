import pandas as pd
from trading_env import TradingEnv
from models.dqn import DQNAgent
from mil_reward_model.logger import TrajectoryLogger


REWARD_WINDOW = 20

df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

env = TradingEnv(df)
logger = TrajectoryLogger(window_size=REWARD_WINDOW)

for i in range(10):  # эпизоды
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, _, _ = env.step(action)
        logger.log_step(obs[-1], action, reward)
        obs = next_obs
    print(f"Episode {i}")

logger.save("trajectories.pkl")
