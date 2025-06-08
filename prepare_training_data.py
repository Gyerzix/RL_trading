# prepare_training_data.py
import pandas as pd
from trading_env import TradingEnv
from models.dqn import DQNAgent
from mil_reward_model.logger import TrajectoryLogger

df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

env = TradingEnv(df)
agent = DQNAgent(env)
logger = TrajectoryLogger()

for i in range(100):
    obs, _ = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, done, _, _ = env.step(action)
        logger.log_step(obs[-1], action, reward)
        obs = next_obs
    logger.end_episode()
    print(i)

logger.save("trajectories.pkl")
print("Saved trajectories to trajectories.pkl")
