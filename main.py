import pandas as pd
from trading_env import TradingEnv
from q_agent import QLearningAgent

df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

env = TradingEnv(df)
agent = QLearningAgent(actions=[0, 1, 2])  # Hold, Buy, Sell

episodes = 50

for episode in range(episodes):
    obs, _ = env.reset()
    state = env.get_discrete_state(obs)
    total_reward = 0
    done = False

    for i in range(500):
        action = agent.act(state)
        next_obs, reward, done, truncated, info = env.step(action)
        next_state = env.get_discrete_state(next_obs)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    agent.update_epsilon()
    print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

# Последний прогон с визуализацией
obs, _ = env.reset()
state = env.get_discrete_state(obs)
done = False

for i in range(100):
    action = agent.act(state)
    next_obs, reward, done, _, _ = env.step(action)
    next_state = env.get_discrete_state(next_obs)
    state = next_state

env.render()