import pandas as pd
import matplotlib.pyplot as plt
from trading_env import TradingEnv
from models.q_learning import QLearningAgent
from models.sarsa import SARSAAgent
from models.dqn import DQNAgent
from models.a2c import A2CAgent
from mil_reward_model.reward_wrapper import MILRewardModel

USE_MIL_REWARD = True

# Загрузка и подготовка данных
df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Инициализация среды
REWARD_WINDOW = 20
mil_model = MILRewardModel("mil_reward_model/mil_model.pt",
                           state_dim=7, action_dim=3,
                           reward_window=REWARD_WINDOW) if USE_MIL_REWARD else None
env = TradingEnv(df, mil_reward_model=mil_model)

results = {}

# Q-Learning
# print("Training Q-Learning Agent...")
# q_agent = QLearningAgent(env)
# q_rewards = q_agent.train(episodes=100)
# final_net_worth_q = env.history[-1]['net_worth'] if env.history else env.initial_balance
# results['Q-Learning'] = {'final_net_worth': final_net_worth_q, 'rewards': q_rewards[-1]}

# SARSA
# print("Training SARSA Agent...")
# sarsa_agent = SARSAAgent(env)
# sarsa_rewards = sarsa_agent.train(episodes=100)
# final_net_worth_sarsa = env.history[-1]['net_worth'] if env.history else env.initial_balance
# results['SARSA'] = {'final_net_worth': final_net_worth_sarsa, 'rewards': sarsa_rewards[-1]}

# DQN
print("Training DQN Agent...")
dqn_agent = DQNAgent(env)
dqn_rewards = dqn_agent.train(episodes=10)
final_net_worth_dqn = env.history[-1]['net_worth'] if env.history else env.initial_balance
results['DQN'] = {'final_net_worth': final_net_worth_dqn, 'rewards': dqn_rewards[-1]}

# A2C
# print("Training A2C Agent...")
# a2c_agent = A2CAgent(env)
# a2c_rewards = a2c_agent.train(episodes=100)
# final_net_worth_a2c = env.history[-1]['net_worth'] if env.history else env.initial_balance
# results['A2C'] = {'final_net_worth': final_net_worth_a2c, 'rewards': a2c_rewards[-1]}

# Вывод результатов
print("\nFinal Results:")
for agent, data in results.items():
    print(f"{agent}: Final Net Worth = {data['final_net_worth']:.2f}, Total Reward = {data['rewards']:.2f}")

# Сравнительный график
plt.figure(figsize=(10, 6))
agents = list(results.keys())
final_net_worths = [data['final_net_worth'] for data in results.values()]
plt.bar(agents, final_net_worths, color=['blue', 'green', 'red', 'orange'])
plt.title('Final Net Worth Comparison')
plt.ylabel('Net Worth (USDT)')
plt.xlabel('Agent')
for i, v in enumerate(final_net_worths):
    plt.text(i, v + 50, f'{v:.0f}', ha='center')
plt.tight_layout()
plt.show()