import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

from mil_reward_model.reward_wrapper import MILRewardModel

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, commission=0.0005, window_size=20, mil_reward_model=None):
        super(TradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.mil_model = mil_reward_model

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, 7), dtype=np.float32
        )

        self.df['RSI'] = RSIIndicator(self.df['Close'], window=14).rsi()
        self.df['SMA'] = SMAIndicator(self.df['Close'], window=20).sma_indicator()
        self.df = self.df.dropna().reset_index(drop=True)

        self.current_step = 0
        self.balance = initial_balance
        self.position = 0
        self.net_worth = initial_balance
        self.history = []
        self.episode = 0

        self.fig = None
        self.ax = None
        self.ax2 = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.net_worth = self.initial_balance
        self.history = []
        self.episode += 1

        if self.mil_model:
            self.mil_model.reset()

        return self._get_observation(), {}

    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        window = self.df.iloc[start:end][['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'SMA']].values
        return window.astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']

        if self.mil_model:
            reward = self.mil_model.get_reward(self._get_observation()[-1], action)
        else:
            reward = 0
            if action == 1:  # Buy
                if self.balance > 0:
                    btc_to_buy = (self.balance * 0.1 * (1 - self.commission)) / current_price
                    self.position += btc_to_buy
                    self.balance -= btc_to_buy * current_price
                    reward -= self.commission * 100
            elif action == 2:  # Sell
                if self.position > 0:
                    btc_to_sell = self.position * 0.1
                    self.balance += btc_to_sell * current_price * (1 - self.commission)
                    self.position -= btc_to_sell
                    reward -= self.commission * 100

            prev_net_worth = self.net_worth
            self.net_worth = self.balance + self.position * current_price
            reward += (self.net_worth - prev_net_worth) / self.initial_balance * 100
            reward = np.clip(reward, -100, 100)

        self.history.append({
            'step': self.current_step,
            'price': current_price,
            'action': action,
            'net_worth': self.net_worth,
            'reward': reward
        })

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False

        return self._get_observation(), reward, done, truncated, {}

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            self.ax2 = self.ax.twinx()

        self.ax.clear()
        self.ax2.clear()
        self.history = self.history[-len(self.df):]

        prices = [h['price'] for h in self.history]
        steps = [h['step'] for h in self.history]
        actions = [h['action'] for h in self.history]
        net_worths = [h['net_worth'] for h in self.history]

        buy_steps = [s for s, a in zip(steps, actions) if a == 1]
        buy_prices = [p for p, a in zip(prices, actions) if a == 1]
        sell_steps = [s for s, a in zip(steps, actions) if a == 2]
        sell_prices = [p for p, a in zip(prices, actions) if a == 2]

        self.ax.plot(steps, prices, label='Price (Close)', color='blue', alpha=0.7)
        self.ax.scatter(buy_steps, buy_prices, color='green', marker='^', label='Buy', s=100)
        self.ax.scatter(sell_steps, sell_prices, color='red', marker='v', label='Sell', s=100)
        self.ax2.plot(steps, net_worths, label='Net Worth', color='orange', linestyle='--', linewidth=2, alpha=0.7)

        self.ax.set_xlabel('Step', fontsize=12)
        self.ax.set_ylabel('Price (USDT)', fontsize=12, color='blue')
        self.ax2.set_ylabel('Net Worth (USDT)', fontsize=12, color='orange')
        self.ax.set_title(f'Trading Environment - Episode {self.episode}', fontsize=14)

        self.ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        self.ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):d}'))
        self.ax.tick_params(axis='y', labelcolor='blue')
        self.ax2.tick_params(axis='y', labelcolor='orange')
        self.ax.grid(True, linestyle='--', alpha=0.5)

        lines1, labels1 = self.ax.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'trading_plot_episode_{self.episode}.png', dpi=300)
        plt.show(block=True)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
