import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trading_env import TradingEnv
from mil_reward_model.reward_wrapper import MILRewardModel
from models.sarsa import SARSAAgent
from models.dqn import DQNAgent
from models.a2c import A2CAgent


MODELS_DIR = "saved_models"
RESULTS_DIR = "results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

INITIAL_BALANCE = 10000
REWARD_WINDOW = 20
EPISODES = 1

def save_model(agent, filename, agent_type):
    """Сохраняет модель агента и дополнительные параметры"""
    path = os.path.join(MODELS_DIR, filename)
    save_dict = {'agent_type': agent_type}

    if agent_type == 'sarsa':
        save_dict['q_table'] = agent.q_table
        save_dict['epsilon'] = agent.epsilon
    else:  # Для DQN, A2C
        save_dict['model_state_dict'] = agent.model.state_dict()
        save_dict['optimizer_state_dict'] = agent.optimizer.state_dict()
        save_dict['epsilon'] = getattr(agent, 'epsilon', None)

    torch.save(save_dict, path)
    print(f"Model saved to {path}")

def load_model(agent, filename):
    """Загружает модель и дополнительные параметры для агента"""
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        agent_type = checkpoint['agent_type']

        if agent_type == 'sarsa':
            agent.q_table = checkpoint['q_table']
            agent.epsilon = checkpoint['epsilon']
        else:
            agent.model.load_state_dict(checkpoint['model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['epsilon'] is not None:
                agent.epsilon = checkpoint['epsilon']

        print(f"Model loaded from {path}")
    return agent

def calculate_metrics(history):
    """Вычисляет метрики производительности"""
    returns = np.diff([h['net_worth'] for h in history])
    total_return = (history[-1]['net_worth'] - INITIAL_BALANCE) / INITIAL_BALANCE

    equity = [h['net_worth'] for h in history]
    max_drawdown = 0
    peak = equity[0]
    for x in equity:
        if x > peak:
            peak = x
        drawdown = (peak - x) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'avg_daily_return': np.mean(returns) if len(returns) > 0 else 0,
        'volatility': np.std(returns) if len(returns) > 0 else 0
    }

def plot_results(env, results, agent_name):
    """Визуализация результатов торговли"""
    env.render()

    history = results['history']
    metrics = results['metrics']

    fig, ax = plt.subplots(figsize=(12, 4))
    steps = [h['step'] for h in history]
    rewards = [h['reward'] for h in history]

    ax.bar(steps, rewards, color=np.where(np.array(rewards) > 0, 'g', 'r'), alpha=0.7)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title(f'Rewards Distribution ({agent_name})', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{agent_name}_trading_rewards.png'), dpi=300)
    plt.show()

def train_and_evaluate_agent(
    df,
    agent_class,
    agent_params=None,
    episodes=EPISODES,
    use_mil_reward=False,
    mil_model_path="mil_model.pt",
    initial_balance=INITIAL_BALANCE,
    reward_window=REWARD_WINDOW,
    model_filename=None,
    load_existing_model=False,
    agent_name=None
):
    """
    Универсальная функция для обучения и оценки агента.

    Args:
        df (pd.DataFrame): Данные для среды (OHLCV).
        agent_class: Класс агента (например, SARSAAgent, DQNAgent, A2CAgent).
        agent_params (dict): Параметры для инициализации агента.
        episodes (int): Количество эпизодов обучения.
        use_mil_reward (bool): Использовать MIL-награду (True) или марковскую (False).
        mil_model_path (str): Путь к MIL-модели.
        initial_balance (float): Начальный баланс.
        reward_window (int): Размер окна для MIL-награды.
        model_filename (str): Имя файла для сохранения/загрузки модели.
        load_existing_model (bool): Загружать существующую модель.
        agent_name (str): Имя агента для логирования и визуализации.

    Returns:
        dict: Результаты (history, rewards, metrics).
    """
    df = df.rename(columns={
        'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close',
        'volume': 'Volume'
    })
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    mil_model = None
    if use_mil_reward:
        mil_model = MILRewardModel(
            mil_model_path,
            state_dim=7,  # Open, High, Low, Close, Volume, RSI, SMA
            action_dim=3,
            reward_window=reward_window
        )

    env = TradingEnv(
        df,
        initial_balance=initial_balance,
        mil_reward_model=mil_model
    )
    obs, _ = env.reset()
    print(f"[DEBUG] Observation shape: {obs.shape}")

    # Инициализация агента
    agent_params = agent_params or {}
    agent = agent_class(env, **agent_params)

    # Определение типа агента и имени файла модели
    agent_type = agent_name.lower() if agent_name else agent_class.__name__.lower()
    model_filename = model_filename or f"{agent_type}_model.pt"

    # Загрузка модели, если требуется
    if load_existing_model:
        agent = load_model(agent, model_filename)

    # Обучение агента
    print(f"Training {agent_name or agent_class.__name__}...")
    rewards_history = agent.train(episodes=episodes)

    # Сохранение модели
    # save_model(agent, model_filename, agent_type)

    results = {
        'history': env.history.copy(),
        'rewards': rewards_history,
        'metrics': calculate_metrics(env.history)
    }

    plot_results(env, results, agent_type)

    print(f"\n=== {agent_name or agent_class.__name__} Performance ===")
    print(f"Final Net Worth: {results['history'][-1]['net_worth']:.2f}")
    print(f"Total Return: {results['metrics']['total_return'] * 100:.2f}%")
    print(f"Max Drawdown: {results['metrics']['max_drawdown'] * 100:.2f}%")
    print(f"Volatility: {results['metrics']['volatility']:.2f}")

    return results

def main():
    df = pd.read_pickle("data/binance-BTCUSDT-1h.pkl")

    agent_configs = [
        {
            'agent_class': SARSAAgent,
            'agent_params': {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'min_epsilon': 0.01},
            'agent_name': 'SARSA',
            'use_mil_reward': True
        }
        # {
        #     'agent_class': DQNAgent,
        #     'agent_params': {'gamma': 0.99, 'lr': 1e-3, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'min_epsilon': 0.01, 'buffer_size': 10000, 'batch_size': 64},
        #     'agent_name': 'DQN',
        #     'use_mil_reward': True
        # }
        # {
        #     'agent_class': A2CAgent,
        #     'agent_params': {'gamma': 0.99, 'lr': 1e-3},
        #     'agent_name': 'A2C',
        #     'use_mil_reward': True
        # }
    ]

    for config in agent_configs:
        results = train_and_evaluate_agent(
            df=df,
            agent_class=config['agent_class'],
            agent_params=config['agent_params'],
            episodes=EPISODES,
            use_mil_reward=config['use_mil_reward'],
            mil_model_path="mil_reward_model/mil_model.pt",
            initial_balance=INITIAL_BALANCE,
            reward_window=REWARD_WINDOW,
            model_filename=f"{config['agent_name'].lower()}_model.pt",
            load_existing_model=False,
            agent_name=config['agent_name']
        )

if __name__ == "__main__":
    main()