"""
Solving Blackjack with Q-Learning
=========================================================================================
Inspired from: https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
"""

# import os
# from pathlib import Path
# from collections import defaultdict
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from matplotlib.patches import Patch
from tqdm import tqdm
import gymnasium as gym

from agents import BlackjackAgent
from visualization import create_training_plots, create_grids, create_plots


# name of the experiment (used for saving the plots)
EXPERIMENT_NAME = "test_experiment"

# blackjack environment from Sutton & Barto (sab)
env = gym.make("Blackjack-v1", sab=True)

# hyperparameters
learning_rate = 0.01
n_episodes = 100000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)


### Training the agent

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()


### Visualize the training

create_training_plots(env, agent, rolling_length=500, show=False, save=True, tag=EXPERIMENT_NAME)


### Visualize the policy

# state values & policy with usable ace (ace counts as 11)
value_grid, policy_grid = create_grids(agent, usable_ace=True)
create_plots(value_grid, policy_grid, title="With usable ace", show=False, save=True, tag=EXPERIMENT_NAME)

# state values & policy without usable ace (ace counts as 1)
value_grid, policy_grid = create_grids(agent, usable_ace=False)
create_plots(value_grid, policy_grid, title="Without usable ace", show=False, save=True, tag=EXPERIMENT_NAME)
