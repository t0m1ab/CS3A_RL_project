"""
Solving Sokoban with DQN
=========================================================================================
Inspired from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import gymnasium as gym

from cs3arl.deeprl.trainers import DQNTrainer
from cs3arl.sokoban.dataloaders import MySokobanLoader
from cs3arl.sokoban.sokoban_env import SokobanEnv


def main():
    """ 
    Main process to train an agent on the Sokoban environement. 
    """

    import gymnasium as gym

    namespace = "sokoban"
    env_id = "sokoban-v0"

    with gym.envs.registration.namespace(ns=namespace):
        gym.register(
            id=env_id,
            entry_point=SokobanEnv,
        )

    # collection of maps
    map_collection = MySokobanLoader(level="easy", file_id=0)

    env = gym.make(
        id=f"{namespace}/{env_id}",
        map_collection=map_collection,
        merge_move_push=True,
        reset_mode="next", # only one map in the collection, so the reset mode does not matter (for the moment)
        max_steps=100,
    )

    # DQN trainer
    trainer = DQNTrainer(
        batch_size = 64,
        gamma = 0.99,
        eps_max = 0.9, # initial maximum exploration rate
        eps_min = 0.05, # final minimum exploration rate
        eps_start_decay = 0.1, # percentage of the total number of episodes
        eps_end_decay = 0.8, # percentage of the total number of episodes
        tau = 5e-3,
        learning_rate = 1e-4,
        n_episodes = 1000,
        memory_capacity = 1000,
        device = None,
        save_dir = None,
        save_results = True,
        n_checkpoints = 10,
    )

    # name of the experiment (used to create a subfolder to save the plots related to this agent/training)
    # if EXPERIMENT_NAME is set to None, a default name will be used based on the name of the trainer (e.g. DQNTrainer => DQN)
    EXPERIMENT_NAME = "DQN-sokoban"

    agent = trainer.train(
        env=env,
        net_type="convolutional",
        experiment_name=EXPERIMENT_NAME,
    )

    trainer.plot_exploration_decay()


if __name__ == "__main__":
    main()