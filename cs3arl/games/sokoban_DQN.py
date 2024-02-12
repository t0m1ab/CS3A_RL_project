"""
Solving Cartpole with DQN
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
        reset_mode="random", # only one map in the collection, so the reset mode does not matter (for the moment)
        max_steps=100,
    )

    # DQN trainer
    trainer = DQNTrainer(
        batch_size = 128,
        gamma = 0.99,
        eps_start = 0.5,
        eps_end = 0.05,
        eps_decay = 100,
        tau = 5e-3,
        learning_rate = 1e-2,
        n_episodes = 5000,
        memory_capacity = 100,
        device = None,
        save_dir = None,
        save_results = False,
        n_checkpoints = 10,
    )

    # name of the experiment (used to create a subfolder to save the plots related to this agent/training)
    # if EXPERIMENT_NAME is set to None, a default name will be used based on the name of the trainer (e.g. DQNTrainer => DQN)
    EXPERIMENT_NAME = "DQN-sokoban-test"

    agent = trainer.train(
        env=env,
        experiment_name=EXPERIMENT_NAME,
    )

    # save plots
    # trainer.plot_durations()


if __name__ == "__main__":
    main()