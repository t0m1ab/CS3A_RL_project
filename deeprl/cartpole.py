"""
Solving Cartpole with DQN
=========================================================================================
Inspired from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import gymnasium as gym

from cs3arl.deeprl.trainers import DQNTrainer


def main():
    """ 
    Main process to train an agent how to handle the CartPole environement. 
    """

    env = gym.make("CartPole-v1")

    # DQN trainer
    trainer = DQNTrainer(
        batch_size = 128,
        gamma = 0.99,
        eps_start = 0.9,
        eps_end = 0.05,
        eps_decay = 100,
        tau = 5e-3,
        learning_rate = 1e-4,
        n_episodes = 500,
        memory_capacity = 10000,
        device = None,
        save_dir = None,
        track_results = True,
    )

    # name of the experiment (used to create a subfolder to save the plots related to this agent/training)
    # if EXPERIMENT_NAME is set to None, a default name will be used based on the name of the trainer (e.g. DQNTrainer => DQN)
    EXPERIMENT_NAME = None

    agent = trainer.train(
        env=env,
        experiment_name=EXPERIMENT_NAME,
    )

    # save plots
    trainer.plot_durations(show_result=True)


if __name__ == "__main__":
    main()