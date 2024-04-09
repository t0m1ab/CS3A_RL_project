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
        eps_max = 0.9,
        eps_min = 0.05,  
        eps_start_decay = 0.1,
        eps_end_decay = 0.9,
        tau = 5e-3,
        learning_rate = 1e-4,
        n_episodes = 1000,
        memory_capacity = 10000,
        device = None,
        save_dir = None,
        save_results = True,
        n_checkpoints = 10,
    )

    # name of the experiment (used to create a subfolder to save the plots related to this agent/training)
    # if EXPERIMENT_NAME is set to None, a default name will be used based on the name of the trainer (e.g. DQNTrainer => DQN)
    EXPERIMENT_NAME = "DQN-cartpole"

    agent = trainer.train(
        env=env,
        net_type="fully_connected",
        experiment_name=EXPERIMENT_NAME,
    )

    # save plots
    trainer.plot_durations()


if __name__ == "__main__":
    main()