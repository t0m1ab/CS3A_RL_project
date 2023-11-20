"""
Solving Blackjack with Q-Learning
=========================================================================================
Inspired from: https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
"""

import gymnasium as gym

from trainers import QlearningTrainer, MCESTrainer


def main():
    """ Main process to train an agent how to play blackjack. """

    # name of the experiment (used for saving the plots)
    EXPERIMENT_NAME = "test_QLEARNING"

    # blackjack environment from Sutton & Barto (sab)
    env = gym.make("Blackjack-v1", sab=True)

    # Qlearning
    trainer = QlearningTrainer(
        learning_rate=0.01,
        n_episodes=100000,
        start_epsilon=1.0,
        final_epsilon=0.1,
        discount_factor=0.95,
    )

    # Monte-Carlo Exploring Starts
    # trainer = MCESTrainer(
    #     n_episodes=1000000,
    #     start_epsilon=1.0,
    #     final_epsilon=0.1,
    #     discount_factor=0.7,
    # )

    agent = trainer.train(
        env=env,
        experiment_name=EXPERIMENT_NAME,
    )

    # save plots
    if trainer.has_training_curves():
        trainer.create_training_plots(rolling_length=500, save=True)
    trainer.create_value_policy_plots(save=True)


if __name__ == "__main__":
    main()