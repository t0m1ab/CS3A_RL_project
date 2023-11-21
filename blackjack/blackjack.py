"""
Solving Blackjack with Q-Learning
=========================================================================================
Inspired from: https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
"""

import gymnasium as gym

from trainers import QlearningTrainer, MCESTrainer, SARSATrainer


def main():
    """ 
    Main process to train an agent how to play blackjack. 
    Different methods such as MCES or Qlearning are available to train the agent.
    Choose the method by uncommenting the corresponding trainer lines.
    """

    # blackjack environment from Sutton & Barto (sab)
    env = gym.make("Blackjack-v1", sab=True)

    # QLEARNING
    trainer = QlearningTrainer(
        learning_rate=0.01,
        n_episodes=100000,
        start_epsilon=1.0,
        final_epsilon=0.1,
        discount_factor=0.95,
    )

    # MCES
    # trainer = MCESTrainer(
    #     n_episodes=100000,
    #     discount_factor=1.0,
    # )

    # SARSA
    # trainer = SARSATrainer(
    #     learning_rate=0.01,
    #     n_episodes=100000,
    #     start_epsilon=1.0,
    #     final_epsilon=0.1,
    #     discount_factor=0.95,
    # )

    # name of the experiment (used to create a subfolder to save the plots related to this agent/training)
    # if EXPERIMENT_NAME is set to None, a default name will be used based on the name of the traine (e.g. QlearningTrainer => QLEARNING)
    EXPERIMENT_NAME = None

    agent = trainer.train(
        env=env,
        experiment_name=EXPERIMENT_NAME,
    )

    # save plots
    if trainer.has_training_curves():
        trainer.create_training_plots(save=True)
    trainer.create_value_policy_plots(save=True)


if __name__ == "__main__":
    main()