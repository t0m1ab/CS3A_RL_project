import gym
import gym_sokoban
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def tuple_to_array(t):
    arr1 = [list(row) for row in t[0]]
    arr2 = [list(row) for row in t[1]]
    arr3 = [list(row) for row in t[2]]
    arr4 = [list(row) for row in t[3]]
    array_a = [arr1, arr2, arr3, arr4]
    return array_a

env = gym.make('Sokoban-v2')

from trainers import QlearningTrainer

def main():
    """ 
    Main process to train an agent how to play blackjack. 
    Different methods such as MCES or Qlearning are available to train the agent.
    Choose the method by uncommenting the corresponding trainer lines.
    """

    env = gym.make("Sokoban-v2")

    # QLEARNING
    trainer = QlearningTrainer(
        learning_rate=0.01,
        n_episodes=10,
        start_epsilon=1.0,
        final_epsilon=0.1,
        discount_factor=0.95,
    )

    EXPERIMENT_NAME = None

    agent, q_values, rewards_curve, obs_final = trainer.train(
        env=env,
        experiment_name=EXPERIMENT_NAME,
    )

    plt.plot(rewards_curve)
    plt.xlabel('Épisode')
    plt.ylabel('Récompense')
    plt.title('Récompenses par épisode')
    plt.savefig('rewards_curve.png')  # Enregistre le graphique en tant qu'image PNG
    plt.show()
    image = Image.fromarray(obs_final)

    save_path = 'path_to_save_image.png'

    image.save(save_path)

if __name__ == "__main__":
    main()