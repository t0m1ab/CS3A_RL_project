import gymnasium as gym

from cs3arl.classicrl.trainers import QlearningTrainer_Sokoban
from cs3arl.sokoban.sokoban_env import SokobanEnv
from cs3arl.sokoban.dataloaders import MySokobanLoader
import matplotlib.pyplot as plt


def main():
    """ 
    Main process to train an agent how to play blackjack. 
    Different methods such as MCES or Qlearning are available to train the agent.
    Choose the method by uncommenting the corresponding trainer lines.
    """

    map_collection = MySokobanLoader(level='easy', file_id=1)


    namespace = "sokoban"
    env_id = "sokoban-v0"
    with gym.envs.registration.namespace(ns=namespace):
        gym.register(id=env_id, entry_point=SokobanEnv, max_episode_steps=10000000)



    env = gym.make(id=f"{namespace}/{env_id}", map_collection=map_collection)

    # QLEARNING
    trainer = QlearningTrainer_Sokoban(
        learning_rate=0.01,
        n_episodes=5000,
        start_epsilon=1.0,
        final_epsilon=0.1,
        discount_factor=1,
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

    agent, q_values, rewards_curve = trainer.train(
        env=env,
        experiment_name=EXPERIMENT_NAME,
    )

    reward_average = [sum(rewards_curve[max(0, i-99):i+1]) / (i+1 if i < 100 else 100) for i in range(len(rewards_curve))]

    plt.plot(rewards_curve, label='Nombre de steps par épisode', color='blue')
    plt.plot(reward_average, label='Nombre de steps moyen', color='red')
    plt.xlabel('Épisode')
    plt.ylabel('Nombre de steps')
    plt.title('Récompenses par épisode')
    plt.savefig('rewards_curve.png')  # Enregistre le graphique en tant qu'image PNG
    plt.show()

    #image = Image.fromarray(obs_final)

    #save_path = 'path_to_save_image.png'

    #image.save(save_path)

if __name__ == "__main__":
    main()