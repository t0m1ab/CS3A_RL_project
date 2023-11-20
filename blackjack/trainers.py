import gymnasium as gym
from tqdm import tqdm

from agents import BlackjackAgent, QlearningAgent
from visualization import create_training_plots, create_grids, create_value_policy_plots, create_policy_plots


class Trainer():

    def __init__(self) -> None:
        self.env = None
        self.agent = None
        self.experiment_name = None
    
    def create_training_plots(self, rolling_length: int, show: bool = False, save: bool = False):

        if self.experiment_name is None:
            raise ValueError("Need to train an agent first")
        
        # state values & policy with usable ace (ace counts as 11)
        create_training_plots(self.env, self.agent, rolling_length=rolling_length, show=show, save=save, tag=self.experiment_name)
    
    def create_value_policy_plots(self, show: bool = False, save: bool = False):

        if self.experiment_name is None:
            raise ValueError("Need to train an agent first")
        
        # state values & policy with usable ace (ace counts as 11)
        usable_ace_value_grid, usable_ace_policy_grid = create_grids(self.agent, usable_ace=True)
        create_value_policy_plots(usable_ace_value_grid, usable_ace_policy_grid, title="Policy usable ace", show=show, save=save, tag=self.experiment_name)

        # state values & policy without usable ace (ace counts as 1)
        no_usable_ace_value_grid, no_usable_ace_policy_grid = create_grids(self.agent, usable_ace=False)
        create_value_policy_plots(no_usable_ace_value_grid, no_usable_ace_policy_grid, title="Policy no usable ace", show=show, save=save, tag=self.experiment_name)

        # total policy (usable & no usable ace)
        create_policy_plots(usable_ace_policy_grid, no_usable_ace_policy_grid, title="Policy", show=show, save=save, tag=self.experiment_name)


class QlearningTrainer(Trainer):
    
    def __init__(self, learning_rate: float, n_episodes: int, start_epsilon: float, final_epsilon: float, discount_factor: float) -> None:
        super().__init__()
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.n_episodes = n_episodes
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = start_epsilon / (n_episodes / 2) # reduce the exploration over time
    
    def train(self, env: gym.Env, experiment_name: str) -> BlackjackAgent:
        
        self.env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=self.n_episodes)

        self.agent = QlearningAgent(
            action_space_size=self.env.action_space.n,
            learning_rate=self.lr,
            start_epsilon=self.start_epsilon,
            final_epsilon=self.final_epsilon,
            epsilon_decay=self.epsilon_decay,
            discount_factor=self.discount_factor,
        )

        self.experiment_name = experiment_name

        for episode in tqdm(range(self.n_episodes)):
            obs, info = self.env.reset()
            done = False

            # play one episode
            while not done:
                action = self.agent.get_action(self.env, obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # update the agent
                self.agent.update(obs, action, reward, terminated, next_obs)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs

            self.agent.decay_epsilon()
                
        return self.agent


if __name__ == "__main__":

    # Test the creation of a QlearningTrainer
    import gymnasium as gym
    env = gym.make("Blackjack-v1", sab=True) 
    trainer = QlearningTrainer(
        learning_rate=0.01,
        n_episodes=100000,
        start_epsilon=1.0,
        final_epsilon=0.1,
        discount_factor=0.95,
    )

    print("Trainer is ready!")