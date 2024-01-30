from tqdm import tqdm
import numpy as np
import gymnasium as gym

from cs3arl.blackjack.agents import BlackjackAgent, QlearningAgent, MCESAgent, SARSAAgent
from cs3arl.blackjack.visualization import create_training_plots, create_grids, create_value_policy_plots, create_policy_plots


class Trainer():

    def __init__(self) -> None:
        self.env = None
        self.agent = None
        self.experiment_name = None
    
    def has_training_curves(self) -> bool:
        if self.agent is not None:
            return len(self.agent.training_error) > 0
        return False
    
    def create_training_plots(self, show: bool = False, save: bool = False) -> None:

        if self.experiment_name is None:
            raise ValueError("Need to train an agent first")
    
        if len(self.agent.training_error) == 0:
            raise ValueError("Agent didn't record training error in its attribute 'training_error'")
        
        # state values & policy with usable ace (ace counts as 11)
        create_training_plots(self.env, self.agent, rolling_length=500, show=show, save=save, tag=self.experiment_name)
    
    def create_value_policy_plots(self, show: bool = False, save: bool = False) -> None:

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
        self.n_episodes = n_episodes
        self.discount_factor = discount_factor
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = start_epsilon / (n_episodes / 2) # reduce the exploration over time
    
    def train(self, env: gym.Env, experiment_name: str = None) -> BlackjackAgent:
        
        self.env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=self.n_episodes)

        self.agent = QlearningAgent(
            action_space_size=self.env.action_space.n,
            learning_rate=self.lr,
            start_epsilon=self.start_epsilon,
            final_epsilon=self.final_epsilon,
            epsilon_decay=self.epsilon_decay,
            discount_factor=self.discount_factor,
        )

        self.experiment_name = experiment_name if experiment_name is not None else "QLEARNING"

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


class MCESTrainer(Trainer):
    
    def __init__(self, n_episodes: int, discount_factor: float) -> None:
        super().__init__()
        self.n_episodes = n_episodes
        self.discount_factor = discount_factor
    
    def train(self, env: gym.Env, experiment_name: str = None) -> BlackjackAgent:
        
        self.env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=self.n_episodes)

        self.agent = MCESAgent(
            action_space_size=self.env.action_space.n,
            discount_factor=self.discount_factor,
        )

        self.experiment_name = experiment_name if experiment_name is not None else "MCES"

        for episode in tqdm(range(self.n_episodes)):

            obs, _ = self.env.reset()
            random_action = np.random.choice([0,1]) # random action to explore the environment
            next_obs, reward, terminated, _, _ = self.env.step(random_action)

            # define lists of S_t, A_t, R_t values [see S&B section 5.3]
            states = [obs, next_obs] # S list
            actions = [random_action] # A list
            rewards = [None, reward] # R list

            while not terminated:

                action = self.agent.get_action(states[-1])
                next_obs, reward, terminated, _, _ = self.env.step(action)

                states.append(next_obs)
                actions.append(action)
                rewards.append(reward)
            
            if terminated: # the last state is probably out of the state space (because the chain terminated) so we don't need it
                states.pop()

            # at this stage states and actions should have T elements and rewards T+1 elements (uncomment the following lines to check)
            # assert len(actions) == len(states)
            # assert len(rewards) == len(states) + 1

            # update the agent
            self.agent.update(states, actions, rewards)
                
        return self.agent


class SARSATrainer(Trainer):
    
    def __init__(self, learning_rate: float, n_episodes: int, start_epsilon: float, final_epsilon: float, discount_factor: float) -> None:
        super().__init__()
        self.lr = learning_rate
        self.n_episodes = n_episodes
        self.discount_factor = discount_factor
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = start_epsilon / (n_episodes / 2) # reduce the exploration over time
    
    def train(self, env: gym.Env, experiment_name: str = None) -> BlackjackAgent:
        
        self.env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=self.n_episodes)

        self.agent = SARSAAgent(
            action_space_size=self.env.action_space.n,
            learning_rate=self.lr,
            start_epsilon=self.start_epsilon,
            final_epsilon=self.final_epsilon,
            epsilon_decay=self.epsilon_decay,
            discount_factor=self.discount_factor,
        )

        self.experiment_name = experiment_name if experiment_name is not None else "SARSA"

        for episode in tqdm(range(self.n_episodes)):

            state, _ = self.env.reset() # S1
            action = self.agent.get_action(self.env, state) # A1
            done = False

            # play one episode
            while not done:

                next_state, reward, terminated, truncated, _ = self.env.step(action) # R1, S2
                next_action = self.agent.get_action(self.env, next_state) # A2

                self.agent.update(state, action, reward, next_state, next_action, terminated)

                state = next_state
                action = next_action
                done = terminated or truncated

            self.agent.decay_epsilon()
                
        return self.agent


def main():

    # Test the creation of a Trainer
    import gymnasium as gym
    env = gym.make("Blackjack-v1", sab=True) 

    # QlearningTrainer
    trainer = QlearningTrainer(
        learning_rate=0.01,
        n_episodes=100000,
        start_epsilon=1.0,
        final_epsilon=0.1,
        discount_factor=0.95,
    )

    print("Qlearning Trainer is ready!")

    # MCESTrainer
    trainer = MCESTrainer(
        n_episodes=100,
        discount_factor=1.0,
    )

    print("MCES Trainer is ready!")

    # SARSATrainer
    trainer = SARSATrainer(
        learning_rate=0.01,
        n_episodes=100000,
        start_epsilon=1.0,
        final_epsilon=0.1,
        discount_factor=0.95,
    )

    print("SARSA Trainer is ready!")


if __name__ == "__main__":
    main()