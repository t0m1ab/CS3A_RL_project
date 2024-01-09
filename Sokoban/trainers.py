from tqdm import tqdm
import numpy as np
import gymnasium as gym

from agents import SokobanAgent, QlearningAgent, MCESAgent, MC_soft_policyAgent


class Trainer():

    def __init__(self) -> None:
        self.env = None
        self.agent = None
        self.experiment_name = None
    
    

class QlearningTrainer(Trainer):
    
    def __init__(self, learning_rate: float, n_episodes: int, start_epsilon: float, final_epsilon: float, discount_factor: float) -> None:
        super().__init__()
        self.lr = learning_rate
        self.n_episodes = n_episodes
        self.discount_factor = discount_factor
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = start_epsilon / (n_episodes / 2) # reduce the exploration over time
    
    def train(self, env: gym.Env, experiment_name: str = None) -> SokobanAgent:
        
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
        reward_curve=[]
        obs, info = self.env.reset()
        for episode in tqdm(range(self.n_episodes)):
            obs, info = self.env.reset_episode()
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
                reward_curve.append(self.agent.reward)
            
            self.agent.decay_epsilon()
            
                
        return self.agent, self.agent.q_values, reward_curve, env.render(mode="rgb_array")


class MCESTrainer(Trainer):
    
    def __init__(self, n_episodes: int, discount_factor: float) -> None:
        super().__init__()
        self.n_episodes = n_episodes
        self.discount_factor = discount_factor
    
    def train(self, env: gym.Env, experiment_name: str = None) -> SokobanAgent:
        
        self.env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=self.n_episodes)

        self.agent = MCESAgent(
            action_space_size=self.env.action_space.n,
            discount_factor=self.discount_factor,
        )

        self.experiment_name = experiment_name if experiment_name is not None else "MCES"
        for episode in tqdm(range(self.n_episodes)):

            obs, _ = self.env.reset_episode()
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

class MC_soft_policyTrainer(Trainer):
    
    def __init__(self, n_episodes: int, start_epsilon: float, discount_factor: float) -> None:
        super().__init__()
        self.n_episodes = n_episodes
        self.epsilon = start_epsilon
        self.discount_factor = discount_factor
    
    def train(self, env: gym.Env, experiment_name: str = None) -> SokobanAgent:
        
        self.env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=self.n_episodes)

        self.agent = MC_soft_policyAgent(
            action_space_size=self.env.action_space.n,
            discount_factor=self.discount_factor,
            start_epsilon=0.1,
        )

        self.experiment_name = experiment_name if experiment_name is not None else "MC_soft_policy"

        for episode in tqdm(range(self.n_episodes)):

            obs, _ = self.env.reset()
            random_action = np.random.choice([0,1]) # random action to explore the environment
            next_obs, reward, terminated, _, _ = self.env.step(random_action)

            # define lists of S_t, A_t, R_t values [see S&B section 5.3]
            states = [obs, next_obs] # S list
            actions = [random_action] # A list
            rewards = [None, reward] # R list

            while not terminated:

                action = self.agent.get_action(env, states[-1])
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


if __name__ == "__main__":

    # Test the creation of a Trainer
    import gymnasium as gym
    env = gym.make('Sokoban-v2')

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

    # MC_soft_policyTrainer
    trainer = MC_soft_policyTrainer(
        n_episodes=100,
        discount_factor=1.0,
    )

    print("MCES Trainer is ready!")