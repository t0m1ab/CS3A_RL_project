from collections import defaultdict
import numpy as np


class ClassicAgent:

    def __init__(
        self,
        action_space_size: int,
        discount_factor: float,
        learning_rate: float = None,
        start_epsilon: float = None,
        final_epsilon: float = None,
        epsilon_decay: float = None,
    ) -> None:
        """
        Initialize a Reinforcement Learning agent with an empty dictionary of state-action values (q_values), a learning rate and an epsilon.

        ARGUMENTS:
            - learning_rate: The learning rate
            - start_epsilon: The initial epsilon value
            - epsilon_decay: The decay for epsilon
            - final_epsilon: The final epsilon value
            - discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(action_space_size)) # default q_value for a given state is given by np.zeros(2) = [0, 0]

        self.lr = learning_rate # alpha
        self.discount_factor = discount_factor # gamma

        self.epsilon = start_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay # reduce the exploration over time

        self.training_error = []
        self.reward_list = []
        self.reward_per_episode_list = []
        self.num_steps_list = []
    
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class QlearningAgent(ClassicAgent):

    def __init__(self, serializer=None, **kwargs):
        super().__init__(**kwargs)
        self.serializer=serializer

    def get_action(self, env, obs) -> int:
        """
        Returns the best action with probability (1 - epsilon) otherwise a random action with probability epsilon to ensure exploration.
        """
        if np.random.random() < self.epsilon: # with probability epsilon return a random action to explore the environment
            return env.action_space.sample()
        else: # with probability (1 - epsilon) act greedily (exploit)
            if self.serializer is None:
                return int(np.argmax(self.q_values[obs]))
            else :
                return int(np.argmax(self.q_values[self.serializer(obs)]))

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ) -> None:
        """
        Updates the Q-value of an action following the Q-learning method [see S&B section 6.5].
        """
        if self.serializer is not None:
            obs_serializer = self.serializer(obs)
            next_obs_serializer = self.serializer(next_obs)
        else :
            obs_serializer = obs
            next_obs_serializer = next_obs       
        future_q_value = (not terminated) * np.max(self.q_values[next_obs_serializer])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs_serializer][action]
        self.q_values[obs_serializer][action] = self.q_values[obs_serializer][action] + self.lr * temporal_difference
        self.training_error.append(temporal_difference)
        self.reward_list.append(reward)


class MCESAgent(ClassicAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q_values = defaultdict(lambda: np.random.rand(kwargs["action_space_size"])) # default q_value are randomly initialized
        self.mean_return = defaultdict(lambda: (0,0)) # returns[x] is (0,0) by default for any x and represents (n, R_n) [see S&B section 2.4]

    def get_action(self, state: tuple[int, int, bool]) -> int:
        """
        Returns the best action following a greedy policy wrt the state-action function.
        """
        return int(np.argmax(self.q_values[state]))
    
    def update_mean_return(self, state: tuple[int, int, bool], action: int, return_value: float) -> None:
        """ 
        Update the mean return of the state-action pair (state, action) following the incremental mean formula [see S&B section 2.4].
        """
        (n, R_n) = self.mean_return[(state, action)]
        self.mean_return[(state, action)] = (n+1, (n * R_n + return_value) / (n+1))

    def update(
        self,
        states: list[tuple[int, int, bool]],
        actions: list[int],
        rewards: list[float],
    ) -> None:
        """
        Updates the Q-value of an action following the Monte-Carlo Exploring Starts method [see S&B section 5.3].
        """
        state_action_pairs = list(zip(states, actions)) # need to transform into a list because zip is an iterator and will be consumed by the first for loop
        T = len(states)
        G = 0

        for t in range(T-1,-1,-1): # loop over the state-action pairs in reverse order (from T-1 to 0)
            G = self.discount_factor * G + rewards[t+1]
            if not state_action_pairs[t] in state_action_pairs[:t]: # first visit of the (state, action) pair in the episode
                state_t, action_t = state_action_pairs[t]
                self.update_mean_return(state_t, action_t, return_value=G) # update the mean return for the (state_t, action_t) pair
                self.q_values[state_t][action_t] = self.mean_return[state_action_pairs[t]][1]


class SARSAAgent(ClassicAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action(self, env, state: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon) otherwise a random action with probability epsilon to ensure exploration.
        """
        if np.random.random() < self.epsilon: # with probability epsilon return a random action to explore the environment
            return env.action_space.sample()
        else: # with probability (1 - epsilon) act greedily (exploit)
            return int(np.argmax(self.q_values[state]))

    def update(
        self,
        state: tuple[int, int, bool],
        action: int,
        reward: float,
        next_state: tuple[int, int, bool],
        next_action: int,
        terminated: bool,
    ) -> None:
        """
        Updates the Q-value of an action following the SARSA method [see S&B section 6.4].
        """
        q_value = self.q_values[state][action]
        next_q_value = (not terminated) * self.q_values[next_state][next_action]
        td_error = reward + self.discount_factor * next_q_value - q_value
        self.q_values[state][action] = q_value + self.lr * td_error


def main():

    # Test the creation of an agent following the rules from Sutton & Barto
    import gymnasium as gym
    env = gym.make("Blackjack-v1", sab=True) 

    # QlearningAgent
    Qlearning_agent = QlearningAgent(
        action_space_size=env.action_space.n,
        learning_rate=0.01,
        start_epsilon=1.0,
        final_epsilon=0.1,
        epsilon_decay=1e-3,
        discount_factor=0.95,
    )

    print("QlearningAgent is ready!")

    # MCESAgent
    MCES_agent = MCESAgent(
        action_space_size=env.action_space.n,
        discount_factor=1.0,
    )

    print("MCESAgent is ready!")

    # SARSAAgent
    MCES_agent = SARSAAgent(
        action_space_size=env.action_space.n,
        learning_rate=0.01,
        start_epsilon=1.0,
        final_epsilon=0.1,
        epsilon_decay=1e-3,
        discount_factor=0.95,
    )

    print("SARSAAgent is ready!")


if __name__ == "__main__":
    main()