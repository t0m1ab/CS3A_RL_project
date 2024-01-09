from collections import defaultdict
import numpy as np

def array_to_tuple(a):
    arr1, arr2, arr3, arr4 = a
    tuple1 = tuple(tuple(row) for row in arr1)
    tuple2 = tuple(tuple(row) for row in arr2)
    tuple3 = tuple(tuple(row) for row in arr3)
    tuple4 = tuple(tuple(row) for row in arr4)
    tuple_a = (tuple1, tuple2, tuple3, tuple4)
    return(tuple_a)

class SokobanAgent:

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
        self.reward = 0
        self.training_error = []


class QlearningAgent(SokobanAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action(self, env, obs: np.array) -> int:
        """
        Returns the best action with probability (1 - epsilon) otherwise a random action with probability epsilon to ensure exploration.
        """
        if np.random.random() < self.epsilon: # with probability epsilon return a random action to explore the environment
            return env.action_space.sample()
        else: # with probability (1 - epsilon) act greedily (exploit)
            return int(np.argmax(self.q_values[array_to_tuple(obs)]))

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
        #if isinstance(next_obs, np.ndarray):
        #    tuple_obs = tuple(next_obs.flatten())
        #elif isinstance(next_obs, tuple):
        #    tuple_obs = tuple(el.flatten() if isinstance(el, np.ndarray) else el for el in next_obs)
        #elif next_obs is None:
        #    tuple_obs = (0,)  # ou toute autre valeur par défaut
        #else:
        #    raise TypeError("Unsupported type for next_obs")
        #tuple_obs=(tuple(next_obs[0].flatten()),tuple(next_obs[1].flatten()),tuple(next_obs[2].flatten()),tuple(next_obs[3]).flatten())
        # Supposons que vos tableaux numpy sont nommés arr1, arr2, arr3, arr4
        tuple_next_obs=array_to_tuple(next_obs)
        future_q_value = (not terminated) * np.max(self.q_values[tuple_next_obs])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[tuple_next_obs][action]
        self.q_values[tuple_next_obs][action] = self.q_values[tuple_next_obs][action] + self.lr * temporal_difference
        self.training_error.append(temporal_difference)
        self.reward=reward + self.discount_factor*self.reward

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class MCESAgent(SokobanAgent):

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
        self.reward=G



class MC_soft_policyAgent(SokobanAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q_values = defaultdict(lambda: np.random.rand(kwargs["action_space_size"])) # default q_value are randomly initialized
        self.mean_return = defaultdict(lambda: (0,0)) # returns[x] is (0,0) by default for any x and represents (n, R_n) [see S&B section 2.4]

    def get_action(self, env, state: tuple[int, int, bool]) -> int:
        """
        Returns the best action following a greedy policy wrt the state-action function.
        """
        if np.random.random() < self.epsilon: # with probability epsilon return a random action to explore the environment
            return env.action_space.sample()
        else: # with probability (1 - epsilon) act greedily (exploit)
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
        Updates the Q-value of an action following the Monte-Carlo Soft Policy method [see S&B section 5.3].
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



if __name__ == "__main__":

    # Test the creation of an agent following the rules from Sutton & Barto
    import gymnasium as gym
    env = gym.make("Sokoban-v2", sab=True) 

    # QlearningAgent
    Qlearning_agent = QlearningAgent(
        action_space_size=env.action_space.n,
        learning_rate=0.01,
        start_epsilon=1.0,
        final_epsilon=0.1,
        epsilon_decay=1e-3,
        discount_factor=0.95,
    )

    print("Qlearning agent is ready!")

    # MCESAgent
    MCES_agent = MCESAgent(
        action_space_size=env.action_space.n,
        discount_factor=1.0,
    )

    print("MCES agent is ready!")


    # MC_soft_policyAgent
    MC_soft_policy_agent = MC_soft_policyAgent(
        action_space_size=env.action_space.n,
        discount_factor=1.0,
    )

    print("MC_soft_policy agent is ready!")