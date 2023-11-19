from collections import defaultdict
import numpy as np


class BlackjackAgent:

    def __init__(
        self,
        action_space_size: int,
        learning_rate: float,
        start_epsilon: float,
        final_epsilon: float,
        epsilon_decay: float,
        discount_factor: float,
    ):
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


class QlearningAgent(BlackjackAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action(self, env, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon) otherwise a random action with probability epsilon to ensure exploration.
        """
        if np.random.random() < self.epsilon: # with probability epsilon return a random action to explore the environment
            return env.action_space.sample()
        else: # with probability (1 - epsilon) act greedily (exploit)
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """
        Updates the Q-value of an action following the Q-learning method.
        """
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


if __name__ == "__main__":

    # Test the creation of an agent following the rules from Sutton & Barto
    import gymnasium as gym
    env = gym.make("Blackjack-v1", sab=True) 
    agent = BlackjackAgent(
        action_space_size=env.action_space.n,
        learning_rate=0.01,
        start_epsilon=1.0,
        final_epsilon=0.1,
        epsilon_decay=1e-3,
        discount_factor=0.95,
    )

    print("Agent is ready!")

