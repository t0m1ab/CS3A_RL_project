import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F

from cs3arl.deeprl.buffers import Transition, ReplayMemory
from cs3arl.deeprl.networks import DQN


class DeepRLAgent:

    def __init__(
        self,
        obs_space_size: int,
        action_space_size: int,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        learning_rate: float,
        device: str,
    ) -> None:
        """
        Initialize a Reinforcement Learning agent with an empty dictionary of state-action values (q_values), a learning rate and an epsilon.

        ARGUMENTS:
            - MYTODO...
        """
        
        self.__name__ = "DeepRLAgent"
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.epsilon = eps_start
        self.lr = learning_rate
        self.device = device
        self.n_observations = obs_space_size
        self.n_actions = action_space_size
        self.steps_done = 0
    
    def decay_epsilon(self) -> None:
        eps_factor = np.exp(-1. * self.steps_done / self.eps_decay)
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * eps_factor


class DQNAgent(DeepRLAgent):

    def __init__(
            self, 
            obs_space_size: int,
            action_space_size: int,
            eps_start: float = 0.9,
            eps_end: float = 0.05,
            eps_decay: float = 1000,
            learning_rate: float = 1e-4,
            gamma: float = 0.99,
            tau: float = 5e-3,
            batch_size: int = 128,
            memory_capacity: int = 10000,
            device: str = None,
        ) -> None:

        super().__init__(
            obs_space_size=obs_space_size,
            action_space_size=action_space_size,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
            learning_rate=learning_rate,
            device=device,
        )


        self.__name__ = "DQNAgent"
        self.gamma = gamma
        self.tau = tau
        self.bs = batch_size
        self.memory_capacity = memory_capacity
        self.memory = ReplayMemory(self.memory_capacity)

        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # copy policy into target
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()
    
    def get_action(self, env: gym.Env, state: torch.Tensor) -> torch.Tensor:
        
        self.decay_epsilon()

        self.steps_done += 1

        if random.random() > self.epsilon:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)
    
    def update_policy_net(self) -> None:

        if len(self.memory) < self.bs: # first fill the memory before starting to sample experience and train
            return
        
        transitions = self.memory.sample(self.bs)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.bs, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    
    def update_target_net(self) -> None:

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_dict = self.target_net.state_dict()
        policy_net_dict = self.policy_net.state_dict()

        for key in policy_net_dict:
            target_net_dict[key] = policy_net_dict[key]*self.tau + target_net_dict[key]*(1-self.tau)
        
        self.target_net.load_state_dict(target_net_dict)


def main():

    dqn_agent = DQNAgent(
        obs_space_size=100,
        action_space_size=4,
    )
    print(f"{dqn_agent.__name__} is ready!")


if __name__ == "__main__":
    main()