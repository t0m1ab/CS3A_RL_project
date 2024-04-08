import os
from pathlib import Path
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from cs3arl.sokoban.sokoban_env import SokobanEnv
from cs3arl.deeprl.buffers import Transition, ReplayMemory
from cs3arl.deeprl.networks import ConvDQNCartPole, ConvDQNSokoban, FCDQNSokoban, NetType


class DeepAgent():

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
            net_type: str = None,
            device: str = None,
        ) -> None:

        self.__name__ = "DeepAgent"
        self._mode = "eval"
        self.device = device

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.epsilon = eps_start

        self.n_observations = obs_space_size
        self.n_actions = action_space_size
        self.lr = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.bs = batch_size
        self.memory_capacity = memory_capacity
        self.memory = ReplayMemory(self.memory_capacity)

        self.net_type = NetType.get_type(net_type)
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.criterion = nn.SmoothL1Loss()

        self.steps_done = 0
    
    @property
    def mode(self) -> str:
        return self._mode
    
    @mode.setter
    def mode(self, mode: str) -> None:
        if mode not in ["train", "eval"]:
            raise ValueError(f"Mode {mode} not recognized. Choose between 'train' and 'eval'.")
        self._mode = mode

    def train(self) -> None:
        """ Set the agent in training mode. """
        self.mode = "train"
        self.policy_net.train()
        self.target_net.train()
    
    def eval(self) -> None:
        """ Set the agent in evaluation mode. """
        self.mode = "eval"
        self.policy_net.eval()
        self.target_net.eval()
    
    def decay_epsilon(self) -> None:
        eps_factor = np.exp(-1. * self.steps_done / self.eps_decay)
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * eps_factor
    
    def get_action_tensor(self, env: gym.Env, state: torch.Tensor) -> torch.Tensor:
        
        if self.mode == "train": # train mode
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
        else: # eval mode
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
    
    def push_to_memory(self, state, action, next_state, reward) -> None:
        """ Push a transition to the memory without processing. """
        self.memory.push(state, action, next_state, reward)
    
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
    
    def save_agent(self, save_path: str, experiment_name: str = None) -> None:
        """ Save the agent's policy network. """
        fname = experiment_name if experiment_name is not None else self.__name__
        Path(save_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(save_path, f"{fname}.pt"))
    
    def from_pretrained(self, path: str) -> None:
        """ Load a pretrained network as policy network. """
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())


class DQNAgentCartPole(DeepAgent):

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)
        self.__name__ = "DQNAgentCartpole"

        if self.net_type == NetType.CONV:
            self.policy_net = ConvDQNCartPole(self.n_observations, self.n_actions).to(self.device)
            self.target_net = ConvDQNCartPole(self.n_observations, self.n_actions).to(self.device)
        else:
            raise ValueError(f"Network type {self.net_type.value} not implemented for CartPole.")
        
        self.target_net.load_state_dict(self.policy_net.state_dict()) # copy policy into target
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
    
    def to_tensor_state(self, state) -> torch.Tensor:
        if state is None:
            return None
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def push_to_memory(self, state, action, next_state, reward) -> None:
        state_tensor = self.to_tensor_state(state)
        next_state_tensor = self.to_tensor_state(next_state)
        reward_tensor = torch.tensor([reward], device=self.device)
        self.memory.push(state_tensor, action, next_state_tensor, reward_tensor)
    
    def get_action(self, env: gym.Env, state) -> torch.Tensor:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.get_action_tensor(env, state_tensor)


class DQNAgentSokoban(DeepAgent):

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)
        self.__name__ = "DQNAgentSokoban"

        if self.net_type == NetType.CONV:
            self.policy_net = ConvDQNSokoban(self.n_observations, self.n_actions).to(self.device)
            self.target_net = ConvDQNSokoban(self.n_observations, self.n_actions).to(self.device)
        elif self.net_type == NetType.FC:
            self.policy_net = FCDQNSokoban(self.n_observations, self.n_actions).to(self.device)
            self.target_net = FCDQNSokoban(self.n_observations, self.n_actions).to(self.device)
        else:
            raise ValueError(f"Network type {self.net_type.value} not implemented for Sokoban.")

        self.target_net.load_state_dict(self.policy_net.state_dict()) # copy policy into target
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)

    def to_tensor_state(self, state) -> torch.Tensor:
        if state is None:
            return None
        bloc_state = SokobanEnv.to_bloc_state(map=state[0], player_position=state[1])
        state_tensor = torch.tensor(bloc_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return state_tensor
    
    def push_to_memory(self, state, action, next_state, reward) -> None:
        """ Push a transition to the memory with channels disentanglement preprocessing. """
        state_tensor = self.to_tensor_state(state)
        next_state_tensor = self.to_tensor_state(next_state)
        reward_tensor = torch.tensor([reward], device=self.device)
        self.memory.push(state_tensor, action, next_state_tensor, reward_tensor)
    
    def get_action(self, env: gym.Env, state) -> torch.Tensor:
        state_tensor = self.to_tensor_state(state)
        return self.get_action_tensor(env, state_tensor)


DQN_AGENTS = {
    "cartpole": DQNAgentCartPole,
    "sokoban": DQNAgentSokoban,
}


def main():

    dqn_agent = DeepAgent(
        obs_space_size=100,
        action_space_size=4,
    )
    print(f"{dqn_agent.__name__} is ready!")


if __name__ == "__main__":
    main()