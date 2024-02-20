import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import gymnasium as gym
import torch
from tqdm import tqdm
import json

import cs3arl
from cs3arl.deeprl.agents import DeepRLAgent, DQNAgent, DQNAgentCartPole, DQNAgentSokoban
from cs3arl.sokoban.sokoban_env import SokobanEnv


class DeepTrainer():
    """ Deep RL trainer """

    DEVICES = ["cpu", "cuda", "mps"]

    DEFAULT_PATH = cs3arl.deeprl.__path__[0]

    def __init__(self, device: str, save_dir: str, save_results: bool) -> None:
        
        self.__name__ = "DeepTrainer"
        self.set_device("cpu" if device is None else device)
        self.save_dir = os.path.join(DeepTrainer.DEFAULT_PATH, "outputs/") if save_dir is None else save_dir

        self.env = None
        self.env_name = None
        self.agent = None
        self.experiment_name = None
        self.save_results = save_results
    
    def set_device(self, device: str) -> None:

        self.device = device
        
        if self.device not in self.DEVICES:
            raise ValueError(f"Device {self.device} is not supported. Please choose one of {self.DEVICES}.")
        
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA is not available. Using CPU instead.")
            self.device = "cpu"
        
        if self.device == "mps" and not(torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            print("Warning: MPS is not available. Using CPU instead.")
            self.device = "cpu"
        
    def save_results_to_json(self, **kwargs) -> None:

        exp_name = kwargs["experiment_name"] if "experiment_name" in kwargs else self.experiment_name

        data_dict = {
            "env": self.env_name,
            "agent": self.agent.__name__,
            "experiment_name": exp_name,
        }

        for key, data in kwargs.items():
            data_dict[key] = data
        
        save_path = os.path.join(self.save_dir, self.experiment_name)
        filename = f"{self.experiment_name}.json"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(save_path, filename), "w") as f:
            f.write(json.dumps(data_dict, indent=4))


class DQNTrainer(DeepTrainer):

    KEYWORD_TO_AGENT = {
        "cartpole": DQNAgentCartPole,
        "sokoban": DQNAgentSokoban,
    }
    
    def __init__(
            self,
            batch_size: int = 128,
            gamma: float = 0.99,
            eps_start: float = 0.9,
            eps_end: float = 0.05,
            eps_decay: float = 100,
            tau: float = 5e-3,
            learning_rate: float = 1e-4,
            n_episodes: int = 50, # low by default because can be slow if CPU
            memory_capacity: int = 10000,
            device: str = None,
            save_dir: str = None,
            save_results: bool = True,
            n_checkpoints: int = None,
        ) -> None:

        super().__init__(device, save_dir, save_results)

        self.__name__ = "DQNTrainer"
        self.bs = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = learning_rate
        self.n_episodes = n_episodes
        self.memory_capacity = memory_capacity
        self.n_checkpoints = n_checkpoints
        self.episode_durations = []
    
    def is_checkpoint_episode(self, episode_idx: int) -> bool:
        """ Check if the current episode is a checkpoint episode which triggers save/plot methods. """
        idx = episode_idx + 1
        return idx % (self.n_episodes // self.n_checkpoints) == 0
    
    def __is_sokoban_env(self):
        if self.env_name is None:
            raise ValueError("No environment was set.")
        return "sokoban" in self.env_name.lower()
    
    def __get_agent_constructor(self, env_name: str = None):
        """ Get the constructor of the DQN agent based on the environment name. """
        env_name = env_name if env_name is not None else self.env_name
        for keyword, agent_constructor in self.KEYWORD_TO_AGENT.items():
            if keyword in env_name.lower():
                return agent_constructor
        raise ValueError(f"No agent found in {DQNTrainer.KEYWORD_TO_AGENT} for environment '{env_name}'.")
    
    def train(self, env: gym.Env, experiment_name: str = None) -> DeepRLAgent:
        
        self.env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=self.n_episodes)
        self.env_name = self.env.unwrapped.spec.id

        # get the size of the observation space
        if self.__is_sokoban_env():
            (map, _), _ = self.env.reset()
            if map.shape[0] != map.shape[1]:
                raise ValueError(f"Observation returned by the environnement is not a square map: shape={obs[0].shape}")
            obs_space_size = map.shape[0] * map.shape[1] # number of cells in the map
        else:
            obs_space_size = len(self.env.reset()[0])
        # get the size of the action space
        action_space_size = self.env.action_space.n

        self.agent = self.__get_agent_constructor()(
            obs_space_size = obs_space_size,
            action_space_size = action_space_size,
            eps_start = self.eps_start,
            eps_end = self.eps_end,
            eps_decay = self.eps_decay,
            learning_rate = self.lr,
            gamma = self.gamma,
            tau = self.tau,
            batch_size = self.bs,
            memory_capacity = self.memory_capacity,
            device = self.device,
        )

        self.experiment_name = experiment_name if experiment_name is not None else "DQN"

        loop_log = f"Training DQN on {self.env_name} for {self.n_episodes} episodes"
        for episode_idx in tqdm(range(self.n_episodes), desc=loop_log):

            # Initialize the environment and get its state
            state, _ = self.env.reset()

            if self.__is_sokoban_env(): # convert to bloc state to feed a DQN
                state = SokobanEnv.to_bloc_state(map=state[0], player_position=state[1])

            ### MYTODO: convert state to tensor representation (list of values)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # infinite loop with integer counter that breaks when the episode is done
            for t in count():

                action = self.agent.get_action(self.env, state_tensor)

                observation, reward, terminated, truncated, _ = self.env.step(action.item())

                if self.__is_sokoban_env(): # convert to bloc state to feed a DQN
                    observation = SokobanEnv.to_bloc_state(map=observation[0], player_position=observation[1])

                reward_tensor = torch.tensor([reward], device=self.device)

                done = terminated or truncated

                if terminated: # no next state if the episode is terminated
                    next_state_tensor = None
                else:
                    next_state_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.agent.memory.push(state_tensor, action, next_state_tensor, reward_tensor)

                # Move to the next state
                state_tensor = next_state_tensor

                # Perform one step of the optimization (on the policy network)
                self.agent.update_policy_net() # optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                self.agent.update_target_net()

                if done:
                    self.episode_durations.append(t + 1)
                    # self.plot_durations()
                    break
            
            # only save results at checkpoint if a number of checkpoints is specified
            if self.n_checkpoints is not None and self.is_checkpoint_episode(episode_idx):
                checkpoint_name = f"{self.experiment_name}_{episode_idx+1}-{self.n_episodes}"
                # save data to JSON
                self.save_results_to_json(
                    experiment_name = checkpoint_name,
                    n_episodes = episode_idx + 1,
                    episode_durations = self.episode_durations,
                )
                # create plots
                self.plot_durations(episode_idx = episode_idx + 1)
                # save agent
                self.agent.save_agent(
                    save_path = os.path.join(self.save_dir, self.experiment_name, "checkpoints/"),
                    experiment_name = checkpoint_name,
                )

        if self.save_results:
            # save data to JSON
            self.save_results_to_json(
                n_episodes = self.n_episodes,
                episode_durations = self.episode_durations,
            )
            # create plots
            self.plot_durations()
            # save agent
            self.agent.save_agent(os.path.join(self.save_dir, self.experiment_name))
        
        return self.agent
    
    def plot_durations(self, episode_idx: int = None):
        """
        Plot the duration of each episode along the training.
        """

        is_training = episode_idx is not None

        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if is_training:
            plt.clf()
            plt.title(f"Training [{episode_idx}/{self.n_episodes}]")
        else:
            plt.title(f"Result of {self.experiment_name} training")
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        
        save_path = os.path.join(self.save_dir, self.experiment_name)
        filename = f"{self.experiment_name}.png"
        if is_training:
            save_path = os.path.join(save_path, "checkpoints/")
            filename = f"{self.experiment_name}_{episode_idx}-{self.n_episodes}.png"

        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(save_path, filename))
        plt.close()


def main():

    import gymnasium as gym

    env = gym.make("CartPole-v1")

    trainer = DQNTrainer(
        batch_size=128,
        gamma=0.99,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=100,
        tau=5e-3,
        learning_rate=1e-4,
        n_episodes=50,
        memory_capacity=10000,
        device="cpu",
    )

    print(f"{trainer.__name__} is ready!")


if __name__ == "__main__":
    main()