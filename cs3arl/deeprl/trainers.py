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
from cs3arl.deeprl.agents import DeepAgent, DQN_AGENTS


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
        
    def save_infos_json(self, **kwargs) -> None:

        exp_name = kwargs["experiment_name"] if "experiment_name" in kwargs else self.experiment_name

        data_dict = {
            "env": self.env_name,
            "agent": self.agent.__name__,
            "experiment_name": exp_name,
        }

        for key, data in kwargs.items():
            data_dict[str(key)] = data
                
        save_path = os.path.join(self.save_dir, self.experiment_name)
        filename = f"{self.experiment_name}.json"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(save_path, filename), "w") as f:
            f.write(json.dumps(data_dict, indent=4))


class DQNTrainer(DeepTrainer):
    """ Deep QLearning trainer """
    
    def __init__(
            self,
            batch_size: int = 128,
            gamma: float = 0.99,
            eps_max: float = 0.9,
            eps_min: float = 0.05,
            eps_start_decay: float = 0.1,
            eps_end_decay: float = 0.9,
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
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_start_decay = eps_start_decay
        self.eps_end_decay = eps_end_decay
        self.tau = tau
        self.lr = learning_rate
        self.n_episodes = n_episodes
        self.memory_capacity = memory_capacity
        self.n_checkpoints = n_checkpoints
        self.episode_durations = []
        self.is_sokoban_env = False
    
    def is_checkpoint_episode(self, episode_idx: int) -> bool:
        """ Check if the current episode is a checkpoint episode which triggers save/plot methods. """
        idx = episode_idx + 1
        return idx % (self.n_episodes // self.n_checkpoints) == 0
    
    def __get_agent_constructor(self, env_name: str = None):
        """ Get the constructor of the DQN agent based on the environment name. """
        for keyword, agent_constructor in DQN_AGENTS.items():
            if keyword in env_name.lower():
                if "sokoban" in keyword:
                    self.is_sokoban_env = True
                return agent_constructor
        raise ValueError(f"No agent found in DQN_AGENTS for environment '{env_name}'.")

    def train(self, env: gym.Env, net_type: str = None, experiment_name: str = None) -> DeepAgent:
        
        self.env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=self.n_episodes)
        self.env_name = self.env.unwrapped.spec.id

        # get the constructor of the agent based on the environment name
        agent_constructor = self.__get_agent_constructor(self.env_name)

        # get the size of the observation space
        if self.is_sokoban_env:
            (map, _), _ = self.env.reset()
            if map.shape[0] != map.shape[1]:
                raise ValueError(f"Observation returned by the environnement is not a square map: shape={map.shape}")
            obs_space_size = int(map.shape[0] * map.shape[1]) # number of cells in the map
        else:
            obs_space_size = len(self.env.reset()[0])
        # get the size of the action space
        action_space_size = int(self.env.action_space.n)

        # init the agent
        self.agent = agent_constructor(
            obs_space_size = obs_space_size,
            action_space_size = action_space_size,
            eps_max = self.eps_max,
            eps_min = self.eps_min,
            eps_start_decay = self.eps_start_decay,
            eps_end_decay = self.eps_end_decay,
            learning_rate = self.lr,
            gamma = self.gamma,
            tau = self.tau,
            batch_size = self.bs,
            memory_capacity = self.memory_capacity,
            net_type = net_type,
            device = self.device,
        )

        self.agent.train() # set the agent in training mode

        self.experiment_name = experiment_name if experiment_name is not None else "DQN"

        pbar = tqdm(range(self.n_episodes), desc=f"Training DQN on {self.env_name} for {self.n_episodes} episodes")
        for episode_idx in pbar:

            # update agent internal state (exploration rate, etc.) 
            self.agent.set_episode(episode_idx, self.n_episodes)

            state, _ = self.env.reset() # init the environment and get its state
            cum_reward = 0
            
            for t in count(): # infinite loop with integer counter that breaks when the episode is done

                action = self.agent.get_action(self.env, state)

                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                cum_reward += reward
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = observation

                self.agent.push_to_memory(state, action, next_state, reward)

                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.agent.update_policy_net()

                # Soft update of the target network's weights using the following: θ′ ← τ θ + (1 −τ )θ′
                self.agent.update_target_net()

                if done:
                    self.episode_durations.append(t + 1)
                    break
            
            # update progress bar
            pbar.set_postfix({
                "last_ep": self.episode_durations[-1],
                "mean_ep": np.mean(self.episode_durations[-100:]),
                "cum_reward": f"{cum_reward:.1f}",
            })
            
            # only save results at checkpoint if a number of checkpoints is specified
            if self.n_checkpoints is not None and self.is_checkpoint_episode(episode_idx):
                checkpoint_name = f"{self.experiment_name}_{episode_idx+1}-{self.n_episodes}"
                # save data to JSON
                self.save_infos_json(
                    experiment_name = checkpoint_name,
                    observation_space_size = obs_space_size,
                    net_type = net_type,
                    action_space_size = action_space_size,
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
            self.save_infos_json(
                observation_space_size = obs_space_size,
                action_space_size = action_space_size,
                net_type = net_type,
                n_episodes = self.n_episodes,
                episode_durations = self.episode_durations,
            )
            # create plots
            self.plot_durations()
            # save agent
            self.agent.save_agent(
                save_path = os.path.join(self.save_dir, self.experiment_name),
                experiment_name = self.experiment_name,
            )
        
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
            plt.title(f"{self.experiment_name} training")
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
    
    def plot_exploration_decay(self):
        """
        Plot the exploration rate decay along the training.
        """
        if self.agent is None:
            raise ValueError("Agent is not initialized. Please train the agent first.")
        
        epsilons = []
        for episode_idx in range(self.n_episodes):
            self.agent.set_episode(episode_idx, self.n_episodes)
            epsilons.append(self.agent.epsilon)
        
        plt.plot(epsilons)
        plt.title("Exploration parameter decay")
        plt.xticks(np.arange(0, self.n_episodes, step=self.n_episodes//10))
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xlabel("episode")
        plt.ylabel("exploration parameter $\epsilon$")
        plt.savefig(os.path.join(self.save_dir, self.experiment_name, "exploration_decay.png"))


def main():

    import gymnasium as gym

    trainer = DQNTrainer(
        batch_size=128,
        gamma=0.99,
        eps_max=0.9,
        eps_min=0.05,
        eps_start_decay=0.1,
        eps_end_decay=0.9,
        tau=5e-3,
        learning_rate=1e-4,
        n_episodes=50,
        memory_capacity=1000,
        device="cpu",
    )

    print(f"{trainer.__name__} is ready!")


if __name__ == "__main__":
    main()