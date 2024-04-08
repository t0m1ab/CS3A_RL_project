import os
from pathlib import Path
import json
import gymnasium as gym

from cs3arl.deeprl.trainers import DeepTrainer
from cs3arl.deeprl.agents import DQNAgentSokoban
from cs3arl.sokoban.dataloaders import MySokobanLoader
from cs3arl.sokoban.sokoban_env import SokobanEnv
from cs3arl.utils import save_gif


DEMO_PATH = "outputs/demo/"


def load_sokoban_DQN_agent(model_name: str, path: str=None):
    """ Load a pretrained DQN agent for the Sokoban environement. """

    path = path if path is not None else os.path.join(DeepTrainer.DEFAULT_PATH, "outputs/")

    model_dir = os.path.join(path, model_name)
    model_json_file = os.path.join(model_dir, f"{model_name}.json")
    model_pt_file = os.path.join(model_dir, f"{model_name}.pt")

    if not os.path.isfile(model_json_file):
        raise FileNotFoundError(f"Model loading error: {model_name}.json not found in {model_dir}")
    if not os.path.isfile(model_pt_file):
        raise FileNotFoundError(f"Model loading error: {model_name}.pt not found in {model_dir}")
    
    with open(model_json_file, "r") as f:
        model_infos = json.load(f)
    
    # build agent from model infos
    agent = DQNAgentSokoban(
        obs_space_size = model_infos["observation_space_size"],
        action_space_size = model_infos["action_space_size"],
        net_type = model_infos["net_type"],
        device = "cpu",
    )

    # load pretrained network in agent
    agent.from_pretrained(model_pt_file)

    return agent


def load_sokoban_env(
        map_collection: MySokobanLoader,
        merge_move_push: bool=True,
        reset_mode: str="fixed",
        max_steps: int=100,
    ):
    """ Load the Sokoban environement. """

    namespace = "sokoban"
    env_id = "sokoban-v0"

    with gym.envs.registration.namespace(ns=namespace):
        gym.register(
            id=env_id,
            entry_point=SokobanEnv,
        )

    env = gym.make(
        id=f"{namespace}/{env_id}",
        map_collection=map_collection,
        merge_move_push=merge_move_push,
        reset_mode=reset_mode,
        max_steps=max_steps,
    )

    return env


def get_episode_frames(
        env: gym.Env,
        agent: DQNAgentSokoban,
        n_episodes: int=1,
    ):
    """ 
    Run episodes with the given agent in the given environement and return the frames for each episode.
    """

    # set agent in evaluation mode
    agent.eval()

    # play an episode
    all_frames = [[] for _ in range(n_episodes)]

    for episode_idx in range(n_episodes):

        done = False
        state, _ = env.reset()
        all_frames[episode_idx].append(env.unwrapped.get_image())

        while not done:

            action = agent.get_action(env, state)
            state, _, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            all_frames[episode_idx].append(env.unwrapped.get_image())
        
        # print("Episode length:", len(frames))

    return all_frames


def demo():

    # load env
    env = load_sokoban_env(
        map_collection=MySokobanLoader(level="easy", file_id=0),
        merge_move_push=True,
        reset_mode="fixed",
        max_steps=100,
    )

    # load pretrained agent
    agent = load_sokoban_DQN_agent(model_name="DQN-sokoban")

    # create frames
    all_frames = get_episode_frames(env, agent, n_episodes=4)

    for episode_idx, episode_frames in enumerate(all_frames):
        save_gif(episode_frames, gif_name="sokoban_demo" if len(all_frames) == 1 else f"sokoban_demo_{episode_idx}")


def main():
    demo()


if __name__ == "__main__":
    main()