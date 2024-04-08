import os
from pathlib import Path
import json
import gymnasium as gym

from cs3arl.deeprl.trainers import DeepTrainer
from cs3arl.deeprl.agents import DQNAgentSokoban
from cs3arl.sokoban.dataloaders import MySokobanLoader
from cs3arl.sokoban.sokoban_env import SokobanEnv


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
    
    print(model_infos.keys())

    #TODO

    return


def demo(model_name: str, path: str=None):
    """ 
    Small demo of a trained DQN agent on the Sokoban environement. 
    """

    # load pretrained agent
    agent = load_sokoban_DQN_agent(model_name=model_name, path=path)

    # TODO: 


def main():

    # load pretrained agent
    load_sokoban_DQN_agent(model_name="DQN-sokoban")


if __name__ == "__main__":
    main()