import os
from pathlib import Path
import gymnasium as gym

from cs3arl.sokoban.sokoban_env import SokobanEnv

if __name__ == "__main__":

    namespace = "sokoban"
    env_id = "sokoban-v0"

    with gym.envs.registration.namespace(ns=namespace):
        gym.register(
            id=env_id,
            entry_point=SokobanEnv,
        )

    env = gym.make(
        id=f"{namespace}/{env_id}",
        merge_move_push=False,
        reset_mode="random"
    )

    print("Sokoban environment is ready!")

    # gym.pprint_registry()