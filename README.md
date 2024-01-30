# Reinforcement Learning for Computer Game playing

**Authors:** Tom LABIAUSSE & Nicolas FABRIGOULE

**Date:** 2023/2024

## 0 - Setup

* Clone the repository and rename it from `CS3A_RL_project/` to `cs3arl/` with the following commands:
```bash
git clone git@github.com:t0m1ab/CS3A_RL_project.git
mv CS3A_RL_project/ cs3arl/
```

* Create a `pyproject.toml` at the same location as `cs3arl/` and copy paste the following lines:
```toml
[project]
name = "cs3arl"
version = "0.1.0"
description = "CS3A Sokoban RL project"

[build-system]
build-backend = "flit_core.buildapi"
requires = ["flit_core >=3.2,<4"]
```

* Run the following command to declare all code in `cs3arl/` as a package in edit mode: 
```bash
pip install -e .
``` 

* Run the following command to install all dependencies: 
```bash
pip install -r cs3arl/requirements.txt
``` 

## 1 - Learn to play Black Jack using classic RL methods

* The folder [blackjack](./blackjack/) contains all the code required to train an agent how to play Black Jack using different RL methods like MC Exploring Starts, Temporal Difference Learning, Q-learning...

## 2 - Sokoban environment and visualization tools

* The folder [sokoban](./sokoban/) contains all the code required to define a Sokoban environment respecting the Gym API and interacting with it through a Gradio UI. It also contains maps as for example a subset of [DeepMind boxoban levels](https://github.com/google-deepmind/boxoban-levels).

## 3 - Deep RL training code

* The folder [deeprl](./deeprl/) contains code to train Deep RL agents. Still in dev...