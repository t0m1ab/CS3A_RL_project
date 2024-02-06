# Reinforcement Learning for Computer Game playing

**Authors:** Tom LABIAUSSE & Nicolas FABRIGOULE

**Date:** 2023/2024

## 0 - Setup

* Clone the repository:
```bash
git clone git@github.com:t0m1ab/CS3A_RL_project.git
```

* Install `cs3arl` as a package in edit mode (see config in `pyproject.toml`): 
```bash
mv CS3A_RL_project/
pip install -e .
``` 

* Test the installation using the newly declared command `cs3arl`: 
```bash
cs3arl --test
``` 

## 1 - Learn to play Black Jack using classic RL methods

* The folder [blackjack](./blackjack/) contains all the code required to train an agent how to play Black Jack using different RL methods like MC Exploring Starts, Temporal Difference Learning, Q-learning...

## 2 - Sokoban environment and visualization tools

* The folder [sokoban](./sokoban/) contains all the code required to define a Sokoban environment respecting the Gym API and interacting with it through a Gradio UI. It also contains maps as for example a subset of [DeepMind boxoban levels](https://github.com/google-deepmind/boxoban-levels).

## 3 - Deep RL training code

* The folder [deeprl](./deeprl/) contains code to train Deep RL agents. Still in dev...