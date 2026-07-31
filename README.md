# SOFA_ZOO
This repository is part of the project "LapGym - An Open Source Framework for Reinforcement Learning in Robot-Assisted Laparoscopic Surgery".
It provides the code for the reinforcement learning experiments as described in the [LapGym paper](https://www.jmlr.org/papers/v24/23-0207.html) for the environments of [sofa_env](https://github.com/ScheiklP/sofa_env).

## My Contributions

This fork extends the original `sofa_zoo` repository (which provides PPO-based training scripts for `sofa_env` environments) with two additions built for ongoing surgical robotics safety research at the JHU MIRACLE Lab.

### 1. SAC Training Support

The original repository only supported Proximal Policy Optimization (PPO). I adapted the training scripts to support **Soft Actor-Critic (SAC)**, an off-policy algorithm better suited for continuous control tasks, and configured it to train against a custom obstacle-avoidance reward (see `sofa_env` fork for environment changes).

### 2. Evaluation Pipeline

The original repository has no built-in way to evaluate a trained policy's safety behavior. I built a standalone evaluation pipeline that:
- Loads a trained SAC policy and its normalization statistics
- Runs the policy deterministically over N episodes in the environment
- Logs per-episode safety violations, obstacle collisions, task success, and reward
- Outputs aggregate statistics (average violations/collisions per episode, overall success rate) to a JSON summary

### Usage

```bash
# 1. Train a policy (generates a checkpoint + normalization stats in envs/reach/runs/)
python envs/reach/sac.py

# 2. Evaluate the trained policy's task success and safety violations
python envs/evaluation/evaluate_reach.py
```

### Technologies Used

Python, Stable-Baselines3, Gymnasium, SOFA / SofaPython3, NumPy

# About the Original Project

## Dependencies
* ffmpeg for recording sample videos (`sudo apt install ffmpeg`)

## Installation
- If not done already: clone the [`sofa_env`](https://github.com/ScheiklP/sofa_env) repository and follow the instructionsof that repository to setup a conda environment and compile SOFA with SofaPython3 support. Do not forget to install the package itself with `pip install -e .` afterwards. Make sure that you did all the steps inside the `sofa` conda environment. If you have already installed SOFA with SofaPython3 in a conda env, it should be enough to pip install the repository.

- Clone this repository ([sofa_zoo](https://github.com/ScheiklP/sofa_zoo)).

- Make sure that the `sofa` conda environment is active.

- Install this repository with `pip install -e .`.

## Citing the Original Work
If you use the project in your work, please consider citing it with:
```bibtex
@article{JMLR:v24:23-0207,
  author  = {Paul Maria Scheikl and Balázs Gyenes and Rayan Younis and Christoph Haas and Gerhard Neumann and Martin Wagner and Franziska Mathis-Ullrich},
  title   = {LapGym - An Open Source Framework for Reinforcement Learning in Robot-Assisted Laparoscopic Surgery},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
  volume  = {24},
  number  = {368},
  pages   = {1--42},
  url     = {http://jmlr.org/papers/v24/23-0207.html}
}
```

## Acknowledgements
This work is supported by the Helmholtz Association under the joint research school "HIDSS4Health – Helmholtz Information and Data Science School for Health".
