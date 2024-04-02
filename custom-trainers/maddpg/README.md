# MADDPG Implementation for Unity

This is an implementation of the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm for the Unity ML-Agents SoccerTwo environment, the algorithm is described in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (Lowe et. al. 2017)](https://arxiv.org/abs/1706.02275).

This implementation was created by [Shariq Iqbal](https://github.com/shariqiqbal2810/maddpg-pytorch) on github. We addapted if to work with the Unity environment.
We use the Petting zoo wrapper provided by Unity to communicate with the environment.

# Environment

The environment used for this work is a custom SoccerTwo environment, we wanted to simplify it to perform faster training. In the future, we may test the implementation on the original SoccerTwos environment to be able to provide some comparative results with the other ml-agents algorithms (e.i mapoca).

## Observation Space

The observability is **total** (all agents have access all informations). As MADDPG is a good algorithm for environments with partial (and different) observations for each agent, it is definitely relevant to test it on the original SoccerTwos environment.

Here is the observation space for each agent:

1. His own velocity (x, y)
2. His mate position (x, y)
3. The opponents positions 2 \* (x, y)
4. The ball position (x, y)
5. The ball velocity (x, y)
6. His own goal position (x, y)
7. The opponent goal position (x, y)

This result in a 16d vector. All observations are relative to the agent's reference.

## Action Space

The action space is continuous and 3d : z direction, x direction and y rotation. A continuous action space is better for MADDPG but the implementation should work with discrete action spaces as well (using the Gumbel-Softmax trick).

# Usage

First, you need the `ml-agents` python package and `pytorch`. Instruction could be found [here](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Installation.md) but we could provide more detailed instructions later.

You can customize environment and training parameters in a yaml file. Default configuration is provided in `config/maddpg/config.yaml`.
If you use another file you should change the CONFIG_PATH variable in `main.py` to the path of your configuration file.

Implementation has not been tested yet on other environments, the executable for SoccerTwo is provided in `executables/MADDPG_Windows`. If needed, the code for our custom environment is provided in `SoccerEnv`.

To run the training, simply run `main.py`. The training will be saved in the `models/maddpg` folder.
