import numpy as np
from mlagents_envs.envs import SoccerTwos  # import unity environment
env = SoccerTwos.env()

num_cycles = 10000

env.reset()
for agent in env.agent_iter(env.num_agents * num_cycles):
    prev_observe, reward, done, info = env.last()
    if reward != 0:
        print("reward: ", reward)
    if isinstance(prev_observe, dict) and 'action_mask' in prev_observe:
        action_mask = prev_observe['action_mask']
    if done:
        action = None
    else:
        action = env.action_spaces[agent].sample()  # randomly choose an action for example
    env.step(action)

env.close()
