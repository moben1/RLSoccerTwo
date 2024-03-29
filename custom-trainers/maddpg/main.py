from typing import Optional
import numpy as np
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from mlagents_envs.environment import UnityEnvironment

import argparse
import os
from trainer.MADDPG import MADDPG

EXECUABLE_PATH = "../CustomSoccer"

EPISODE_NUM = 30000  # total episode num during training procedure
LEARN_INTERVAL = 100  # steps interval between learning time
RANDOM_STEPS = 5e3  # random steps before the agent start to learn
TAU = 1e-3  # soft update parameter
GAMMA = 0.95  # discount factor
BUFFER_CAPACITY = int(1e6)  # capacity of replay buffer
BATCH_SIZE = 2048  # batch-size of replay buffer
ACTOR_LR = 0.0003  # learning rate of actor
CRITIC_LR = 0.0003  # learning rate of critic


def get_env(executable: str, seed: Optional[int] = None):
    """ Get a UnityParallelEnv Wrapped env for the PettingZoo API Wrapper 
        and the dimension info of the environment.

    Args:
        ml_env (BaseEnv): The UnityEnvironment that is being wrapped.
        seed (Optional[int], optional): The seed for the action spaces of the agents.

    Returns:
        _type_: _description_
    """
    u_env = UnityEnvironment(file_name=executable, seed=seed)
    new_env = UnityParallelEnv(u_env)
    print("Agent names:", new_env.agents)
    print("Current agent:", new_env.agents[0])
    print("Observation space of first agent:", new_env.observation_spaces[new_env.agents[0]].shape)
    print("Action space of first agent:", new_env.action_spaces[new_env.agents[0]])
    new_env.reset()
    _dim_info = {}
    for new_agent_id in new_env.agents:
        _dim_info[new_agent_id] = []  # [obs_dim, act_dim]
        _dim_info[new_agent_id].append(new_env.observation_space(new_agent_id).shape[0])
        _dim_info[new_agent_id].append(new_env.action_space(new_agent_id).shape[0])

    return new_env, _dim_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, help='id to save the run', default='unnamed_test_run')
    parser.add_argument('--episode_num', type=int, default=EPISODE_NUM,
                        help='total episode num during training procedure')
    parser.add_argument('--learn_interval', type=int, default=LEARN_INTERVAL,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=RANDOM_STEPS,
                        help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=TAU, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=GAMMA, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=BUFFER_CAPACITY, help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=ACTOR_LR, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=CRITIC_LR, help='learning rate of critic')
    args = parser.parse_args()

    # create folder to save result
    env_dir = os.path.join('./results', args.run_id)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len(list(os.listdir(env_dir)))
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    env, dim_info = get_env(EXECUABLE_PATH)
    maddpg = MADDPG(dim_info,
                    args.buffer_capacity,
                    args.batch_size,
                    args.actor_lr,
                    args.critic_lr,
                    result_dir
                    )

    step = 0  # global step counter
    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}
    for episode in range(args.episode_num):
        obs = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        while env.agents:  # interact with the env for an episode
            step += 1
            if step < args.random_steps:
                action = {agent_id: np.clip(env.action_space(agent_id).sample(), -1.0, 1.0) for agent_id in env.agents}
            else:
                action = maddpg.select_action(obs, explore=True)

            next_obs, reward, done, info = env.step(action)
            maddpg.add(obs, action, reward, next_obs, done)
            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

            if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                maddpg.learn(args.batch_size, args.gamma)
                maddpg.update_target(args.tau)

            obs = next_obs
            if all(done.values()):
                print("FINISHING EPISONDE CAUSE IT4S DONE\n")
                break

        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        # if (episode + 1) % 100 == 0:  # print info every 100 episodes
        message = f'episode {episode + 1}, '
        sum_reward = 0
        for agent_id, r in agent_reward.items():  # record reward
            message += f'{agent_id}: {r:>4f}; '
            sum_reward += r
        message += f'sum reward: {sum_reward}'
        print(message)

    maddpg.save(episode_rewards)  # save model

    # def get_running_reward(arr: np.ndarray, window=100):
    #    """calculate the running reward, i.e. average of last `window` elements from rewards"""
    #    running_reward = np.zeros_like(arr)
    #    for i in range(window - 1):
    #        running_reward[i] = np.mean(arr[:i + 1])
    #    for i in range(window - 1, len(arr)):
    #        running_reward[i] = np.mean(arr[i - window + 1:i + 1])
    #    return running_reward

    # training finishes, plot reward
    # fig, ax = plt.subplots()
    # x = range(1, args.episode_num + 1)
    # for agent_id, reward in episode_rewards.items():
    #    ax.plot(x, reward, label=agent_id)
    #    ax.plot(x, get_running_reward(reward))
    # ax.legend()
    # ax.set_xlabel('episode')
    # ax.set_ylabel('reward')
    # title = f'training result of maddpg solve {args.env_name}'
    # ax.set_title(title)
    # plt.savefig(os.path.join(result_dir, title))

    env.close()
