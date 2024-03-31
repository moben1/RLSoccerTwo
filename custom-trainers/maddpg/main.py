"""
Main script to train MADDPG agents on Unity environment
"""
import argparse
import os
from pathlib import Path
import yaml
import torch
import numpy as np

from gym.spaces import Box

from torch.autograd import Variable
# from tensorboardX import SummaryWriter
from utils.make_env import make_parallel_env
from utils.buffer import ReplayBuffer
from algorithms.maddpg import MADDPG

CONFIG_PATH = "config\\maddpg\\maddpg.yaml"
USE_CUDA = torch.cuda.is_available()


def load_config(config_file):
    """ Load configuration file from yaml
    """
    with open(config_file) as f:
        config_yml = yaml.safe_load(f)
    return config_yml


def run(config):
    model_dir = Path('./models/maddpg') / config['Model']['model_name']

    # If continue training, model_name is the path to the model.pt file
    if config['Model']['continue_training']:
        print(f'Continuing training from existing model at "{model_dir}"')
        if not model_dir.exists():
            raise FileNotFoundError('Could not find model directory')
    # Else build new model directory path
    elif not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = f'run{max(exst_run_nums) + 1}'

    # Create new run directory
    if not config['Model']['continue_training']:
        run_dir = model_dir / curr_run
        log_dir = run_dir / 'logs'
        os.makedirs(log_dir)

    logger = None  # SummaryWriter(str(log_dir))

    # Set random seeds
    torch.manual_seed(config['Environment']['seed'])
    np.random.seed(config['Environment']['seed'])

    if not USE_CUDA:
        torch.set_num_threads(config['Torch']['n_training_threads'])

    # Loading Unity environment
    env = make_parallel_env(**config['Environment'])

    # Load or create model
    if config['Model']['continue_training']:
        maddpg = MADDPG.init_from_save(model_dir)
    else:
        maddpg = MADDPG.init_from_env(env, **config['Model']['Hyperparameters'])

    # Create replay buffer
    replay_buffer = ReplayBuffer(config['Model']['Buffer']['buffer_length'],
                                 maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])

    # Load config parameters
    n_episodes = config['Model']['n_episodes']
    n_rollout_threads = config['Environment']['n_rollout_threads']
    explore = config['Model']['Exploration']
    steps_per_update = config['Model']['steps_per_update']
    save_interval = config['Model']['save_interval']
    batch_size = config['Model']['Buffer']['batch_size']
    rollout_dev = config['Torch']['rollout_dev']
    train_dev = config['Torch']['train_dev']

    # Start training
    t = 0
    for ep_i in range(0, n_episodes, n_rollout_threads):
        print(f"New Episode : {ep_i + 1}-{ep_i + 1 + n_rollout_threads} of {n_episodes}")
        obs = env.reset()

        maddpg.prep_rollouts(device='cpu')

        # Decay exploration noise
        explr_pct_remaining = max(0, explore['n_exploration_eps'] - ep_i) / explore['n_exploration_eps']
        maddpg.scale_noise(explore['final_noise_scale'] + (explore['init_noise_scale']
                           - explore['final_noise_scale']) * explr_pct_remaining)
        maddpg.reset_noise()

        ep_len = 0
        envs_dones = [False for _ in range(n_rollout_threads)]
        while not all(envs_dones):  # interact with the env for an episode
            ep_len += 1
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(n_rollout_threads)]

            # step environment, store transition in replay buffer
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

            obs = next_obs
            t += n_rollout_threads

            # Update all agents each steps_per_update step
            if (len(replay_buffer) >= batch_size
                    and (t % steps_per_update) < n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device=train_dev)
                for u_i in range(n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(batch_size, to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device=rollout_dev)

            for i, done in enumerate(dones):
                if all(done) :
                    print(f"Episode {ep_i} of environment {i} over in {ep_len} steps")
                    envs_dones[i] = True

        # Compute mean episode rewards per agent
        ep_rews = replay_buffer.get_average_rewards(ep_len * n_rollout_threads)
        print("Episode mean rewards: ")
        for a_i, a_ep_rew in enumerate(ep_rews):
            print(f"\t{maddpg.agents_id[a_i]} : {a_ep_rew}")
            # logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        # Save model after every save_interval episodes
        if ep_i % save_interval < n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / f'model_ep{ep_i + 1}.pt')
            maddpg.save(run_dir / 'model.pt')

    # Final save
    maddpg.save(run_dir / 'model.pt')
    env.close()
    # logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    # logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Yaml file with configurations for training',
                        type=str, default=CONFIG_PATH)
    parser.add_argument('--model_name',
                        help='Name of directory to store model/training contents',
                        type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model_name is not None:
        config['Model']['model_name'] = args.model_name

    run(config)
