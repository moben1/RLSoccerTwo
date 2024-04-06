"""
Main script to train MADDPG agents on Unity environment
"""
import os
from pathlib import Path
import logging
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from gym.spaces import Box

from utils.make_env import make_parallel_env
from utils.buffer import ReplayBuffer
from utils.misc import scale_env_properties
from algorithms.maddpg import MADDPG


def train(config: dict, use_cuda: bool, logger: SummaryWriter) -> None:
    """ Train MADDPG agents on Unity environment

    Args:
        config (Dict[str: Any]): configuration dict for environment and model
        use_cuda (bool): whether to use CUDA for training (if available)

    Raises:
        FileNotFoundError: raised if the model should be loaded from a non-existent directory
    """

    logging.info("Starting training in : %s", config['run_dir'])
    if config['Model']['load_from'] is not None:
        config['Model']['load_from'] = Path(config['Model']['load_from'])
        if not config['Model']['load_from'].exists():
            logging.error("Could not find model directory %s", config['Model']['load_from'])
            raise FileNotFoundError(f"Could not find model directory {config['Model']['load_from']}")
        logging.info("Continuing training from model : %s", config['Model']['load_from'])
        logging.warning("Model parameters could be different from the current configuration as it is loaded")

    logging.info("Environment Configuration : %s", config['Environment'])
    logging.info("Model Configuration : %s", config['Model'])
    logging.info("Torch Configuration : %s", config['Torch'])
    logging.info("Scaled properties : %s", config['ScaledProperties'])

    # Set random seeds
    torch.manual_seed(config['Environment']['seed'])
    np.random.seed(config['Environment']['seed'])

    if not use_cuda:
        logging.warning("CUDA not available.")
        torch.set_num_threads(config['Torch']['n_training_threads'])

    # Loading Unity environment and set time scale
    env = make_parallel_env(**config['Environment'])
    env.set_time_scale(config['Environment']['time_scale'])
    logging.info("Environment loaded")

    # Load or create model
    if config['Model']['load_from'] is not None:
        maddpg = MADDPG.init_from_save(config['Model']['load_from'], use_cuda=use_cuda)
    else:
        maddpg = MADDPG.init_from_env(env, **config['Model']['Hyperparameters'])

    # Create replay buffer
    replay_buffer = ReplayBuffer(config['Model']['Buffer']['buffer_length'],
                                 maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])

    logging.info("MADDPG and Replay Buffer initialized")

    # Load config parameters
    n_episodes = config['Model']['n_episodes']
    n_rollout_threads = config['Environment']['n_rollout_threads']
    explore = config['Model']['Exploration']
    steps_per_update = config['Model']['steps_per_update']
    save_interval = config['Model']['save_interval']
    batch_size = config['Model']['Buffer']['batch_size']
    rollout_dev = config['Torch']['rollout_dev']
    train_dev = config['Torch']['train_dev']
    max_steps = config['Model']['max_steps']

    # Start training
    t = 0
    for ep_i in range(0, n_episodes, n_rollout_threads):
        logging.debug("Starting episode %i to %i of %i episodes",
                      ep_i + 1, ep_i + n_rollout_threads, n_episodes)
        obs = env.reset()

        maddpg.prep_rollouts(device=rollout_dev)

        # Decay exploration noise and env properties
        scale_env_properties(env, config['ScaledProperties'], ep_i)

        explr_pct_remaining = max(0, explore['n_exploration_eps'] - ep_i) / explore['n_exploration_eps']
        scale = explore['final_noise_scale'] + (explore['init_noise_scale']
                                                - explore['final_noise_scale']) * explr_pct_remaining
        maddpg.scale_noise(scale)
        maddpg.reset_noise()
        logging.debug("Decaying noise scale : %.4f", scale)

        # interact with the env for an episode
        ep_len = 0
        envs_dones = [False for _ in range(n_rollout_threads)]
        while not all(envs_dones):
            ep_len += 1
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False).to(device=rollout_dev)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.cpu().data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(n_rollout_threads)]

            # step environment, store transition in replay buffer
            next_obs, rewards, dones, _ = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

            obs = next_obs
            t += n_rollout_threads

            # Update all agents each steps_per_update step
            if (len(replay_buffer) >= batch_size
                    and (t % steps_per_update) < n_rollout_threads):
                maddpg.prep_training(device=train_dev)
                for _ in range(n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(batch_size, to_gpu=use_cuda)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device=rollout_dev)

            # Checking for environment done
            for i, done in enumerate(dones):
                if all(done) and not envs_dones[i]:
                    envs_dones[i] = True
                    ep_log_id = ep_i + envs_dones.count(True)
                    logger.add_scalar('episode_length', ep_len, ep_log_id)
            if ep_len > max_steps:
                logging.warning("Episode %i reached max steps", ep_i)
                break

        # Compute mean episode rewards per agent
        ep_rews = replay_buffer.get_average_rewards(ep_len * n_rollout_threads)
        for a_i, a_ep_rews in zip(maddpg.log_ids, ep_rews):
            logger.add_scalar('%s/mean_episode_rewards' % a_i, a_ep_rews, ep_i)

        # Save model after every save_interval episodes
        if ep_i % save_interval < n_rollout_threads:
            logging.info("Saving model at %s", config['run_dir']
                         / 'incremental' / f'model_ep{ep_i + 1}.pt')
            os.makedirs(config['run_dir'] / 'incremental', exist_ok=True)
            maddpg.save(config['run_dir'] / 'incremental' / f'model_ep{ep_i + 1}.pt')
            maddpg.save(config['run_dir'] / 'model.pt')

    # Final save
    logging.info("Training complete. Saving final model.")
    logging.info("Saving model %s", config['run_dir'] / 'model_ep.pt')
    maddpg.save(config['run_dir'] / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(config['log_dir'] / 'summary.json'))
    logger.close()
