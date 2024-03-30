import argparse
import torch
import os
import numpy as np
from gym.spaces import Box
from pathlib import Path
from torch.autograd import Variable
# from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from algorithms.maddpg import MADDPG

USE_CUDA = torch.cuda.is_available()

EXECUTABLE = "..\CustomSoccer"
DEFAULT_DIRECTORY = "unnamed_model"
SEED = 1
N_TRAINING_THREADS = 6
BUFFER_LENGTH = int(1e6)
N_EPISODES = 25000
STEPS_PER_UPDATE = 100  # Update model every STEPS_PER_UPDATE steps
BATCH_SIZE = 1024
N_EXPLORATION_EPS = 25000
INIT_NOISE_SCALE = 0.5
FINAL_NOISE_SCALE = 0.0
SAVE_INTERVAL = 100  # Save model every SAVE_INTERVAL episodes
HIDDEN_DIM = 64
LR = 0.01
TAU = 0.01
GAMMA = 0.95


def run(config):
    model_dir = Path('./models/maddpg') / config.model_name

    # If continue training, model_name is the path to the model.pt file
    if config.continue_training:
        print("Continuing training from existing model at '", model_dir, "'")
        if not model_dir.exists():
            raise FileNotFoundError("Could not find model directory")
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
    if not config.continue_training:
        run_dir = model_dir / curr_run
        log_dir = run_dir / 'logs'
        os.makedirs(log_dir)

    logger = None  # SummaryWriter(str(log_dir))

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    # Loading Unity environment
    env = make_env(config.executable, config.seed, config.discrete_action)

    # Load or create model
    if config.continue_training:
        maddpg = MADDPG.init_from_save(model_dir)
    else:
        maddpg = MADDPG.init_from_env(env,
                                      tau=config.tau,
                                      lr=config.lr,
                                      gamma=config.gamma,
                                      hidden_dim=config.hidden_dim)

    # Create replay buffer
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_spaces.values()],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_spaces.values()])

    # Start training
    t = 0
    for ep_i in range(config.n_episodes):
        print(f"New Episode : {ep_i + 1} of {config.n_episodes}")
        obs = env.reset_env(maddpg.agents_id)

        maddpg.prep_rollouts(device='cpu')

        # Decay exploration noise
        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale
                           - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        ep_len = 0
        while env.agents:  # interact with the env for an episode
            t += 1
            ep_len += 1
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i].reshape(1, -1))),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            actions = [ac.data.numpy() for ac in torch_agent_actions]

            # step environment, store transition in replay buffer
            next_obs, rewards, dones, infos = env.step_env(actions, maddpg.agents_id)
            replay_buffer.push(obs, actions, rewards, next_obs, dones)
            obs = next_obs

            # Update all agents each steps_per_update step
            if len(replay_buffer) >= config.batch_size and t % config.steps_per_update == 0:
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for a_i in range(maddpg.nagents):
                    sample = replay_buffer.sample(config.batch_size, to_gpu=USE_CUDA)
                    maddpg.update(sample, a_i, logger=logger)
                maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')

            if all(all(done) for done in dones):
                print("all dones")
            elif any(any(done) for done in dones):
                print("one dones")
            elif len(env.agents) > 4:
                print(env.agents)

        # Compute mean episode rewards per agent
        ep_rews = replay_buffer.get_average_rewards(ep_len)
        print("Episode mean rewards: ")
        for a_i, a_ep_rew in enumerate(ep_rews):
            print(f"\t{maddpg.agents_id[a_i]} : {a_ep_rew}")
            # logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        # Save model after every save_interval episodes
        if ep_i % config.save_interval == 0:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    # Final save
    maddpg.save(run_dir / 'model.pt')
    env.close()
    # logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    # logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", help="Path to Unity environment executable",
                        type=str, default=EXECUTABLE)
    parser.add_argument("--model_name",
                        help="Name of directory to store model/training contents",
                        type=str, default=DEFAULT_DIRECTORY)
    parser.add_argument("--seed", default=SEED, type=int, help="Random seed")
    parser.add_argument("--n_training_threads", default=N_TRAINING_THREADS, type=int)
    parser.add_argument("--buffer_length", default=BUFFER_LENGTH, type=int)
    parser.add_argument("--n_episodes", default=N_EPISODES, type=int)
    parser.add_argument("--steps_per_update", default=STEPS_PER_UPDATE, type=int)
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=N_EXPLORATION_EPS, type=int)
    parser.add_argument("--init_noise_scale", default=INIT_NOISE_SCALE, type=float)
    parser.add_argument("--final_noise_scale", default=FINAL_NOISE_SCALE, type=float)
    parser.add_argument("--save_interval", default=SAVE_INTERVAL, type=int)
    parser.add_argument("--hidden_dim", default=HIDDEN_DIM, type=int)
    parser.add_argument("--lr", default=LR, type=float)
    parser.add_argument("--tau", default=TAU, type=float)
    parser.add_argument("--gamma", default=GAMMA, type=float)
    parser.add_argument("--continue_training", action='store_true')
    parser.add_argument("--discrete_action", action='store_true')

    config = parser.parse_args()

    run(config)
