import argparse
import torch
import os
import numpy as np
from gym.spaces import Box
from pathlib import Path
from torch.autograd import Variable
# from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import CommMemory, CommBatchEpisodeMemory
from algorithms.qmix import QMix
from utils.agents import QmixAgents

USE_CUDA = torch.cuda.is_available()

EXECUTABLE = None
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
MAX_CYCLES = 40

train_config = {
    "epochs": 100000,
    "evaluate_epoch" : 1,
    "show_evaluate_epoch" : 20,
    "memory_batch" : 32,
    "memory_size" : 1000,
    "run_episode_before_train" : 3,  # Run several episodes with the same strategy, used in on-policy algorithms
    "learn_num" : 2,
    "lr_actor" : 1e-4,
    "lr_critic" : 1e-3,
    "gamma" : 0.99,  # reward discount factor
    "epsilon" : 0.7,
    "grad_norm_clip" : 10,
    "target_update_cycle" : 100,
    "save_epoch" : 1000,
    "model_dir" : r"./models",
    "result_dir" : r"./results",
    "cuda" : True,
}

result_buffer = []

config = {
    "model_name": "qmix",
    "continue_training": False,
    "seed": 1,
    "n_training_threads": 6,
    "discrete_action": False,
    "executable": None,


}

def run(config):
    global result_buffer
    model_dir = Path('./models/qmix') / config.model_name

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

    qmix = QMix(env_info=env)
    agents = QmixAgents(env_info=env)

    replay_buffer = CommMemory()
    batch_episode_memory = CommBatchEpisodeMemory(continuous_actions=False, n_actions=env.action_spaces[env.agents[0]].n,
                                                  n_agents=len(env.agents))
    
    t = 0
    for epoch in range(config.n_episodes):
        print(f"New episode: {epoch + 1} of {config.n_episodes}")
        obs = env.reset()
        terminated = False

        obs = env.reset()[0]
        finish_game = False
        cycle = 0
        while not finish_game and cycle < MAX_CYCLES:
            state = env.state()
            actions_with_name, actions, log_probs = agents.choose_actions(obs)
            obs_next, rewards, finish_game, infos = env.step(actions_with_name)
            state_next = env.state()
            batch_episode_memory.store_one_episode(one_obs=obs, one_state=state, action=actions,
                                                            reward=rewards, one_obs_next=obs_next,
                                                            one_state_next=state_next)
            total_reward += rewards
            obs = obs_next
            cycle += 1
        batch_episode_memory.set_per_episode_len(cycle)

        replay_buffer.store_episode(batch_episode_memory)
        batch_episode_memory.clear_memories()
        if replay_buffer.get_memory_real_size() >= 10:
            for i in range(train_config.learn_num):
                batch = replay_buffer.sample(train_config.memory_batch)
                agents.learn(batch, epoch)
        one_result_buffer = [total_reward]
        result_buffer.append(one_result_buffer)
        # if epoch % train_config.save_epoch == 0 and epoch != 0:
        #     save_model_and_result(epoch)
        print("episode_{} over, total_reward {}".format(epoch, total_reward))
    env.close()


# def save_model_and_result(episode: int):
#     agents.save_model()
#     with open(result_path, 'a', newline='') as f:
#         f_csv = csv.writer(f)
#         f_csv.writerows(result_buffer)
#         result_buffer.clear()
#     if "ppo" not in env_config.learn_policy:
#         with open(memory_path, 'wb') as f:
#             memory.episode = episode
#             pickle.dump(memory, f)
    

if __name__ == "__main__":
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
