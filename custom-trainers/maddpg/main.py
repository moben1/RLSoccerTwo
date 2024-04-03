"""
Main file to prepare and run MADDPG training
"""
import os
from pathlib import Path
import logging
import torch

from utils.misc import get_curr_run, load_config
from train import train

CONFIG_PATH = "custom-trainers/config/maddpg/maddpg.yaml"
USE_CUDA = torch.cuda.is_available()


if __name__ == '__main__':
    config = load_config(CONFIG_PATH)

    # Setting up the run directory
    model_dir = Path('./models/maddpg') / config['Model']['model_name']
    curr_run = get_curr_run(model_dir)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    config['run_dir'] = run_dir
    config['log_dir'] = log_dir

    # Setting up logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler(log_dir / 'train.log'),
                            logging.StreamHandler()
                        ])

    # Checking if cuda is available
    if not USE_CUDA and config['Torch']['rollout_dev'] == 'cuda':
        logging.warning("CUDA not available. Switching to CPU for rollout device.")
        config['Torch']['rollout_dev'] = 'cpu'
    if not USE_CUDA and config['Torch']['train_dev'] == 'cuda':
        logging.warning("CUDA not available. Switching to CPU for training device.")
        config['Torch']['train_dev'] = 'cpu'

    # Training the model
    train(config, USE_CUDA)
