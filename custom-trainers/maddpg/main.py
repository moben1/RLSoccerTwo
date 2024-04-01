"""
Main script to train MADDPG agents on Unity environment
"""
import yaml
import torch
import logging

from train import train

CONFIG_PATH = "config\\maddpg\\maddpg.yaml"
USE_CUDA = torch.cuda.is_available()
LOGGING_LEVEL = logging.INFO


def load_config(config_file):
    """ Load configuration file from yaml
    """
    with open(config_file) as f:
        config_yml = yaml.safe_load(f)

    if not USE_CUDA and config_yml['Torch']['rollout_dev'] == 'gpu':
        logging.warning("CUDA not available. Switching to CPU for rollout device.")
        config_yml['Torch']['rollout_dev'] = 'cpu'
    if not USE_CUDA and config_yml['Torch']['train_dev'] == 'gpu':
        logging.warning("CUDA not available. Switching to CPU for training device.")
        config_yml['Torch']['train_dev'] = 'cpu'
    return config_yml


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S')
    config = load_config(CONFIG_PATH)
    train(config, USE_CUDA)
