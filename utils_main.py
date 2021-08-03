import os
import json
import random
import string
import argparse
from shutil import copyfile
from datetime import datetime
from ppo import ProximalPolicyOptimization as PPO


def get_unique_save_path(len_rand_str: int = 10):
    # Returns a pretty unique name for some folder in which to save the outcome of training a PPO agent

    now = datetime.now()
    today_string = now.strftime("%Y_%m_%d__%H_%M_%S")
    random_string = ''.join(random.choice(string.ascii_letters) for i in range(len_rand_str))

    return today_string + '__' + random_string


def save(args: json, attribute: str, saver):
    # Generic function to make sure that a given save path exists and to save something calling a given 'saver' function

    # Check if saving-path is provided. Otherwise don't save
    if hasattr(args, attribute):
        # Check if provided path exists, if not create it
        save_path = '/'.join(getattr(args, attribute).split('/')[:-1])
        print("Save path: ", getattr(args, attribute))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # Save content, using saver-method, to given path specified in the arguments
        saver(getattr(args, attribute))


def save_ppo(ppo: PPO, args, save_dir: str, config: json):
    # Saves all components that may be saved after a training run as requested by user through command line arguments

    if args.policy_net_path != '-':
        save(args=args, attribute='policy_net_path', saver=ppo.save_policy_net)

    if args.value_net_path != '-':
        save(args=args, attribute='value_net_path', saver=ppo.save_value_net)

    if args.stats_path != '-':
        save(args=args, attribute='stats_path', saver=ppo.save_train_stats)


def copy_config_to_results_folder(src_path, save_path):
    # Check if provided path exists, if not create it
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Copy config file
    copyfile(src=src_path, dst=save_path+src_path.split('/')[-1])


def get_argparser(save_dir: str):
    # Creates and returns an argument parser
    parser = argparse.ArgumentParser(description='Train or Demo the performance of a PPO agent.')

    # Training
    parser.add_argument('-c', '--config_path', type=str, required=False,
                        help='Specify path from where to load non-default config file',
                        default='./default_config_files/config_mountain_car_continuous.py')
    parser.add_argument('-s', '--stats_path', type=str, required=False,
                        help='Specify path where to save training stats. "-" for False.',
                        default=save_dir + 'train_stats.json')
    parser.add_argument('-p', '--policy_net_path', type=str, required=False,
                        help='Specify path where to save policy net. "-" for False.',
                        default=save_dir + 'policy_model.pt')
    parser.add_argument('-v', '--value_net_path', type=str, required=False,
                        help='Specify path where to save value net. "-" for False.',
                        default=save_dir + 'val_net_model.pt')

    # Demo/Eval
    parser.add_argument('-d', '--demo_path', type=str, required=False,
                        help='Specify path from where to load trained policy model for demonstrating its learning outcome visually.')

    return parser
