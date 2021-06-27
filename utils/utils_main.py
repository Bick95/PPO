import os
import json
import random
import string
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
        # Save policy net
        saver(getattr(args, attribute))


def save_ppo(ppo: PPO, args, save_dir: str, train_stats: dict, config: json):
    # Saves all components that may be saved after a training run as requested by user through command line arguments

    if args.policy_net_path != '-':
        save(args=args, attribute='policy_net_path', saver=ppo.save_policy_net)

    if args.value_net_path != '-':
        save(args=args, attribute='value_net_path', saver=ppo.save_value_net)

    if args.stats_path != '-':
        save(args=args, attribute='stats_path', saver=ppo.save_train_stats)

    # Save config
    with open(save_dir + '/config.json', 'w') as f:
        json.dump(config, f)

