import os
import json
import random
import string
from datetime import datetime
from plot import plot_avg_trajectory_len
from ppo import ProximalPolicyOptimization as PPO


def get_unique_save_path(len_rand_str: int = 10):
    now = datetime.now()
    today_string = now.strftime("%Y_%m_%d__%H_%M_%S")
    random_string = ''.join(random.choice(string.ascii_letters) for i in range(len_rand_str))

    return today_string + '__' + random_string


def save(args: json, attribute: str, saver):
    # Check if saving-path is provided. Otherwise don't save
    if hasattr(args, attribute):
        # Check if provided path exists, if not create it
        save_path = '/'.join(getattr(args, attribute).split('/')[:-1])
        print("Save path: ", save_path)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # Save policy net
        saver(getattr(args, attribute))


def save_ppo(ppo: PPO, args, save_dir: str, train_stats):
    if args.policy_net_path != '-':
        save(args=args, attribute='policy_net_path', saver=ppo.save_policy_net)

    if args.value_net_path != '-':
        save(args=args, attribute='value_net_path', saver=ppo.save_value_net)

    if args.stats_path != '-':
        save(args=args, attribute='stats_path', saver=ppo.save_train_stats)

    if args.graphic_path != '-':
        # Check if provided path exists, if not create it
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        # Save statistics
        plot_avg_trajectory_len(train_stats, save_path=args.graphic_path)
