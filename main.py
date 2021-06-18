import os
import ast
import argparse
from utils import save, get_unique_save_path
from plot import plot_avg_trajectory_len
from ppo import ProximalPolicyOptimization as PPO

# Path to directory where to save all data to be saved after training
save_dir = './train_results/' + get_unique_save_path() + '/'

# Create parser
parser = argparse.ArgumentParser(description='Train or Demo the performance of a PPO agent.')

# Training
parser.add_argument('-c', '--config_path', type=str, required=False, help='Specify path from where to load non-default config file', default='./config.py')
parser.add_argument('-s', '--stats_path', type=str, required=False, help='Specify path where to save training stats. "-" for False.', default=save_dir+'train_stats.json')
parser.add_argument('-p', '--policy_net_path', type=str, required=False, help='Specify path where to save policy net. "-" for False.', default=save_dir+'policy_model.pt')
parser.add_argument('-v', '--value_net_path', type=str, required=False, help='Specify path where to save value net. "-" for False.', default=save_dir+'val_net_model.pt')
parser.add_argument('-g', '--graphic_path', type=str, required=False, help='Specify path where to save graphic/plot. "-" for False.', default=save_dir+'traj_len_fig.png')

# Demo/Eval
parser.add_argument('-d', '--demo_path', type=str, required=False, help='Specify path from where to load trained policy model for demonstrating its learning outcome visually.')

# Parse arguments
args = parser.parse_args()


def main(args):

    print('Args:\n', args)

    # Load configurations from file
    file = open(args.config_path, 'r').read()
    config = ast.literal_eval(file)

    # Print config for feedback purposes
    print('Config:\n', config)

    if args.demo_path:
        # Demo mode - Replay trained agent

        print('Demo mode. Model used for running a performance demonstration:', args.demo_path)

        ppo = PPO(env=config['env'])
        ppo.load(args.demo_path)
        ppo.eval(time_steps=5000, render=True)

    else:
        # Training mode

        # Set up PPO agent as specified in configurations file
        ppo = PPO(**config)

        # Train the PPO agent
        train_stats = ppo.learn()

        # Print the stats
        print(train_stats)

        # Save as requested
        #if 'policy_net_path' in args.keys():
        #    # Check if provided path exists, if not create it
        #    if not os.path.isdir(args.policy_net_path):
        #        os.makedirs(args.policy_net_path)
        #    # Save policy net
        #    ppo.save_policy_net(path_policy=args.policy_net_path)
        save(args=args, attribute='policy_net_path', saver=ppo.save_policy_net)

        if 'value_net_path' in args.keys():
            # Check if provided path exists, if not create it
            if not os.path.isdir(args.value_net_path):
                os.makedirs(args.value_net_path)
            # Save value net
            ppo.save_value_net(path_val_net=args.value_net_path)

        if 'stats_path' in args.keys():
            # Check if provided path exists, if not create it
            if not os.path.isdir(args.stats_path):
                os.makedirs(args.stats_path)
            # Save statistics
            ppo.save_train_stats(path=args.stats_path)

        if 'graphic_path' in args.keys():
            # Check if provided path exists, if not create it
            if not os.path.isdir(args.graphic_path):
                os.makedirs(args.graphic_path)
            # Save statistics
            plot_avg_trajectory_len(train_stats, save_path=args.graphic_path)

        print('Done.')


if __name__ == "__main__":
    main(args)
