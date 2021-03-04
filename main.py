import ast
import json
import argparse
from plot import plot_avg_trajectory_len
from ppo import ProximalPolicyOptimization as PPO


# Create parser
parser = argparse.ArgumentParser(description='Interact with an PPO agent.')

# Training # TODO for later: remove defaults when done with debugging
parser.add_argument('-c', '--config_path', type=str, required=False, help='Specify path from where to load non-default config file', default='./config.py')
parser.add_argument('-s', '--stats_path', type=str, required=False, help='Specify path where to save training stats.', default='./train_stats.json')
parser.add_argument('-p', '--policy_net_path', type=str, required=False, help='Specify path where to save policy net.', default='./policy_model.pt')
parser.add_argument('-v', '--value_net_path', type=str, required=False, help='Specify path where to save value net.', default='./val_net_model.pt')
parser.add_argument('-g', '--graphic_path', type=str, required=False, help='Specify path where to save graphic/plot.', default='./traj_len_fig.png')

# Demo/Eval
parser.add_argument('-d', '--demo_path', type=str, required=False, help='Specify path from where to load model for demonstrating its learning outcome visually.')

# Parse arguments
args = parser.parse_args()


def main(args):

    print('Args:\n', args)

    if args.demo_path:
        # Demo mode

        print('Demo mode. Model used for running demonstration:', args.demo_path)
        # TODO

    else:
        # Training mode

        # Load configurations from file
        file = open(args.config_path, 'r').read()
        config = ast.literal_eval(file)

        # Print config for feedback purposes
        print('Config:\n', config)

        # Set up PPO agent as specified in configurations
        ppo = PPO(**config)

        # Train the PPO agent
        train_stats = ppo.learn()

        # Print the stats
        print(train_stats)

        # Save as requested
        if args.policy_net_path or args.value_net_path:
            ppo.save(path_policy=args.policy_net_path, path_val_net=args.value_net_path)

        if args.stats_path:
            ppo.save_train_stats(path=args.stats_path)

        if args.graphic_path:
            plot_avg_trajectory_len(train_stats, save_path=args.graphic_path)

        print('Done.')


if __name__ == "__main__":
    main(args)
