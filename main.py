import ast
import torch
import argparse
from ppo import ProximalPolicyOptimization as PPO
from utils_main import get_unique_save_path, save_ppo

# Path to directory where to save all data to be saved after training
save_dir = './train_results/' + get_unique_save_path() + '/'

# Create parser
parser = argparse.ArgumentParser(description='Train or Demo the performance of a PPO agent.')

# Training
parser.add_argument('-c', '--config_path', type=str, required=False, help='Specify path from where to load non-default config file', default='./default_config_files/config_mountain_car_continuous.py')
parser.add_argument('-s', '--stats_path', type=str, required=False, help='Specify path where to save training stats. "-" for False.', default=save_dir+'train_stats.json')
parser.add_argument('-p', '--policy_net_path', type=str, required=False, help='Specify path where to save policy net. "-" for False.', default=save_dir+'policy_model.pt')
parser.add_argument('-v', '--value_net_path', type=str, required=False, help='Specify path where to save value net. "-" for False.', default=save_dir+'val_net_model.pt')

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

    if not args.demo_path:
        # Training mode

        # Set up PPO agent as specified in configurations file
        ppo = PPO(**config)

        try:
            # Train the PPO agent
            train_stats = ppo.learn()
        except KeyboardInterrupt:
            # In case of keyboard interrupt, don't discard full test run so far, but make final eval and save outcome so far
            from constants import FINAL
            ppo.eval_and_log(eval_type=FINAL)

        # Save as requested
        save_ppo(ppo=ppo, args=args, save_dir=save_dir, train_stats=train_stats, config=config)

        print('Done.')

    else:
        # Demo mode - Replay trained agent

        print('Demo mode. Model used for running a performance demonstration:', args.demo_path)

        ppo = PPO(env=config['env'])
        ppo.load(args.demo_path)
        ppo.eval(time_steps=5000, render=True)



    # Clean up cuda session
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main(args)
