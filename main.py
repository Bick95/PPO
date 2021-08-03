import ast
import torch
from constants import FINAL
from ppo import ProximalPolicyOptimization as PPO
from utils_main import get_unique_save_path, save_ppo, get_argparser, copy_config_to_results_folder

# Path to directory where to save all data to be saved after training
save_dir = './train_results/' + get_unique_save_path() + '/'

# Create parser for parsing command line arguments entered by user
parser = get_argparser(save_dir=save_dir)

# Parse arguments provided via command line
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

        # Make a copy of the config file for documentation purposes
        copy_config_to_results_folder(src_path=args.config_path, save_path=save_dir)

        # Set up PPO agent as specified in configurations file
        ppo = PPO(**config)

        try:
            # Train the PPO agent
            ppo.learn()
        except KeyboardInterrupt:
            # In case of keyboard interrupt, don't discard full test run so far, but make final eval and save outcome so far
            ppo.eval_and_log(eval_type=FINAL)

        # Save as requested
        save_ppo(ppo=ppo, args=args, save_dir=save_dir, config=config)

    else:
        # Demo mode - Replay trained agent

        print('Demo mode. Model used for running a performance demonstration:', args.demo_path)

        # Set up PPO agent as specified in configurations file
        ppo = PPO(**config)
        # Load trained model
        ppo.load(args.demo_path)
        # Do the visual evaluation
        ppo.eval(time_steps=5000, render=True)

    print('Done.')

    # Clean up cuda session
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main(args)
