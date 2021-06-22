import gym
import json
import time
import random
import torch.optim
import numpy as np
from PIL import Image
from policy import Policy
from value_net import ValueNet
import torch.nn.functional as F
from torchvision.transforms import Resize
from torchvision.transforms import Grayscale
from constants import DISCRETE, CONTINUOUS
from ppo_utils import add_batch_dimension, simulation_is_stuck, visualize_markov_state


class ProximalPolicyOptimization:

    def __init__(self,
                 env: gym.Env or str,
                 epochs: int = 10,
                 total_num_state_transitions: int = 5000000,
                 time_steps_extensive_eval: int = 10000,
                 parallel_agents: int = 10,
                 param_sharing: bool = True,
                 learning_rate_pol: float = 0.0001,
                 learning_rate_val: float = 0.0001,
                 trajectory_length: int = 1000,
                 discount_factor: float = 0.99,
                 batch_size: int = 32,
                 clipping_parameter: float or dict = 0.2,
                 entropy_contrib_factor: float = 0.15,
                 vf_contrib_factor: float = .9,
                 input_net_type: str = 'MLP',
                 show_final_demo: bool = False,
                 intermediate_eval_steps: int = 200,
                 standard_dev=torch.ones,
                 nonlinearity: torch.nn.functional = F.relu,
                 markov_length: int = 1,  # How many environmental state get concatenated to one state representation
                 grayscale_transform: bool = False,  # Whether to transform RGB inputs to grayscale or not (if applicable)
                 network_structure: list = None,  # Replacement for hidden parameter,
                 deterministic_eval: bool = False,  # Whether to compute actions stochastically or deterministically throughout evaluation
                 resize_visual_inputs: tuple = None,
                 ):

        # Decide on which device to run model: CUDA/GPU vs CPU
        try:
            self.device = torch.device('cuda')
        except Exception:
            self.device = torch.device('cpu')

        # Save variables
        self.epochs = epochs
        self.iterations = total_num_state_transitions // (trajectory_length * parallel_agents)
        self.parallel_agents = parallel_agents  # Parameter N in paper
        self.T = trajectory_length              # Parameter T in paper
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.h = torch.tensor(entropy_contrib_factor, device=self.device, requires_grad=False)   # Parameter h in paper
        self.v = torch.tensor(vf_contrib_factor, device=self.device, requires_grad=False)        # Parameter v in paper
        self.input_net_type = input_net_type
        self.show_final_demo = show_final_demo  # Do render final demo visually or not
        self.intermediate_eval_steps = intermediate_eval_steps
        self.markov_length = markov_length
        self.time_steps_extensive_eval = time_steps_extensive_eval
        self.deterministic_eval = deterministic_eval

        # Determine how to handle clipping constant - keep it constant or anneal from some max value to some min value
        if isinstance(clipping_parameter, float):
            # Keep clipping parameter epsilon constant
            self._epsilon = torch.tensor(clipping_parameter, device=self.device, requires_grad=False)
            self.epsilon = lambda _: self._epsilon

        elif isinstance(clipping_parameter, dict):
            # Anneal clipping parameter between some values
            self._max_clipping_constant = clipping_parameter['max'] if 'max' in clipping_parameter.keys() else 1.
            self._min_clipping_constant = clipping_parameter['min'] if 'min' in clipping_parameter.keys() else 0.

            if clipping_parameter['decay_type'].lower() == 'linear':
                # Clipping parameter epsilon gets linearly annealed from max to min throughout training
                self.epsilon = lambda iteration: torch.tensor(
                    max(self._min_clipping_constant,
                        self._max_clipping_constant * ((self.iterations - iteration) / self.iterations)
                    ), device=self.device, requires_grad=False)

            elif clipping_parameter['decay_type'].lower() == 'exponential':
                # Clipping parameter epsilon gets exponentially annealed from max to min throughout training
                raise NotImplementedError("Exponential decay not implemented yet...")

            else:
                raise NotImplementedError("Decay can only be linear or exponential.")

        else:
            raise NotImplementedError("Clipping constant must be of type float or dict.")

        self.grayscale_transform = (input_net_type.lower() == "cnn" or input_net_type.lower() == "visual") and grayscale_transform
        self.grayscale_layer = Grayscale(num_output_channels=1) if self.grayscale_transform else None

        self.resize_transform = True if resize_visual_inputs else False
        self.resize_layer = Resize(size=resize_visual_inputs) if resize_visual_inputs else None

        # Set up documentation of training stats
        self.training_stats = {
            # How loss accumulated over one epoch develops over time
            'devel_epoch_loss': [],
            # How average accumulated over all epochs develops per iteration
            'devel_itera_loss': [],

            'init_avg_traj_len': [],
            'init_acc_reward': [],

            'train_avg_traj_len': [],
            'train_acc_reward': [],

            'final_avg_traj_len': [],
            'final_acc_reward': [],
        }

        # Create Gym env if not provided as such
        if isinstance(env, str):
            env = gym.make(env)

        self.env_name = env.unwrapped.spec.id
        self.action_space = env.action_space
        self.dist_type = DISCRETE if isinstance(self.action_space, gym.spaces.Discrete) else CONTINUOUS

        # Obtain information about the size of the portion of the Markov state to be dropped at the end when updating Markov state
        self.depth_processed_env_state = self.preprocess_env_state(add_batch_dimension(gym.make(self.env_name).reset())).shape[-1]

        observation_sample = self.sample_observation_space()

        # Create policy net
        self.policy = Policy(action_space=self.action_space,
                             observation_sample=observation_sample,
                             input_net_type=self.input_net_type,
                             nonlinearity=nonlinearity,
                             standard_dev=standard_dev,
                             dist_type=self.dist_type,
                             network_structure=network_structure
                             ).to(device=self.device)

        # Create value net (either sharing parameters with policy net or not)
        if param_sharing:
            self.val_net = ValueNet(shared_layers=self.policy.get_non_output_layers()).to(device=self.device)
        else:
            self.val_net = ValueNet(observation_sample=observation_sample,
                                    input_net_type=self.input_net_type,
                                    nonlinearity=nonlinearity,
                                    network_structure=network_structure,
                                    ).to(device=self.device)


        # Create optimizers (for policy network and state value network respectively) + potentially respective learning
        # rate schedulers:

        if isinstance(learning_rate_pol, float):
            # Simple optimizer with constant learning rate for policy net
            self.optimizer_p = torch.optim.Adam(params=self.policy.parameters(), lr=learning_rate_pol)
            self.lr_scheduler_pol = None

        elif isinstance(learning_rate_pol, dict):
            # Create optimizer plus a learning rate scheduler associated with optimizer
            self.optimizer_p = torch.optim.Adam(params=self.policy.parameters(), lr=learning_rate_pol['max'])

            if learning_rate_pol['decay_type'].lower() == 'linear':
                # TODO: define lambda
                self.lr_scheduler_pol = None  # TODO: implement scheduler
            else:
                raise NotImplementedError("Currently, only linear learning rate decay is supported.")

        else:
            raise NotImplementedError("learning_rate_pol must be (constant) float or dict.")

        if isinstance(learning_rate_val, float):
            # Simple optimizer with constant learning rate for value net
            self.optimizer_v = torch.optim.Adam(params=self.val_net.parameters(), lr=learning_rate_val)
            self.lr_scheduler_val = None

        elif isinstance(learning_rate_val, dict):
            # Create optimizer plus a learning rate scheduler associated with optimizer
            self.optimizer_v = torch.optim.Adam(params=self.val_net.parameters(), lr=learning_rate_val['max'])

            if learning_rate_val['decay_type'].lower() == 'linear':
                # TODO: define lambda
                self.lr_scheduler_val = None  # TODO: implement scheduler
            else:
                raise NotImplementedError("Currently, only linear learning rate decay is supported.")


        else:
            raise NotImplementedError("learning_rate_val must be (constant) float or dict.")

        # Vectorize env for each parallel agent to get its own env instance
        self.env = gym.vector.make(id=self.env_name, num_envs=self.parallel_agents, asynchronous=False)

        self.print_network_summary()


    def print_network_summary(self):
        print('Networks successfully created:')
        print('Policy network:\n', self.policy)
        print('Value net:\n', self.val_net)


    def sample_observation_space(self):
        # Returns a sample of observation/state space for a single minibatch example
        return self.init_markov_state(add_batch_dimension(gym.make(self.env_name).reset()))[0]


    def random_env_start(self, env):
        # Runs a given environment for a given number of time steps and returns both the env and the resulting observation
        # Actions are sampled at random, since some gym environments require sampling special actions to really kick-off the
        # simulation. Those actions are usually not the actions incidentally sampled by deterministic policies (during evaluation)

        # Reset environment and init Markov state
        last_env_state = env.reset()
        state = self.init_markov_state(add_batch_dimension(last_env_state))

        total_reward = 0
        identical_states = True

        while identical_states:
            # Randomly select action and perform it in environment
            action = env.action_space.sample()
            next_state, reward, terminal_state, _ = env.step(action)

            if terminal_state:
                # We experienced a terminal state during initialization phase of env.
                # Recursive call to make sure to always return non-terminated envs.
                return self.random_env_start(env)

            # Update Markov state
            state = self.env2markov(old_markov_state=state, new_env_state=add_batch_dimension(next_state))

            # Check whether simulation still hasn't started producing different states
            comparison = last_env_state == next_state
            identical_states = comparison.all()

            # Book-keeping
            total_reward += reward
            last_env_state = next_state

        return env, state, total_reward


    def grayscale(self, image_batch):
        return self.grayscale_layer(image_batch)


    def resize(self, image_batch):
        return self.resize_layer(image_batch)


    def preprocess_env_state(self, state_batch: np.ndarray):

        state_batch = torch.tensor(state_batch, dtype=torch.float)

        if self.resize_transform or self.grayscale_transform:

            # Required channel ordering for resizing and grayscaling: (Batch, Color, Height, Width)
            state_batch = state_batch.permute(0, 3, 1, 2)

            if self.resize_transform:
                state_batch = self.resize(state_batch)

            if self.grayscale_transform:
                state_batch = self.grayscale(state_batch)            # Convert to grayscale

            # Bring dimensions back into right order: (Batch, Height, Width, Color=1)
            # Color-dimension (=1) is used here as that dimension along which different env states get stacked to form a Markov state
            state_batch = state_batch.permute(0, 2, 3, 1)

        return state_batch


    def init_markov_state(self, initial_env_state: np.ndarray):
        # Takes a batch of initial states as returned by (vectorized) gym env, and returns a batch tensor with each
        # state being repeated a few times (as many times as a Markov state consists of given self.markov_length param)

        initial_env_state = self.preprocess_env_state(initial_env_state)

        # Repeat initial state a few times to form initial markov state
        init_markov_state = torch.cat(self.markov_length * [initial_env_state], dim=-1)

        return init_markov_state


    def env2markov(self, old_markov_state: torch.tensor, new_env_state: np.ndarray):
        # Function to update Markov states by dropping the oldest environmental state in Markov state and instead adding
        # the latest environmental state observation to the Markov state representation

        # Preprocessing of new env state
        new_env_state = self.preprocess_env_state(new_env_state)

        # Obtain new Markov state by dropping oldest state, shifting all other states to the back, and adding latest
        # env state observation to the front
        new_markov_state = torch.cat([new_env_state, old_markov_state[..., : -self.depth_processed_env_state]], dim=-1)

        return new_markov_state


    def learn(self):
        # Function to train the PPO agent

        # Evaluate the initial performance of policy network before training
        print('Initial demo:')
        total_rewards, avg_traj_len = self.eval(time_steps=self.time_steps_extensive_eval, render=False)
        self.training_stats['init_acc_reward'].append(total_rewards)
        self.training_stats['init_avg_traj_len'].append(avg_traj_len)

        # Start training
        for iteration in range(self.iterations):
            # Each iteration consists of two steps:
            #   1. Collecting new training data
            #   2. Updating nets based on newly generated training data

            print('Iteration:', iteration, "\nEpsilon:", self.epsilon(iteration))

            # Init data collection and storage for current iteration
            num_observed_train_steps = 0
            observations = []
            obs_temp = []

            # Init (parallel) envs
            state = self.init_markov_state(self.env.reset())

            # Collect training data
            with torch.no_grad():
                while num_observed_train_steps < self.T:

                    # Predict action (actually being multiple parallel ones)
                    action = self.policy(state.to(self.device)).cpu()

                    # Perform action in env
                    next_state, reward, terminal_state, _ = self.env.step(action.numpy())

                    # Transform latest observations to tensor data
                    reward = torch.tensor(reward)
                    next_state = self.env2markov(state, next_state)     # Note: state == Markov state, next_state == state as returned by env
                    terminal_state = torch.tensor(terminal_state)       # Boolean indicating whether one of parallel envs has terminated or not

                    # Get log-prob of chosen action (= log \pi_{\theta_{old}}(a_t|s_t) in accompanying report)
                    log_prob = self.policy.log_prob(action.to(self.device)).cpu()

                    # Store observable data
                    observation = (state.unsqueeze(1), action, reward, next_state.unsqueeze(1), terminal_state, log_prob)
                    obs_temp.append(observation)

                    # Add number of newly (in parallel) experienced state transitions to counter
                    num_observed_train_steps += self.parallel_agents


                    # Prepare for next iteration
                    if terminal_state.any() or num_observed_train_steps >= self.T:
                        # Reset env for case where num_observed_train_steps < max_trajectory_length T (in which case a new iteration would follow)
                        state = self.init_markov_state(self.env.reset())

                        # Add temporarily stored observations (made during current iteration) to list of all freshly
                        # observed training data collected so far for next weight update step; to be stored in 'observations':

                        # Compute state value of final observed state (= V(s_T))
                        last_state = obs_temp[-1][3]
                        target_state_val = self.val_net(last_state.to(self.device).squeeze(1)).squeeze().cpu()  # V(s_T)
                        termination_mask = (1 - obs_temp[-1][4].int()).float()  # Only associate last observed state with valuation of 0 if it is terminal
                        target_state_val = target_state_val * termination_mask

                        # Compute the target state value and advantage estimate for each state in agent's trajectory
                        # (batch-wise for all parallel agents in parallel)
                        for t in range(len(obs_temp)-1, -1, -1):
                            # Compute target state value:
                            # V^{target}_t = r_t + \gamma * r_{t+1} + ... + \gamma^{n-1} * r_{t+n-1} + \gamma^n * V(s_{t+n}), where t+n=T
                            target_state_val = obs_temp[t][2] + self.gamma * target_state_val

                            # Compute advantage estimate
                            state_val = self.val_net(obs_temp[t][0].to(self.device).squeeze(1)).squeeze().cpu()  # V(s_t)
                            advantage = target_state_val - state_val

                            # Augment a previously observed observation tuple
                            extra = (target_state_val, advantage)
                            augmented_obs = obs_temp[t] + extra

                            # Add all parallel agents' individual observations to overall (iteration's) observations list
                            for i in range(self.parallel_agents):
                                # Create i^th agent's private observation tuple for time step t in its current trajectory
                                # element \in {state, action, reward, next_state, terminal_state, log_prob, target_val, advantage}
                                single_agent_tuple = tuple([element[i] for element in augmented_obs])
                                observations.append(single_agent_tuple)

                        # Empty temporary list of observations after they have been added to more persistent list of freshly collected train data
                        obs_temp = []

                    else:
                        # Trajectory continues from time step t to t+1 (for all parallel agents)
                        state = next_state

            # Perform weight updates for multiple epochs on freshly collected training data stored in 'observations'
            iteration_loss = 0.
            for epoch in range(self.epochs):
                acc_epoch_loss = 0.   # Loss accumulated over multiple minibatches during epoch

                # Shuffle data
                random.shuffle(observations)  # Shuffle in place!

                # Perform weight update on each minibatch contained in shuffled observations
                for i in range(0, len(observations), self.batch_size):
                    # Reset all gradients
                    self.optimizer_p.zero_grad()
                    self.optimizer_v.zero_grad()

                    # Sample minibatch
                    minibatch = observations[i: i+self.batch_size]

                    # Get all states, actions, log_probs, target_values, and advantage_estimates from minibatch
                    state, action, _, _, _, log_prob_old, target_state_val, advantage = zip(*minibatch)

                    # Transform batch of tuples to batch tensors
                    state_ = torch.vstack(state).to(self.device)      # Minibatch of states
                    target_state_val_ = torch.vstack(target_state_val).squeeze().to(self.device)
                    advantage_ = torch.vstack(advantage).squeeze().to(self.device)
                    log_prob_old_ = torch.vstack(log_prob_old).squeeze().to(self.device)

                    if self.dist_type == DISCRETE:
                        action_ = torch.vstack(action).squeeze().to(self.device)        # Minibatch of actions
                    else:
                        action_ = torch.vstack(action).to(self.device)                  # Minibatch of actions

                    #print("State_ shape:", state_.shape)
                    #print("action_:", action_, action_.shape)
                    #print("log_prob_old_:", log_prob_old_, log_prob_old_.shape)
                    #print("target_state_val_:", target_state_val_, target_state_val_.shape)
                    #print("advantage_:", advantage_, advantage_.shape)

                    # Compute log_prob for minibatch of actions
                    _ = self.policy(state_)
                    log_prob = self.policy.log_prob(action_).squeeze()

                    #print("log_prob_old_:", log_prob_old_, log_prob_old_.shape)
                    #print("log_prob:", log_prob, log_prob.shape)

                    # Compute current state value estimates
                    state_val = self.val_net(state_).squeeze()

                    #print("state_val:", state_val)

                    # Evaluate loss function first component-wise, then combined:
                    # L^{CLIP}
                    L_CLIP = self.L_CLIP(log_prob, log_prob_old_, advantage_, iteration)

                    # L^{V}
                    L_V = self.L_VF(state_val, target_state_val_)

                    if self.dist_type == DISCRETE:
                        # Adding entropy is only needed in discrete case, since standard deviation is fixed in continuous case

                        # H (= Entropy)
                        L_ENTROPY = self.L_ENTROPY()

                        # L^{CLIP + H + V} = L^{CLIP} + h*H + v*L^{V}
                        loss = - L_CLIP - self.h * L_ENTROPY + self.v * L_V

                    else:
                        # L^{CLIP + H + V} = L^{CLIP} + h*H + v*L^{V}
                        loss = - L_CLIP + self.v * L_V

                    # Backprop loss
                    loss.backward()

                    # Perform weight update for both the policy and value net
                    self.optimizer_p.step()
                    self.optimizer_v.step()

                    # Document training progress after one weight update
                    acc_epoch_loss += loss.cpu().detach().numpy()

                # Document training progress after one full epoch/iteration of training
                iteration_loss += acc_epoch_loss
                self.training_stats['devel_epoch_loss'].append(acc_epoch_loss)

                if self.lr_scheduler_pol:
                    self.lr_scheduler_pol.step()

                if self.lr_scheduler_val:
                    self.lr_scheduler_val.step()

            # Document training progress at the end of a full iteration
            self.training_stats['devel_itera_loss'].append(iteration_loss)
            print('Average epoch loss of current iteration:', (iteration_loss/self.epochs))
            print("Current iteration's demo:")
            total_rewards, avg_traj_len = self.eval()
            self.training_stats['train_acc_reward'].append(total_rewards)
            self.training_stats['train_avg_traj_len'].append(avg_traj_len)
            print()

        # Clean up after training
        self.env.close()

        # Final evaluation
        print('Final demo:')
        if self.show_final_demo:
            input("Waiting for user confirmation... Hit ENTER.")
            total_rewards, avg_traj_len = self.eval(time_steps=self.time_steps_extensive_eval, render=True)
        else:
            total_rewards, avg_traj_len = self.eval(time_steps=self.time_steps_extensive_eval, render=False)
        self.training_stats['final_acc_reward'].append(total_rewards)
        self.training_stats['final_avg_traj_len'].append(avg_traj_len)

        return self.training_stats


    def L_CLIP(self, log_prob, log_prob_old, advantage, current_train_iteration):
        # Computes PPO's main objective L^{CLIP}

        prob_ratio = torch.exp(log_prob - log_prob_old)

        unclipped = prob_ratio * advantage
        clipped = torch.clip(prob_ratio, min=1.-self.epsilon(current_train_iteration), max=1.+self.epsilon(current_train_iteration)) * advantage

        return torch.mean(torch.min(unclipped, clipped))


    def L_ENTROPY(self):
        # Computes entropy bonus for policy net
        return torch.mean(self.policy.entropy())


    def L_VF(self, state_val: torch.tensor, target_state_val: torch.tensor):
        # Loss function for state-value network. Quadratic loss between predicted and target state value
        return torch.mean((state_val - target_state_val) ** 2)


    def save_policy_net(self, path_policy: str = './policy_model.pt'):
        torch.save(self.policy, path_policy)
        print('Saved policy net.')


    def save_value_net(self, path_val_net: str = './val_net_model.pt'):
        torch.save(self.val_net, path_val_net)
        print('Saved value net.')


    def load(self, path_policy: str = './policy_model.pt', path_val_net: str = None, train_stats_path: str = None):
        self.policy = torch.load(path_policy)
        self.policy.eval()
        print('Loaded policy net.')

        if path_val_net:
            self.val_net = torch.load(path_val_net)
            self.val_net.eval()
            print('Loaded value net.')

        if train_stats_path:
            with open(train_stats_path) as json_file:
                self.training_stats = json.load(json_file)
            print('Loaded training stats.')


    def save_train_stats(self, path: str = './train_stats.json'):
        with open(path, 'w') as outfile:
            json.dump(self.training_stats, outfile)
        print('Saved training stats.')

    def eval(self, time_steps: int = None, render=False):

        # Let a single agent interact with its env for a given nr of time steps and obtain performance stats

        if time_steps is None:
            time_steps = self.intermediate_eval_steps

        total_rewards = 0.
        total_restarts = 1.

        env = gym.make(self.env_name)
        if self.deterministic_eval:
            env, state, total_rewards = self.random_env_start(env)
        else:
            state = self.init_markov_state(add_batch_dimension(env.reset()))

        # Uncomment for debugging purposes or to see how states are represented:
        #visualize_markov_state(state=state,
        #                       env_state_depth=self.depth_processed_env_state,
        #                       markov_length=self.markov_length,
        #                       color_code='L' if self.grayscale_transform else 'RGB',
        #                       confirm_message='Confirm Eval Init state (1)')

        last_state = state.clone()
        sample_next_action_randomly = False

        with torch.no_grad():  # No need to compute gradients here
            for t in range(time_steps):

                # Select action
                if sample_next_action_randomly:
                    # If simulation is stuck, sample random action to recover simulation
                    action = env.action_space.sample()
                else:
                    # Predict action using policy - Either by stochastic sampling or deterministically
                    if self.deterministic_eval:
                        action = self.policy.forward_deterministic(state.to(self.device)).squeeze().cpu().numpy()
                    else:
                        action = self.policy(state.to(self.device)).squeeze().cpu().numpy()

                # Perform action in env (taking into consideration various input-output behaviors of Gym envs')
                try:
                    # Some envs require actions of format: 1.0034
                    next_state, reward, terminal_state, _ = env.step(action)
                except IndexError:
                    # Some envs require actions of format: [1.0034]
                    next_state, reward, terminal_state, _ = env.step([action])

                # Count accumulative rewards
                total_rewards += reward

                if render and t < min(time_steps, 500):
                    env.render()
                    time.sleep(0.1)

                # Compute new Markov state
                state = self.env2markov(state, add_batch_dimension(next_state))

                # Uncomment for debugging purposed or to see how states are represented:
                #visualize_markov_state(state=state,
                #                       env_state_depth=self.depth_processed_env_state,
                #                       markov_length=self.markov_length,
                #                       color_code='L' if self.grayscale_transform else 'RGB',
                #                       confirm_message='Confirm Eval Updates state (2)')

                if terminal_state:
                    # Simulation has terminated
                    total_restarts += 1

                    # Reset simulation because it has terminated
                    if self.deterministic_eval:
                        env, state, total_rewards = self.random_env_start(env)
                    else:
                        state = self.init_markov_state(add_batch_dimension(env.reset()))

                # Check whether simulation is stuck
                sample_next_action_randomly = simulation_is_stuck(last_state, state)

                # Book-keeping
                last_state = state.clone()

        env.close()

        avg_traj_len = time_steps / total_restarts

        print('Total accumulated reward over', time_steps, 'time steps:', total_rewards)
        print('Average trajectory length in time steps:', avg_traj_len)

        return total_rewards, avg_traj_len
