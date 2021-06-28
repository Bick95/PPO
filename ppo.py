import gym
import json
import time
import random
import torch.optim
import numpy as np
from PIL import Image
from policy import Policy
from value_net import ValueNet
from torchvision.transforms import Resize
from torchvision.transforms import Grayscale
from constants import DISCRETE, CONTINUOUS, INITIAL, INTERMEDIATE, FINAL
from utils_ppo import add_batch_dimension, simulation_is_stuck, visualize_markov_state, \
    get_scheduler, get_optimizer, get_lr_scheduler, get_non_linearity, is_trainable, nan_error, print_nan_error_loss


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
                 standard_dev: float or dict = None,
                 nonlinearity: str = 'relu',
                 markov_length: int = 1,  # How many environmental state get concatenated to one state representation
                 grayscale_transform: bool = False,  # Whether to transform RGB inputs to grayscale or not (if applicable)
                 network_structure: list = None,  # Replacement for hidden parameter,
                 deterministic_eval: bool = False,  # Whether to compute actions stochastically or deterministically throughout evaluation
                 stochastic_eval: bool = False,
                 resize_visual_inputs: tuple = None,
                 dilation: int = 1,  # How many times to execute a generated action
                 frame_duration: float or int = 0.001,
                 max_render_time_steps: int = 5000,
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
        self.stochastic_eval = stochastic_eval
        self.dilation = dilation
        self.frame_duration = float(frame_duration)
        self.max_render_time_steps = max_render_time_steps

        # Epsilon will be a lambda function which always evaluates to the current value for the clipping parameter epsilon
        self.epsilon = get_scheduler(parameter=clipping_parameter,
                                     device=self.device,
                                     train_iterations=self.iterations,
                                     parameter_name="Epsilon",
                                     verbose=True)

        # Set up PyTorch functionality to grayscale visual state representations if required
        self.grayscale_transform = (input_net_type.lower() == "cnn" or input_net_type.lower() == "visual") and grayscale_transform
        self.grayscale_layer = Grayscale(num_output_channels=1) if self.grayscale_transform else None

        # Set up PyTorch functionality to resize visual state representations if required
        self.resize_transform = True if resize_visual_inputs else False
        self.resize_layer = Resize(size=resize_visual_inputs) if resize_visual_inputs else None

        # Set up documentation of training stats
        self.training_stats = {
            # How loss accumulated over one epoch develops over time
            'devel_epoch_loss': [],
            # How loss accumulated over all epochs develops per train iteration
            'devel_itera_loss': [],

            # Total nr of restarts per given trajectory length during initial testing
            'init_total_restarts': [],
            # Total reward accumulated over given trajectory length during initial testing
            'init_acc_reward': [],

            # Total nr of restarts per given trajectory length during intermediate testing
            'train_total_restarts': [],
            # Total reward accumulated over given trajectory length during intermediate testing
            'train_acc_reward': [],
        }

        if self.deterministic_eval:
            # Total nr of restarts per given trajectory length during final testing when using deterministic generation of actions
            self.training_stats['final_det_total_restarts'] = []
            # Total reward accumulated over given trajectory length during final testing when using deterministic generation of actions
            self.training_stats['final_det_acc_reward'] = []

        if self.stochastic_eval:
            # Total nr of restarts per given trajectory length during final testing when using stochastic generation of actions
            self.training_stats['final_stoch_total_restarts'] = []
            # Total reward accumulated over given trajectory length during final testing when using stochastic generation of actions
            self.training_stats['final_stoch_acc_reward'] = []

        # Assign functional nonlinearity
        nonlinearity = get_non_linearity(nonlinearity)

        # Create Gym env if not provided as such
        if isinstance(env, str):
            env = gym.make(env)

        if sum(env.action_space.shape) > 0:
            raise NotImplementedError("This implementation supports only action spaces with a single action to be predicted per time step.")

        self.env_name = env.unwrapped.spec.id
        self.action_space = env.action_space
        self.dist_type = DISCRETE if isinstance(self.action_space, gym.spaces.Discrete) else CONTINUOUS

        # Obtain information about the size of the portion of the Markov state to be dropped at the end when updating Markov state
        self.depth_processed_env_state = self.preprocess_env_state(add_batch_dimension(gym.make(self.env_name).reset())).shape[-1]

        observation_sample = self.sample_observation_space()

        if standard_dev is None:
            standard_dev = 1.

        # Create policy net
        self.policy = Policy(action_space=self.action_space,
                             observation_sample=observation_sample,
                             input_net_type=self.input_net_type,
                             device=self.device,
                             nonlinearity=nonlinearity,
                             standard_dev=standard_dev,
                             dist_type=self.dist_type,
                             network_structure=network_structure,
                             train_iterations=self.iterations,
                             ).to(device=self.device)

        # Determine whether standard deviation is a trainable parameter or not when facing continuous action space
        if self.dist_type is CONTINUOUS and is_trainable(standard_dev):
            self.std_trainable = True
        else:
            self.std_trainable = False

        # Save whether to decay standard deviation or not
        if isinstance(standard_dev, float) or self.std_trainable:
            # Don't decay if standard deviation is a constant or trainable
            self.decay_standard_dev = False
        else:
            self.decay_standard_dev = True

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

        self.optimizer_p = get_optimizer(learning_rate=learning_rate_pol, model_parameters=self.policy.parameters())
        self.lr_scheduler_pol = get_lr_scheduler(learning_rate=learning_rate_pol, optimizer=self.optimizer_p,
                                                 train_iterations=self.iterations)

        self.optimizer_v = get_optimizer(learning_rate=learning_rate_val, model_parameters=self.val_net.parameters())
        self.lr_scheduler_val = get_lr_scheduler(learning_rate=learning_rate_val, optimizer=self.optimizer_v,
                                                 train_iterations=self.iterations)


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
        self.eval_and_log(eval_type=INITIAL)

        # Start training
        for iteration in range(self.iterations):
            # Each iteration consists of two steps:
            #   1. Collecting new training data
            #   2. Updating nets based on newly generated training data

            print('Iteration:', iteration+1, "of", self.iterations, "\nEpsilon: {:.2f}".format(self.epsilon.value))

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

                    # Perform action in env (self.dilation times)
                    accumulated_reward = torch.zeros((self.parallel_agents,))
                    for _ in range(self.dilation):
                        next_state, reward, terminal_state, _ = self.env.step(action.numpy())

                        # Sum rewards over time steps
                        accumulated_reward += reward

                        # If either env is done, step repeating actions
                        if terminal_state.any():
                            break

                    # Transform latest observations to tensor data
                    reward = accumulated_reward
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
                            target_state_val = obs_temp[t][2] + self.gamma * target_state_val  # <- reward r_t obtained in state s_t + discounted future reward

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
                    state_ = torch.vstack(state).to(self.device)         # Minibatch of states
                    target_state_val_ = torch.vstack(target_state_val).squeeze().to(self.device)
                    advantage_ = torch.vstack(advantage).squeeze().to(self.device)
                    log_prob_old_ = torch.vstack(log_prob_old).squeeze().to(self.device)

                    if self.dist_type == DISCRETE:
                        action_ = torch.vstack(action).squeeze().to(self.device)  # Minibatch of actions
                    else:
                        action_ = torch.vstack(action).to(self.device)            # Minibatch of actions

                    # Compute log_prob for minibatch of actions
                    _ = self.policy(state_)
                    log_prob = self.policy.log_prob(action_).squeeze()

                    # Compute current state value estimates
                    state_val = self.val_net(state_).squeeze()

                    # Evaluate loss function first component-wise, then combined:
                    # L^{CLIP}
                    L_CLIP = self.L_CLIP(log_prob, log_prob_old_, advantage_)

                    # L^{V}
                    L_V = self.L_VF(state_val, target_state_val_)

                    if self.dist_type == DISCRETE or self.std_trainable:
                        # An entropy bonus is added only if the agent faces a discrete action space or if we manually
                        # declare that the standard deviation is trainable in continuous action spaces

                        # H (= Entropy)
                        L_ENTROPY = self.L_ENTROPY()

                        # L^{CLIP + H + V} = L^{CLIP} + h*H + v*L^{V}
                        loss = - L_CLIP - self.h * L_ENTROPY + self.v * L_V

                    else:
                        # L^{CLIP + H + V} = L^{CLIP} + h*H + v*L^{V}
                        loss = - L_CLIP + self.v * L_V

                    if nan_error(loss):
                        if self.dist_type == DISCRETE or self.std_trainable:
                            print_nan_error_loss(loss, L_CLIP, L_V, action_, log_prob, log_prob_old_, state_, state_val, L_ENTROPY)
                        else:
                            print_nan_error_loss(loss, L_CLIP, L_V, action_, log_prob, log_prob_old_, state_, state_val)
                        raise OverflowError('Loss is nan. See print statement above.')

                    # Backprop loss
                    loss.backward()

                    # Perform weight update for both the policy and value net
                    self.optimizer_p.step()
                    self.optimizer_v.step()

                    # Document training progress after one weight update
                    acc_epoch_loss += loss.cpu().detach().numpy()

                # Document training progress after one full epoch/iteration of training
                iteration_loss += acc_epoch_loss
                self.log('devel_epoch_loss', acc_epoch_loss)

            # Potentially decrease learning rate every iteration
            if self.lr_scheduler_pol:
                self.lr_scheduler_pol.step()

            if self.lr_scheduler_val:
                self.lr_scheduler_val.step()

            if self.decay_standard_dev:
                self.policy.std.step()

            # Potentially decrease epsilon
            self.epsilon.step()

            # Document training progress at the end of a full iteration
            self.log_train_stats(iteration_loss=iteration_loss)
            self.eval_and_log(eval_type=INTERMEDIATE)
            print()

        # Clean up after training
        self.env.close()

        # Final evaluation
        self.eval_and_log(eval_type=FINAL)

        return self.training_stats


    def L_CLIP(self, log_prob, log_prob_old, advantage):
        # Computes PPO's main objective L^{CLIP}

        prob_ratio = torch.exp(log_prob - log_prob_old)

        unclipped = prob_ratio * advantage
        clipped = torch.clip(prob_ratio, min=1.-self.epsilon.value, max=1.+self.epsilon.value) * advantage

        return torch.mean(torch.min(unclipped, clipped))


    def L_ENTROPY(self):
        # Computes entropy bonus for policy net
        return torch.mean(self.policy.entropy())


    def L_VF(self, state_val: torch.tensor, target_state_val: torch.tensor):
        # Loss function for state-value network. Quadratic loss between predicted and target state value
        return torch.mean((state_val - target_state_val) ** 2)


    def save_policy_net(self, path_policy: str = './policy_model.pt'):
        del self.policy.std  # Scheduler object can't be saved as part of a model; would break the saving process
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


    def log(self, key, value):
        self.training_stats[key].append(value)


    def eval_and_log(self, eval_type: int):

        if eval_type == INITIAL:
            print('Initial evaluation:')
            total_rewards, _, total_restarts = self.eval(time_steps=self.time_steps_extensive_eval, render=False)
            self.log('init_acc_reward', total_rewards)
            self.log('init_total_restarts', total_restarts)

        elif eval_type == INTERMEDIATE:
            print("Current iteration's demo:")
            total_rewards, _, total_restarts = self.eval()
            self.log('train_acc_reward', total_rewards)
            self.log('train_total_restarts', total_restarts)

        elif eval_type == FINAL:
            print('Final demo(s):')
            # Final demos can be both deterministic and stochastic in order to demonstrate the difference between the two
            if self.deterministic_eval:
                print('Going to perform deterministic evaluation.')
                if self.show_final_demo:
                    # Wait for user to confirm that (s)he is ready to witness the final demo
                    input("Waiting for user confirmation... Hit ENTER.")
                    total_rewards, _, total_restarts = self.eval(time_steps=self.time_steps_extensive_eval, render=True, deterministic_eval=True)
                else:
                    total_rewards, _, total_restarts = self.eval(time_steps=self.time_steps_extensive_eval, render=False, deterministic_eval=True)
                self.log('final_det_acc_reward', total_rewards)
                self.log('final_det_total_restarts', total_restarts)

            if self.stochastic_eval:
                print('Going to perform stochastic evaluation.')
                if self.show_final_demo:
                    # Wait for user to confirm that (s)he is ready to witness the final demo
                    input("Waiting for user confirmation... Hit ENTER.")
                    total_rewards, _, total_restarts = self.eval(time_steps=self.time_steps_extensive_eval, render=True, deterministic_eval=False)
                else:
                    total_rewards, _, total_restarts = self.eval(time_steps=self.time_steps_extensive_eval, render=False, deterministic_eval=False)
                self.log('final_stoch_acc_reward', total_rewards)
                self.log('final_stoch_total_restarts', total_restarts)


    def log_train_stats(self, iteration_loss):
        self.log('devel_itera_loss', iteration_loss)
        print('Total epoch loss of current iteration: {:.2f}'.format(iteration_loss))
        print('Average epoch loss of current iteration: {:.2f}'.format((iteration_loss / self.epochs)))


    def eval(self, time_steps: int = None, render: bool = False, deterministic_eval: bool = True):

        # Let a single agent interact with its env for a given nr of time steps and obtain performance stats

        if time_steps is None:
            time_steps = self.intermediate_eval_steps

        total_rewards = 0.
        total_restarts = 1

        # Make testing environment
        env = gym.make(self.env_name)

        # Initialize testing environment - either for deterministic evaluation or for stochastic evaluation (depending
        # on setting)
        if deterministic_eval:
            env, state, total_rewards = self.random_env_start(env)
        else:
            state = self.init_markov_state(add_batch_dimension(env.reset()))

        # Uncomment for debugging purposes or to see how states are represented in visual state observation setting:
        #visualize_markov_state(state=state,
        #                       env_state_depth=self.depth_processed_env_state,
        #                       markov_length=self.markov_length,
        #                       color_code='L' if self.grayscale_transform else 'RGB',
        #                       confirm_message='Confirm Eval Init state (1)')

        last_state = state.clone()

        # Flag indicating whether next action has to be sampled randomly to recover from stuck simulation state
        sample_next_action_randomly = False

        with torch.no_grad():  # No need to compute gradients here
            for t in range(time_steps):  # Run the simulation for some time steps

                # Select action
                if sample_next_action_randomly:
                    # If simulation is stuck, sample random action to recover simulation
                    action = env.action_space.sample()
                else:
                    # Predict action using policy - Either by stochastic sampling or deterministically
                    if deterministic_eval:
                        action = self.policy.forward_deterministic(state.to(self.device)).squeeze().cpu().numpy()
                    else:
                        action = self.policy(state.to(self.device)).squeeze().cpu().numpy()

                # Perform action in env (taking into consideration various input-output behaviors of Gym envs')
                accumulated_reward = 0.
                try:
                    for _ in range(self.dilation):
                        # Some envs require actions of format: 1.0034
                        next_state, reward, terminal_state, _ = env.step(action)

                        # Sum rewards over time steps
                        accumulated_reward += reward

                        # If env is done, i.e. simulation has terminated, stop repeating actions
                        if terminal_state:
                            break

                except IndexError:
                    for _ in range(self.dilation):
                        # Some envs require actions of format: [1.0034]
                        next_state, reward, terminal_state, _ = env.step([action])

                        # Sum rewards over time steps
                        accumulated_reward += reward

                        # If env is done, i.e. simulation has terminated, stop repeating actions
                        if terminal_state:
                            break

                # Count accumulative rewards
                total_rewards += accumulated_reward

                if render and t < min(time_steps, self.max_render_time_steps):
                    env.render()
                    time.sleep(self.frame_duration)

                # Compute new Markov state
                state = self.env2markov(state, add_batch_dimension(next_state))

                # Uncomment for debugging purposes or to see how states are represented in visual state observation setting:
                #visualize_markov_state(state=state,
                #                       env_state_depth=self.depth_processed_env_state,
                #                       markov_length=self.markov_length,
                #                       color_code='L' if self.grayscale_transform else 'RGB',
                #                       confirm_message='Confirm Eval Updates state (2)')

                if terminal_state:
                    # Simulation has terminated
                    total_restarts += 1

                    # Reset simulation because it has terminated
                    if deterministic_eval:
                        env, state, reward = self.random_env_start(env)

                        # Sum rewards over time steps
                        total_rewards += reward

                    else:
                        state = self.init_markov_state(add_batch_dimension(env.reset()))

                # Check whether simulation is stuck: If simulation is stuck in same state, the agent needs to sample
                # some different action during the next time step, thus random sampling...
                sample_next_action_randomly = simulation_is_stuck(last_state, state)

                # Book-keeping
                last_state = state.clone()

        env.close()

        avg_traj_len = time_steps / total_restarts

        print('Total accumulated reward over', time_steps, 'time steps: {:.2f}'.format(total_rewards))
        print('Average trajectory length in time steps given', total_restarts, 'restarts: {:.2f}'.format(avg_traj_len))

        return total_rewards, avg_traj_len, total_restarts
