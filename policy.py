import gym
from scheduler import Scheduler
import torch.nn.functional as F
from input_net_modules import InMLP, InCNN
from output_net_modules import OutMLP
from constants import DISCRETE, CONTINUOUS
from ppo_utils import get_scheduler, is_trainable, is_provided

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class Policy(nn.Module):

    def __init__(self,
                 dist_type: int,
                 action_space: gym.spaces.Discrete or gym.spaces.Box,
                 observation_sample: torch.tensor,
                 device: torch.device,
                 train_iterations: int,
                 standard_dev: float or dict,
                 input_net_type: str = 'CNN',
                 nonlinearity: torch.nn.functional = F.relu,
                 network_structure: list = None,
                 ):

        super(Policy, self).__init__()


        # Determine whether output distribution is to be Discrete or Continuous
        self.dist_type = dist_type
        # NOTE!
        # In Continuous case, n actions may be sampled concurrently per env.
        # In Discrete case, only one action (out of n options) is sampled at a time from the action space in a given env.
        # Thus, meaning of self.num_actions varies between these two output spaces/cases!
        if self.dist_type is DISCRETE:
            self.num_actions = action_space.n
        else:
            # Assumption: no flattening needed!
            self.num_actions = action_space.shape[0]


        # Set a flag indicating whether std is supposed to be trainable rather than a constant or to be annealed instead
        self.std_trainable = is_provided(standard_dev) and is_trainable(standard_dev)

        if not self.std_trainable:
            # Get a scheduler for the standard deviation parameter in case of continuous action spaces
            self.std = get_scheduler(parameter=standard_dev,
                                     device=device,
                                     train_iterations=train_iterations,
                                     parameter_name="Standard Deviation",
                                     verbose=True)
        else:
            # If standard deviation is trainable, we don't need a scheduler for it
            self.std = None


        # Assign input layer possibly consisting of multiple internal layers; Design dependent on nature of state observations
        if input_net_type.lower() == 'cnn' or input_net_type.lower() == 'visual':
            # Create CNN to encode inputs
            self.input_module = InCNN(
                network_structure=network_structure,
                input_sample=observation_sample,
                nonlinearity=nonlinearity,
            )

        else:
            # Create MLP-NN to encode inputs
            self.input_module = InMLP(network_structure=network_structure,
                                      input_sample=observation_sample,
                                      nonlinearity=nonlinearity,
                                      )

        # Automatically determine how many input nodes the output module/layer is gonna need to have
        input_features_output_module = self.input_module._modules[next(reversed(self.input_module._modules))].out_features

        # Automatically determine how many output nodes the output module/layer is gonna need to have
        output_features_output_module = self.num_actions if not self.std_trainable else 2 * self.num_actions

        # Assign (deterministic) output layer for generating parameterizations of probability distributions over action space to be defined below
        self.output_module = OutMLP(input_features=input_features_output_module,
                                    output_features=output_features_output_module,
                                    output_type=self.dist_type
                                    )

        # Assign stochastic probability distribution (including sampler) for sampling actions
        self.prob_dist = Categorical if self.dist_type is DISCRETE else Normal

        # Construct the deterministic processing pipeline of the policy net
        self.pipeline = [
            self.input_module,
            self.output_module
        ]

        # To be assigned later (during execution). Will contain parameterized distribution (one per parallel agent)
        self.dist = None  # Will contain concrete probability distributions for sampling actions


    def forward(self, x: torch.tensor):

        for layer in self.pipeline:
            x = layer(x)

        if self.dist_type is DISCRETE:
            self.dist = self.parameterize_discrete_distribution(x)

        else:
            self.dist = self.parameterize_continuous_distribution(x)

        action = self.dist.sample()

        return action


    def parameterize_discrete_distribution(self, x):
        # Discrete action space: x contains vector of probability masses (per minibatch example) which is used to
        # parameterize a respective categorical (i.e. multinomial) distribution
        return self.prob_dist(probs=x)


    def parameterize_continuous_distribution(self, x):
        # Continuous action space: x contains either only the means for Gaussians or the means and
        # the respective standard deviations if the latter is trainable

        if self.std_trainable:
            # If standard deviation is trainable, only the first half of x contains the means, second half contains stds
            mean = x[:, :self.num_actions]
            std = torch.exp(x[:, self.num_actions:])  # Std deviation cannot be negative, thus the exponential function
            return self.prob_dist(loc=mean, scale=std)

        else:
            # x contains only the means, while stds are provided by scheduler
            return self.prob_dist(loc=x, scale=self.std.get_value(x.shape[0]))

    def forward_deterministic(self, x: torch.tensor):
        # Executes a standard forward pass through the policy, but then does not sample actions, but determines them deterministically

        for layer in self.pipeline:
            x = layer(x)

        if self.dist_type is DISCRETE:
            #print("Parameterization discrete:", x)
            action = torch.argmax(x, dim=-1)  # Choose for every minibatch example the action with largest probability mass
        else:
            #print("Parameterization continuous:", x)

            if self.std_trainable:
                # If standard deviation is trainable, only the first half of x contains the means, i.e. the deterministic actions
                action = x[:, :self.num_actions]
                #print('action_:', action)
            else:
                action = x  # x contains the mean of a Gaussian per minibatch example. It's like sampling with 0 standard deviation

        #print("Action:", action)
        return action


    def log_prob(self, action):
        return self.dist.log_prob(action)


    def entropy(self):
        return self.dist.entropy()


    def get_non_output_layers(self):
        return self.input_module
