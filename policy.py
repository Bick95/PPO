import gym
from input_net_modules import InMLP, InCNN
from output_net_modules import OutMLP
from constants import DISCRETE, CONTINUOUS

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


# TODO: CONTINUE proof-reading/testing here!
# TODO: ADD RNN CAPABILITY HERE


class Policy(nn.Module):

    def __init__(self,
                 action_space: gym.spaces.Discrete or gym.spaces.Box,
                 observation_space: gym.Space,
                 input_net_type: str = 'CNN'
                 ):

        super(Policy, self).__init__()

        # Save some data
        self.observation_space = observation_space
        self.action_space = action_space
        self.dist_type = DISCRETE if isinstance(self.action_space, gym.spaces.Discrete) else CONTINUOUS

        # Determine whether output distribution is to be Discrete or Continuous
        # NOTE!
        # In Continuous case, n actions may be sampled concurrently per env.
        # In Discrete case, only one action (out of n options) is sampled at a time from the action space in a given env.
        # Thus, meaning of self.num_actions varies between these two output spaces/cases!
        if self.dist_type is DISCRETE:
            self.num_actions = self.action_space.n
        else:
            # Assumption: no flattening needed!
            self.num_actions = action_space.shape[0]

        print('Dist type:', self.dist_type)
        print('Action space:', self.action_space)
        print('Num actions:', self.num_actions)

        # Assign input layer possibly consisting of multiple internal layers; Design dependent on nature of state observations
        if input_net_type.lower() == 'cnn' or input_net_type.lower() == 'visual':
            # Create CNN-NN to encode inputs
            self.input_module = None  # TODO: InCNN

        else:
            # Compute nr of input features for given gym env
            input_features = sum(self.observation_space.sample().shape)

            # Create MLP-NN to encode inputs
            self.input_module = InMLP(input_features)  # TODO: assign config params


        print('Observation space:', self.observation_space)
        print('Input features:', self.observation_space.sample().shape, sum(self.observation_space.sample().shape))

        # Assign (deterministic) output layer for generating parameterizations of probability distributions over action space to be defined below
        self.output_module = OutMLP(input_features=50,
                                    output_features=self.num_actions,
                                    output_type=self.dist_type  # TODO: more configgs?
                                    )

        # Assign stochastic probability distribution (generator) for sampling actions
        self.prob_dist = Categorical if self.dist_type is DISCRETE else Normal

        # Construct the deterministic processing pipeline of the policy net
        self.pipeline = [
            self.input_module,
            self.output_module
        ]

        # To be assigned later
        self.dist = None  # Will contain concrete probability distributions for sampling actions


    def forward(self, x: torch.tensor):

        print('Input:\n', x)
        for layer in self.pipeline:
            x = layer(x)
            print('Modified x:\n', x)
            print('Modified by layer\n:', layer)
            print()

        if self.dist_type is DISCRETE:
            self.dist = self.prob_dist(probs=x)
        else:
            self.dist = self.prob_dist(loc=x, scale=torch.ones(self.num_actions))  # TODO: make variable
        print('Prob dist:\n', self.dist)

        action = self.dist.sample()
        print('Actions sampled:\n', action)

        return action


    def log_prob(self, action):
        return self.dist.log_prob(action)


    def entropy(self):
        return self.dist.entropy()


    def get_non_output_layers(self):
        return self.input_module
