import gym
from input_net_modules import InMLP, InCNN
from output_net_modules import OutMLP
from constants import DISCRETE, CONTINUOUS

import torch
import torch.nn as nn


class ValueNet(nn.Module):

    def __init__(self,
                 observation_space: gym.Space = None,
                 input_net_type: str = 'CNN',
                 shared_layers: torch.nn.Module = None
                 ):

        super(ValueNet, self).__init__()

        self.observation_space = observation_space

        if shared_layers is None:

            if input_net_type.lower() == 'cnn' or input_net_type.lower() == 'visual':
                # Create CNN-NN to encode inputs
                self.input_module = None  # TODO: InCNN

            else:
                # Compute nr of input features for given gym env
                input_features = sum(self.observation_space.sample().shape)

                # Create MLP-NN to encode inputs
                self.input_module = InMLP(input_features)

        else:
            self.input_module = shared_layers

        self.output_module = OutMLP(hidden_features=50,
                                    output_features=1,
                                    output_type=CONTINUOUS
                                    )


    def forward(self, x: torch.tensor):

        hidden = self.input_module(x)
        state_value = self.output_module(hidden)

        return state_value


    def get_non_output_layers(self):
        return self.input_module
