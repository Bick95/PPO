import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: Add RNN input module


class InCNN(nn.Module):

    def __init__(self,
                 input_sample: torch.tensor,
                 nonlinearity: torch.nn.functional = F.relu,
                 network_structure: list = None,
                 ):
        super(InCNN, self).__init__()

        # Determine dimensions of observation (input) sample
        data_height = input_sample.shape[0]
        data_width = input_sample.shape[1]

        # The number of input channels = nr of color channels times nr of stacked environmental states used to get one Markov state:
        if len(input_sample.shape) is 3:
            # There is a color/Markov-channel-dimension
            in_channels = input_sample.shape[2]
        else:
            # There is no third dimension along which different color channels can be indexed
            in_channels = 1

        print("In InCNN model:")
        print("Shape input sample:", input_sample.shape)
        print("Input sample - Height: %i, Width: %i, Color/Markov channels: %i" % (data_height, data_width, in_channels))

        # Default setup of network
        if network_structure is None:

            network_structure = [
                # Dicts for conv layers
                {
                    'in_channels': in_channels,
                    'out_channels': 32,  # 16 output channels = 16 filters
                    'kernel_size': 4,  # 8x8 kernel/filter size
                    'stride': 1
                },
                {
                    'in_channels': 32,
                    'out_channels': 16,
                    'kernel_size': 8,
                    'stride': 2
                },
                # Nr. of nodes for fully connected layers
                256
            ]
        else:
            raise NotImplementedError("TODO: Implement augmentation of network structure by in_channels (+ sth else?)!")
            # TODO: augment network structure by input channels etc.
            # first dict element: in_channels = in_channels
            # for each dict element:
            #    in_channels = out_channels of previous dict element

        # Set up processing pipeline (i.e. policy)
        self.pipeline = []

        for i, layer_specs in enumerate(network_structure):
            if isinstance(layer_specs, dict):
                # Add conv layer
                self.pipeline.append(
                    nn.Conv2d(in_channels=layer_specs['in_channels'],
                              out_channels=layer_specs['out_channels'],
                              kernel_size=layer_specs['kernel_size'],
                              stride=layer_specs['stride']
                              )
                )

            elif isinstance(layer_specs, int) and isinstance(network_structure[i-1], dict):
                # Add flattening before transitioning from conv to FC

                # Compute output size of previous conv layers to determine size of conv-output to be flattened
                for l in range(i):
                    data_height = self.out_dim(dim_in=data_height,
                                               pad=self.extract_params_from_structure(structure=network_structure, index=l, key='padding', vertical_dim=True, default=0),
                                               dial=self.extract_params_from_structure(structure=network_structure, index=l, key='dilation', vertical_dim=True, default=1),
                                               k=self.extract_params_from_structure(structure=network_structure, index=l, key='kernel_size', vertical_dim=True, default=4),
                                               stride=self.extract_params_from_structure(structure=network_structure, index=l, key='stride', vertical_dim=True, default=1))
                    data_width = self.out_dim(dim_in=data_width,
                                              pad=self.extract_params_from_structure(structure=network_structure, index=l, key='padding', vertical_dim=False, default=0),
                                              dial=self.extract_params_from_structure(structure=network_structure, index=l, key='dilation', vertical_dim=False, default=1),
                                              k=self.extract_params_from_structure(structure=network_structure, index=l, key='kernel_size', vertical_dim=False, default=4),
                                              stride=self.extract_params_from_structure(structure=network_structure, index=l, key='stride', vertical_dim=False, default=1))

                data_height, data_width = int(data_height), int(data_width)

                flattened_size = data_height * data_width * network_structure[i-1]['out_channels']

                # Add flattening layer
                self.pipeline.append(
                    nn.Flatten(start_dim=1)
                )

                # Add actual FC layer
                self.pipeline.append(
                    nn.Linear(flattened_size, network_structure[i])
                )

            else:
                # Add FC layer after a previous one has been added already
                self.pipeline.append(
                    nn.Linear(network_structure[i-1], network_structure[i])
                )

        # Register all layers
        for i, layer in enumerate(self.pipeline):
            self.add_module("layer_cnn_in_" + str(i), layer)

        self.nonlinearity = nonlinearity


    @staticmethod
    def extract_params_from_structure(structure: list, index: int, key: str, vertical_dim: bool, default: int):
        # Takes the specification of the structure of a NN and returns a requested element from it
        if key in structure[index].keys():
            # Requested specification is provided
            if isinstance(structure[index][key], int):
                # Same specification for both vertical and horizontal case
                return structure[index][key]
            elif isinstance(structure[index][key], tuple):
                # Different specifications for vertical and horizontal axes respectively
                return structure[index][key][0 if vertical_dim else 1]
            else:
                # Assumed different specifications for vertical and horizontal axes respectively, just not in tuple format
                return tuple(structure[index][key])[0 if vertical_dim else 1]
        else:
            # Requested specification is not provided
            return default


    @staticmethod
    def out_dim(dim_in, pad, dial, k, stride):
        #print(dim_in, pad, dial, k, stride)
        return torch.floor(torch.tensor((dim_in + 2*pad - dial*(k-1) - 1)/stride + 1)).numpy()

    def forward(self, x):

        # Change dimensionality from (B, H, W, C) to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        for layer in self.pipeline:
            x = self.nonlinearity(layer(x))

        return x


class InMLP(nn.Module):
    def __init__(self,
                 input_features: int,
                 hidden_nodes: int or list = [50, 50, 50],
                 nonlinearity: torch.nn.functional = F.relu,
                 markov_length: int = 4,                    # TODO: remove this parameter. It's obsolete!
                 ):
        super(InMLP, self).__init__()

        print("In InMLP:")
        print("input_features:", input_features)
        print("markov_channels:", markov_length)

        # Construct NN-processing pipeline consisting of concatenation of layers to be applied to any input
        self.pipeline = [

            # Add input layer
            nn.Linear(
                input_features * markov_length,
                hidden_nodes[0] if isinstance(hidden_nodes, list) else hidden_nodes
            )

        ]

        # Add optional hidden layers
        if isinstance(hidden_nodes, list):
            for i in range(1, len(hidden_nodes)):
                self.pipeline.append(
                    nn.Linear(hidden_nodes[i-1], hidden_nodes[i])
                )

        # Register all layers
        for i, layer in enumerate(self.pipeline):
            self.add_module("layer_mlp_in_" + str(i), layer)

        self.nonlinearity = nonlinearity


    def forward(self, x):

        for layer in self.pipeline:
            x = self.nonlinearity(layer(x))

        return x


# Testing
#import gym
#env = gym.make('CartPole-v1')
#observation = env.observation_space.sample()
#observation_shape = torch.tensor(observation.shape)
#net = MLP_Module(observation_shape[0])
#net(torch.tensor(observation))
