import torch.nn as nn
from constants import DISCRETE


class OutMLP(nn.Module):
    # This module implicitly assumes that only one action is to be predicted per time step.
    # It computes the parameterization for some probability distribution

    def __init__(self,
                 output_features: int,
                 input_features: int = 50,
                 output_type: int = DISCRETE
                 ):
        super(OutMLP, self).__init__()

        # Construct NN-processing pipeline consisting of concatenation of layers to be applied to any input
        # (Using this pipeline-approach avoids if-statements for determining whether to apply the Softmax in forward pass)
        self.pipeline = [

            # Add output layer
            nn.Linear(
                in_features=input_features,
                out_features=output_features
            )

        ]

        # Register all layers
        for i, layer in enumerate(self.pipeline):
            self.add_module("layer_mlp_out_" + str(i), layer)

        # Add optional normalization of outputs in case of Discrete distribution over action space
        if output_type is DISCRETE:
            self.pipeline.append(nn.Softmax(dim=1))

    def forward(self, x):
        # Forward pass to compute the parameterization for the probability distribution following the policy network

        for layer in self.pipeline:
            x = layer(x)

        return x
