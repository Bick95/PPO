import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from scheduler import Scheduler
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR


def add_batch_dimension(state: np.ndarray):
    return np.expand_dims(state, axis=0)


def simulation_is_stuck(last_state, state):
    # If two consecutive states are absolutely identical, then we assume the simulation to be stuck in a semi-terminal state
    # Returns True if two states are identical, else False

    return torch.eq(last_state, state).all()


def visualize_markov_state(state: np.ndarray or torch.tensor,
                     env_state_depth: int,
                     markov_length: int,
                     color_code: str = 'RGB',
                     confirm_message: str = "Confirm..."):

    if isinstance(state, torch.Tensor):
        state = state.numpy()    # Convert to numpy array

    if len(state.shape) > 3:
        state = state.squeeze()  # Drop batch dimension

    # Get contained environmental state representations
    images = []

    for i in range(markov_length):
        extracted_env_state = state[:, :, i*env_state_depth : (i+1)*env_state_depth].squeeze()
        temp_image = Image.fromarray(extracted_env_state.astype('uint8'), color_code)
        images.append(temp_image)

    # Create empty image container
    image = Image.new(color_code, (images[0].width * markov_length, images[0].height * markov_length))

    # Add individual images
    for i in range(markov_length):
        image.paste(images[i], (i * images[0].width, 0))

    image.show()
    input(confirm_message)


def get_epsilon_scheduler(clipping_parameter: float or dict, device: torch.device, train_iterations: int,
                          parameter_name: str = None, verbose: bool = False):
    # Determine how to handle clipping constant - keep it constant or anneal from some max value to some min value
    # Do the scheduling via a scheduler

    if isinstance(clipping_parameter, float):
        return Scheduler(clipping_parameter, 'constant', device, value_name=parameter_name, verbose=verbose)

    elif isinstance(clipping_parameter, dict):
        # Anneal clipping parameter between some values (from max to min)
        initial_value = clipping_parameter['max'] if 'max' in clipping_parameter.keys() else 1.
        min_value = clipping_parameter['min'] if 'min' in clipping_parameter.keys() else 0.

        # Decay type
        decay_type = clipping_parameter['decay_type'].lower() if 'decay_type' in clipping_parameter.keys() else 'linear'

        # Decay rate
        decay_rate = clipping_parameter['decay_rate'].lower() if 'decay_rate' in clipping_parameter.keys() else None

        # Decay steps
        decay_steps = clipping_parameter['decay_steps'].lower() if 'decay_steps' in clipping_parameter.keys() else train_iterations

        return Scheduler(
            initial_value=initial_value,
            decay_type=decay_type,
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            device=device,
            min_value=min_value,
            value_name=parameter_name,
        )

    else:
        raise NotImplementedError("clipping_parameter mist be float or dict")


def get_non_linearity(nonlinearity):
    if nonlinearity.lower() == 'relu':
        return F.relu
    elif nonlinearity.lower() == 'sigmoid':
        return F.sigmoid
    elif nonlinearity.lower() == 'tanh':
        return F.tanh
    else:
        raise NotImplementedError("Only relu ")


def get_optimizer(learning_rate: float or dict, model_parameters):
    if isinstance(learning_rate, float):
        # Simple optimizer with constant learning rate for neural net
        return torch.optim.Adam(params=model_parameters, lr=learning_rate)

    elif isinstance(learning_rate, dict):
        # Create optimizer plus a learning rate scheduler associated with optimizer
        return torch.optim.Adam(params=model_parameters, lr=learning_rate['initial'])

    else:
        raise NotImplementedError("learning_rate must be (constant) float or dict.")


def get_lr_scheduler(learning_rate: float or dict, optimizer, iterations: int):

    if isinstance(learning_rate, float):
        # Simple optimizer with constant learning rate for neural net, thus no scheduler needed
        return None

    elif isinstance(learning_rate, dict):
        # Whether learning rate scheduler shall print feedback or not
        verbose = learning_rate['verbose'] if 'verbose' in learning_rate.keys() else False

        if learning_rate['decay_type'].lower() == 'linear':
            lambda_lr = lambda epoch: (iterations - epoch) / iterations
            return LambdaLR(optimizer, lr_lambda=lambda_lr, verbose=verbose)

        elif learning_rate['decay_type'].lower() == 'exponential':
            decay_factor = learning_rate['decay_factor'] if 'decay_factor' in learning_rate.keys() else 0.9
            return ExponentialLR(optimizer, gamma=decay_factor, verbose=verbose)

        else:
            raise NotImplementedError("Learning rate decay may only be linear or exponential.")

    else:
        raise NotImplementedError("learning_rate_pol must be (constant) float or dict.")
