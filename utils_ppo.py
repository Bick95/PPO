import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from scheduler import Scheduler
from lr_scheduler import CustomLRScheduler
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


def get_scheduler(parameter: float or dict, device: torch.device, train_iterations: int,
                  parameter_name: str = None, verbose: bool = False):
    # Schedulers are used to decay parameters (e.g. the clipping parameter epsilon or the (non-fixed-)standard deviation.
    # This method takes some specification of how a scheduler is supposed to decay a given parameter and returns a
    # suitable scheduler

    if isinstance(parameter, float):
        return Scheduler(parameter, 'constant', device, value_name=parameter_name, verbose=verbose)

    elif isinstance(parameter, dict):
        # Anneal clipping parameter between some values (from max to min) - Create a scheduler for that

        initial_value = parameter['initial_value'] if 'initial_value' in parameter.keys() else None
        min_value = parameter['min_value'] if 'min_value' in parameter.keys() else None
        decay_type = parameter['decay_type'].lower() if 'decay_type' in parameter.keys() else None
        decay_rate = parameter['decay_rate'] if 'decay_rate' in parameter.keys() else None
        decay_steps = parameter['decay_steps'] if 'decay_steps' in parameter.keys() else train_iterations
        verbose = parameter['verbose'] if 'verbose' in parameter.keys() else verbose

        if decay_type is not None and decay_type == 'trainable':
            # If parameter is not supposed to be annealed, but to be trained, return None
            return None

        return Scheduler(
            initial_value=initial_value,
            decay_type=decay_type,
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            device=device,
            min_value=min_value,
            value_name=parameter_name,
            verbose=verbose
        )

    else:
        raise NotImplementedError("parameter must be float or dict")


def get_non_linearity(nonlinearity):
    # Takes a string-specification of non-linearity-type and turns it into a functional non-linearity
    if nonlinearity.lower() == 'relu':
        return F.relu
    elif nonlinearity.lower() == 'sigmoid':
        return F.sigmoid
    elif nonlinearity.lower() == 'tanh':
        return F.tanh
    else:
        raise NotImplementedError("Only relu or sigmoid or tanh admissible as non-linearities")


def get_optimizer(learning_rate: float or dict, model_parameters):
    # This method returns an Adam optimizer according to some specification
    if isinstance(learning_rate, float):
        # Simple optimizer with constant learning rate for neural net
        return torch.optim.Adam(params=model_parameters, lr=learning_rate)

    elif isinstance(learning_rate, dict):
        # Create optimizer plus a learning rate scheduler associated with optimizer
        return torch.optim.Adam(params=model_parameters, lr=learning_rate['initial_value'])

    else:
        raise NotImplementedError("learning_rate must be (constant) float or dict.")


def get_lr_scheduler(learning_rate: float or dict, optimizer, train_iterations: int,
                     value_name: str = 'Learning Rate to be decreased'):
    # For scheduling learning rates, readily available PyTorch schedulers are used. This function takes some
    # specification of how the scheduler is supposed to work and returns a correspondingly set up scheduler

    if isinstance(learning_rate, float):
        # Simple optimizer with constant learning rate for neural net, thus no scheduler needed
        return None

    elif isinstance(learning_rate, dict):
        # Whether learning rate scheduler shall print feedback or not
        verbose = learning_rate['verbose'] if 'verbose' in learning_rate.keys() else False

        decay_type = learning_rate['decay_type'].lower() if 'decay_type' in learning_rate.keys() else None
        initial_lr = learning_rate['initial_value'] if 'initial_value' in learning_rate.keys() else None
        decay_steps = learning_rate['decay_steps'] if 'decay_steps' in learning_rate.keys() else train_iterations
        decay_rate = learning_rate['decay_rate'] if 'decay_rate' in learning_rate.keys() else None
        min_value = learning_rate['min_value'] if 'min_value' in learning_rate.keys() else None

        # Choose which scheduler to return - Dependent on the requested decay type
        if decay_steps or decay_rate or min_value is not None:
            # If settings are provided that could not be incorporated into PyTorch's own LR-schedulers, use a custom one
            return CustomLRScheduler(optimizer=optimizer, initial_value=initial_lr,
                                     decay_type=decay_type, decay_steps=decay_steps,
                                     decay_rate=decay_rate, min_value=min_value, value_name=value_name, verbose=verbose)

        elif decay_type == 'linear':
            lambda_lr = lambda epoch: (train_iterations - epoch) / train_iterations
            return LambdaLR(optimizer, lr_lambda=lambda_lr, verbose=verbose)

        elif decay_type == 'exponential':
            decay_factor = learning_rate['decay_factor'] if 'decay_factor' in learning_rate.keys() else 0.9
            return ExponentialLR(optimizer, gamma=decay_factor, verbose=verbose)

        elif decay_type == 'constant':
            raise NotImplementedError("Provide a float value as learning rate parameter when intending to keep learning rate constant.")

        else:
            raise NotImplementedError("Learning rate decay may only be linear or exponential.")

    else:
        raise NotImplementedError("learning_rate_pol must be (constant) float or dict.")



def is_provided(param):
    # Returns whether a parameter-specification is provided or not
    if param is not None:
        return True
    return False


def is_trainable(param):
    # Returns true if a provided parameter is supposed to be trainable, otherwise false
    if isinstance(param, dict) and 'decay_type' in param.keys() and param['decay_type'] == 'trainable':
        return True
    return False


def nan_error(tensor):
    # Checks if NAN-error is present in a tensor
    return torch.isnan(tensor).any()


def print_nan_error_loss(loss, L_CLIP, L_V, action, log_prob, log_prob_old, state, state_val, L_ENTROPY=None):
    # Print some special error message when NAN-error is detected
    print(
        "Loss happened to be nan. This indicates loss terms going out of bounds. Please check your hyperparameters once again.")
    print('Values were as follows:\n')
    print('Loss:', loss, '\nL_CLIP:', L_CLIP, '\nL_V:', L_V)
    print('L_ENTROPY:', L_ENTROPY if L_ENTROPY else 'N/A')
    print('action:', action, '\nlog_prob:', log_prob, '\nlog_prob_old:', log_prob_old)
    print('state:', state, '\nstate_val:', state_val)
