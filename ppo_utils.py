import torch
import numpy as np
from PIL import Image


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