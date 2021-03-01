import gym
import torch
import numpy as np




CONTINUOUS_OUT  = 0
DISCRETE_OUT    = 1

class Policy:

    def __init__(self,
                 action_space: gym.Space,
                 observation_space: gym.Space
                 ):

        self.action_space = action_space
        self.observation_space = observation_space

        as_size = np.array(self.action_space.sample().shape).size
        os_size = np.array(self.observation_space.sample().shape).size

        if os_size == 1:
            # Heuristic: assume MLP to apply in this case
            #self.nature_input_layer = None
            pass
            # TODO: immediately create input module here
        else:
            # Heuristic: assume CNN to apply here
            #self.nature_input_layer = None
            pass
            # TODO: immediately create input module here

        # TODO: create output modules here






