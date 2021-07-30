import torch
import numpy as np


class Scheduler:

    def __init__(self,
                 initial_value: float or int,
                 decay_type: str,
                 device: torch.device,
                 decay_rate: float = None,
                 decay_steps: int = None,
                 min_value: float or int = None,
                 value_name: str = None,
                 verbose: bool = False,
                 ):

        # Keeps track of and decays a given value in a given way

        self.decay_type = decay_type.lower() if decay_type is not None else 'linear'
        self.decay_steps = float(decay_steps) if decay_steps is not None else None
        self.value_name = value_name if value_name is not None else 'value to be scheduled'
        self.verbose = verbose
        self.device = device

        if initial_value is None:
            raise Warning("initial_value not provided for " + self.value_name)

        self.value = float(initial_value)

        # Determine minimum value to which initial value may be decayed
        if min_value is None:
            self.min_value = 0.
        elif isinstance(min_value, int):
            self.min_value = float(min_value)
        else:
            self.min_value = min_value

        # Determine decay rate
        if decay_rate is not None:
            self.decay_rate = float(decay_rate)

        # If no decay rate is provided, the proper default for it depends on the type of decay
        elif self.decay_type == 'exponential':
            self.decay_rate = 0.9

        elif self.decay_type == 'linear' and self.decay_steps is not None:
            # We know over which time span to linearly anneal the initial value to the min value
            # Thus, compute the decay rate for linear decay
            self.decay_rate = (self.value - self.min_value) / self.decay_steps

        elif self.decay_type != 'constant':
            self.decay_rate = 0.05
            raise print("Decay rate for " + self.value_name + " not provided! Chosen default: " + str(self.decay_rate))


        self._step = None  # Internal step function decrementing self.value each time that step()-method is invoked
        self.init_step_function()


    def init_step_function(self):
        # Assign function that will decay the value to be decayed whenever step() method is called

        if self.decay_type == 'constant':
            # Handle cases where value is not supposed to decay
            self._step = lambda: self.value

        elif self.decay_type == 'linear':
            # Handle cases where decay type is 'linear'

            # After we know the decay rate (which is either provided or has been computed above), next compute function
            # decaying the value each scheduler-step
            self._step = lambda: float(np.nanmax([self.min_value, self.value - self.decay_rate]))

        elif self.decay_type == 'exponential':
            # Handle cases where decay type is 'exponential'
            self._step = lambda: float(np.nanmax([self.min_value, self.value * self.decay_rate]))

        else:
            raise NotImplementedError('Scheduler can only handle linear or exponential decay.')


    def step(self):
        # Decrease the scheduled value by one quantity

        if self.verbose:
            print("Going to decrease", self.value_name, self.decay_type + "ly by factor {:.5f}".format(self.decay_rate),
                  "to {:.5f}".format(self._step()) + ".")

        self.value = self._step()


    def get_value(self, parallel_agents: int = 1):
        return torch.tensor([[self.value]] * parallel_agents, requires_grad=False, device=self.device)
