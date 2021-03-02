import torch
import gym
from policy import Policy
from value_net import ValueNet


class ProximalPolicyOptimization:

    def __init__(self,
                 env: gym.Env or str,
                 epochs: int = 1000000,
                 parallel_agents: int = 8,
                 param_sharing: bool = True
                 ):

        # Save variables
        self.epochs = epochs
        self.parallel_agents = parallel_agents

        # Create Gym env if not provided as such
        if isinstance(env, str):
            env = gym.make(env)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Create policy net
        policy = Policy(self.action_space, self.observation_space, 'MLP')

        # Create value net
        if param_sharing:
            val_net = ValueNet(shared_layers=policy.get_non_output_layers())

        # Create optimizers
        

        # Vectorize env using info about nr of parallel agents



        pass
