import torch
import gym


class ProximalPolicyOptimization:

    def __init__(self,
                 env: gym.Env or str,
                 epochs: int = 1000000,
                 parallel_agents: int = 8
                 ):

        # Save variables
        self.epochs = epochs
        self.parallel_agents = parallel_agents

        # Create Gym env
        if isinstance(env, str):
            env = gym.make(env)

        # Sample observation and obtain observation space info
        osp = env.observation_space

        # Sample action
        asp = env.action_space

        # Create policy net

        # Create value net

        # Create optimizers

        # Vectorize env using info about nr of parallel agents



        pass
