import gym
import torch.optim
from policy import Policy
from value_net import ValueNet


class ProximalPolicyOptimization:

    def __init__(self,
                 env: gym.Env or str,
                 epochs: int = 1000000,
                 parallel_agents: int = 8,
                 param_sharing: bool = True,
                 learning_rate: float = 0.0001,
                 num_envs: int = 8,
                 trajectory_length: int = 1000
                 ):

        # Save variables
        self.epochs = epochs
        self.parallel_agents = parallel_agents
        self.T = trajectory_length

        # Create Gym env if not provided as such
        if isinstance(env, str):
            env = gym.make(env)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Create policy net
        self.policy = Policy(self.action_space, self.observation_space, 'MLP')

        # Create value net (either sharing parameters with policy net or not)
        if param_sharing:
            self.val_net = ValueNet(shared_layers=self.policy.get_non_output_layers())
        else:
            self.val_net = ValueNet(self.observation_space, 'CNN')

        # Create optimizers
        self.optimizer = torch.optim.Adam(params=[self.policy.base.parameters(),
                                                  self.val_net.base.parameters],
                                          lr=learning_rate)


        # Vectorize env using info about nr of parallel agents
        env_name = env.unwrapped.spec.id
        self.env = gym.vector.make(id=env_name, num_envs=self.parallel_agents, asynchronous=False)


    def learn(self):

        for iteration in range(self.iterations):  # TODO
            # Init data collection and storage
            train_steps = 0
            observations = []
            obs_temp = []

            # Init envs
            state = self.env.reset()

            # Collect training data
            with torch.no_grad():
                while train_steps < self.T:
                    # Predict action
                    action = self.policy(state)

                    # Perform action in env
                    next_state, reward, terminal_state, _ = self.env.step(action)

                    # Transform to tensor data
                    state = torch.tensor(state)
                    reward = torch.tensor(reward)
                    next_state = torch.tensor(next_state)
                    terminal_state = torch.tensor(terminal_state)

                    # Collect observable data
                    observation = (state, action, reward, next_state, terminal_state)
                    obs_temp.append(observation)

                    # Prepare next iteration: reset? state = next_state, ...
                    if terminal_state.any() or train_steps >= self.T:
                        # Reset env for case where train_steps < max_trajectory_length T
                        state = self.env.reset()

                        # Add temp observations to current iteration's observations
                        last_state = obs_temp[-1][4]
                        target_state_val = self.val_net(last_state)
                        for t in range(len(obs_temp)-1, -1, -1):
                            target_state_val = target_state_val + obs_temp[t][2]
                            state_val = self.val_net(obs_temp[t][0])
                            advantage = target_state_val - state_val

                            # Augment previously observed observation tuples
                            extra = (target_state_val, advantage)
                            augmented_obs = obs_temp[t] + extra

                            # Add all parallel agents' individual observations to overall observations
                            for i in range(self.parallel_agents):
                                tpl = tuple([x for x in augmented_obs])
                                # TODO: add tuple FOR EACH agent AND each kind (like state vs next state vs rewards...)
                                observations.append()


                    else:
                        state = next_state

                    train_steps += self.parallel_agents

            # Perform weight updates for multiple epochs on freshly collected training data
            for epoch in range(self.epochs):
                pass
