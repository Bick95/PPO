import gym
import random
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
                 trajectory_length: int = 1000,
                 discount_factor: float = 0.98,
                 batch_size: int = 32,
                 epsilon: float = 0.2
                 # TODO: clean up args... with proper imports!
                 ):

        # Save variables
        self.epochs = epochs
        self.parallel_agents = parallel_agents
        self.T = trajectory_length
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.losses = []

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
                                                  self.val_net.base.parameters()],
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

                    # Get log-prob of chosen action (= log \pi_{\theta_{old}}(a_t|s_t) )
                    log_prob = self.policy.log_prob(action)

                    # Transform to tensor data
                    state = torch.tensor(state)
                    reward = torch.tensor(reward)
                    next_state = torch.tensor(next_state)
                    terminal_state = torch.tensor(terminal_state)
                    log_prob = torch.tensor(log_prob)

                    # Collect observable data
                    observation = (state, action, reward, next_state, terminal_state, log_prob)
                    obs_temp.append(observation)

                    # Add number of added train steps to counter
                    train_steps += self.parallel_agents


                    # Prepare for next iteration
                    if terminal_state.any() or train_steps >= self.T:
                        # Reset env for case where train_steps < max_trajectory_length T
                        state = self.env.reset()

                        # -- Add temporarily stored observations to list of all freshly collected training data, stored in 'observations' --
                        # Compute state value of final observed state
                        last_state = obs_temp[-1][3]
                        target_state_val = self.val_net(last_state)  # V(s_T)

                        # Compute the target state value and advantage estimate for each state in agent's trajectory
                        # (batch-wise for all parallel agents in parallel)
                        for t in range(len(obs_temp)-1, -1, -1):
                            # Compute target state value:
                            # V^{target}_t = r_t + \gamma * r_{t+1} + ... + \gamma^{n-1} * r_{t+n-1} + \gamma^n * V(s_{t+n}), where t+n=T
                            target_state_val = obs_temp[t][2] + self.gamma * target_state_val

                            # Compute advantage estimate
                            state_val = self.val_net(obs_temp[t][0])
                            advantage = target_state_val - state_val

                            # Augment previously observed observation tuples
                            extra = (target_state_val, advantage)
                            augmented_obs = obs_temp[t] + extra

                            # Add all parallel agents' individual observations to overall observations list
                            for i in range(self.parallel_agents):
                                # Create i^th agent's private observation tuple for time step t in its current trajectory
                                # element \in {state, action, reward, next_state, terminal_state, log_prob, target_val, advantage}
                                private_tuple = tuple([element[i] for element in augmented_obs])
                                observations.append(private_tuple)

                        # Empty temporary list of observations after they have been added to more persistent list of freshly collected train data
                        obs_temp = []

                    else:
                        # Trajectory continues from time step t to t+1
                        state = next_state

            # Perform weight updates for multiple epochs on freshly collected training data stored in 'observations'
            for epoch in range(self.epochs):
                # Shuffle data
                random.shuffle(observations)  # Shuffle in place!

                # Perform weight update on each minibatch contained in shuffled observations
                for i in range(0, len(observations), self.batch_size):
                    # Reset all grads
                    self.optimizer.zero_grad()

                    # Sample minibatch
                    minibatch = observations[i: i+self.batch_size]

                    # Get all states, actions, log_probs, target_values, and advantage_estimates from minibatch
                    state, action, _, _, _, log_prob_old, target_state_val, advantage = zip(*minibatch)

                    # Compute log_prob of action(s)
                    _ = self.policy(state)
                    log_prob = self.policy.log_prob(action)

                    # Compute current state value estimates
                    state_val = self.val_net(state)

                    # Evaluate loss function:
                    # L^{CLIP}
                    L_CLIP = self.L_CLIP(log_prob, log_prob_old, advantage)

                    # L^{H=Entropy}
                    L_ENTROPY = self.L_ENTROPY()

                    # L^{V}
                    L_V = self.L_VF(state_val, target_state_val)

                    # L^{CLIP + H + V} = L^{CLIP} + L^{ENTROPY} + L^{V}
                    loss = - L_CLIP - L_ENTROPY + L_V

                    # Backprop loss
                    loss.backward()

                    # Perform weight update
                    self.optimizer.step()

                    # Document loss
                    self.losses.append(loss.numpy())


    def ratio(self, numerator, denominator):
        return numerator / denominator


    def L_CLIP(self, log_prob, log_prob_old, advantage):
        # Computes PPO's main objective L^{CLIP}

        prob_ratio = self.ratio(numerator=log_prob, denominator=log_prob_old)

        unclipped = prob_ratio * advantage
        clipped = torch.clip_(prob_ratio, min=1.-self.epsilon, max=1.+self.epsilon) * advantage

        return torch.mean(torch.min(unclipped, clipped))


    def L_ENTROPY(self):
        # Computes entropy bonus for policy net
        return torch.mean(self.policy.entropy())


    def L_VF(self, state_val: torch.tensor, target_state_val: torch.tensor):
        # Loss function for state-value network. Quadratic loss between predicted and target state value
        return torch.mean((state_val - target_state_val) ** 2)
