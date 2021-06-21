{
    "env": "CartPole-v0",
    'input_net_type': 'MLP', # vs 'CNN'
    'grayscale_transform': False,
    'markov_length': 4,
    "total_num_state_transitions": 1000000,
    "param_sharing": True,

    "epochs": 1,
    "parallel_agents": 10,
    "learning_rate_pol": 0.0001,
    "learning_rate_val": 0.0001,
    "trajectory_length": 500,
    "discount_factor": 0.99,
    "batch_size": 32,
    
    "clipping_constant": 0.2,
    "entropy_contrib_factor": 0.15,
    "vf_contrib_factor": .9,
    
    'show_final_demo': True,
    'deterministic_eval': False,
    
    'intermediate_eval_steps': 100,
    
    #'standard_dev': torch.ones,
    'network_structure': [
		# Nr. of nodes for fully connected layers
		64,
		64,
		64,
	]			
    #'nonlinearity': F.relu
}