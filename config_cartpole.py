{
    "env": "CartPole-v0",
    'input_net_type': 'MLP',  # vs 'CNN'
    'grayscale_transform': False,
    'markov_length': 1,
    "total_num_state_transitions": 100000,
    "param_sharing": True,

    "epochs": 5,
    "parallel_agents": 5,
    "learning_rate_pol": 0.001,
    "learning_rate_val": 0.001,
    "trajectory_length": 200,
    "discount_factor": 0.99,
    "batch_size": 32,
    
    "clipping_parameter": 0.05,
    "entropy_contrib_factor": 0.01,
    "vf_contrib_factor": 1.,
    
    'show_final_demo': True,
    'deterministic_eval': True,
    
    'intermediate_eval_steps': 100,
    
    #'standard_dev': torch.ones,
    'network_structure': [
		# Nr. of nodes for fully connected layers
		32,
		64,
		32,
	]			
    #'nonlinearity': F.relu
}