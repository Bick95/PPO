{
    "env": "Breakout-v0",
    'input_net_type': 'CNN', # vs 'MLP'
    'grayscale_transform': True,
    'markov_length': 4,
    "total_num_state_transitions": 100000,
    "param_sharing": True,

    "epochs": 5,
    "parallel_agents": 10,
    "learning_rate_pol": 0.0001,
    "learning_rate_val": 0.0001,
    "trajectory_length": 1000,
    "discount_factor": 0.99,
    "batch_size": 32,
    
    "clipping_constant": 0.2,
    "entropy_contrib_factor": 0.15,
    "vf_contrib_factor": .9,
    
    'show_final_demo': True,
    'deterministic_eval': True,
	'time_steps_extensive_eval': 1000,
    
    'intermediate_eval_steps': 1000,
    
    #'standard_dev': torch.ones,
    'network_structure': [
        # Dicts for conv layers
		{
			'out_channels': 32,  # 16 output channels = 16 filters
			'kernel_size': 4,  # 8x8 kernel/filter size
			'stride': 1, 
			'padding': 0,
		},
		{
			'out_channels': 32,
			'kernel_size': (
				4, # vertical kernel-size
				4  # horizontal kernel-size
			),
			'stride': (
				2, # vertical stride
				2  # horizontal stride
			),
			'padding': (
				0, # vertical padding
				0  # horizontal padding
			),
		},
		{
			'out_channels': 16,  # 16 output channels = 16 filters
			'kernel_size': 8,  # 8x8 kernel/filter size
			'stride': 1, 
			'padding': 0,
		},
		{
			'out_channels': 16,  # 16 output channels = 16 filters
			'kernel_size': 8,  # 8x8 kernel/filter size
			'stride': 2, 
			'padding': 0,
		},
		# Nr. of nodes for fully connected layers
		128,
		64,
		64,
	]			
    #'nonlinearity': F.relu
}