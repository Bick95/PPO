{
    "env": "Breakout-v0",
    'input_net_type': 'CNN', # vs 'CNN'
    'grayscale_transform': True,
    'markov_length': 5,
    "total_num_state_transitions": 1000,
    "epochs": 1,
    "parallel_agents": 2,
    "param_sharing": True,
    "learning_rate_pol": 0.0001,
    "learning_rate_val": 0.0001,
    "trajectory_length": 500,
    "discount_factor": 0.99,
    "batch_size": 32,
    "clipping_constant": 0.2,
    "entropy_contrib_factor": 0.15,
    "vf_contrib_factor": .9,
    'show_final_demo': False,
    'intermediate_eval_steps': 100,
    #'standard_dev': torch.ones,
    'hidden_nodes_pol': [50, 10, 50],
    'hidden_nodes_vf': [20, 50, 50],
    'network_structure': [
        # Dicts for conv layers
		{
			'out_channels': 16,  # 16 output channels = 16 filters
			'kernel_size': 8,  # 8x8 kernel/filter size
			'stride': 4, 
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
		# Nr. of nodes for fully connected layers
		256
	]			
    #'nonlinearity': F.relu
}