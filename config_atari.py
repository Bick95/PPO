{
    "env": "Breakout-v0",
    'input_net_type': 'CNN',  # vs 'MLP'
    'grayscale_transform': True,
	'resize_visual_inputs': (84, 84),
    'markov_length': 4,
    "total_num_state_transitions": 120000,
    "param_sharing": True,
	'nonlinearity': 'relu',

    "epochs": 4,
    "parallel_agents": 8,
    "trajectory_length": 500,
    "discount_factor": 0.99,
    "batch_size": 32,

    "learning_rate_pol": {
		'decay_type': 'linear',
		'initial_value': 0.00015,
		'verbose': True,

	},
    "learning_rate_val": {
		#'decay_type': 'exponential',
		#'decay_factor': 0.9,  # only used for exponential decay
		#'initial_value': 0.03,
		'decay_type': 'linear',
		'initial_value': 0.00015,
		'verbose': True,

	},
    "clipping_parameter": {
		'decay_type': 'linear',
		'initial_value': .1,
		'min_value': 0.,
	},
    "entropy_contrib_factor": 0.01,
    "vf_contrib_factor": 1.,
    
    'show_final_demo': True,
    'deterministic_eval': True,
	'time_steps_extensive_eval': 1000,
    'intermediate_eval_steps': 1000,
    
    #'standard_dev': torch.ones,  # only used for continuous action spaces
    'network_structure': [
        # Dicts for conv layers
		{
			'out_channels': 64,  # 32 output channels = 32 filters
			'kernel_size': 8,    # 8x8 kernel/filter size
			'stride': 2,
			'padding': 'automatically preserve input size',
		},
		{
			'out_channels': 64,
			'kernel_size': (
				4,  # vertical kernel-size
				4   # horizontal kernel-size
			),
			'stride': (
				2,  # vertical stride
				2   # horizontal stride
			),
			'padding': (
				43,  # vertical padding
				43   # horizontal padding
			),
		},
		{
			'out_channels': 64,  # 64 output channels = 64 filters
			'kernel_size': 3,    # 8x8 kernel/filter size
			'stride': 1,
			'padding': 0,
		},
		{
			'out_channels': 16,  # 16 output channels = 16 filters
			'kernel_size': 8,    # 8x8 kernel/filter size
			'stride': 2, 
			'padding': 0,
		},
		# Nr. of nodes for fully connected layers
		256,
		128,
	],
}