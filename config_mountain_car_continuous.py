{
    "env": "MountainCarContinuous-v0",
    'input_net_type': 'MLP',
    'grayscale_transform': False,
    'markov_length': 10,
    'dilation': 2,
    "total_num_state_transitions": 100000,
    "param_sharing": True,
    'nonlinearity': 'relu',

    'standard_dev': 0.7,  # only used for continuous action spaces

    "epochs": 6,
    "batch_size": 32,

    "parallel_agents": 12,
    "trajectory_length": 4000,

    "learning_rate_pol": {
		'decay_type': 'exponential',
        'decay_rate': 0.95,
		'initial_value': 0.00002,
        'min_value': 0.000017,
		'verbose': True,
	},
    "learning_rate_val": {
		'decay_type': 'exponential',
        'decay_rate': 0.85,
		'initial_value': 0.00002,
        'min_value': 0.000017,
		'verbose': True,
	},
    
    "clipping_parameter": {
		'decay_type': 'exponential',
        'decay_rate': 0.96,
		'initial_value': .15,
		'min_value': .02,
        'verbose': True
	},

    "discount_factor": 0.99,
    "entropy_contrib_factor": 0.05,
    "vf_contrib_factor": 1.,
    
    'show_final_demo': True,
    'frame_duration': 0.001,  # Indicates how long rendered frames are shown to yield a nice and smooth visualization
    'deterministic_eval': True,
    
    'intermediate_eval_steps': 1000,
    'time_steps_extensive_eval': 10000,
    

    'network_structure': [
		# Nr. of nodes for fully connected layers
        64,
		128,
		32,
	],
}