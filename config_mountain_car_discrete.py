{
    "env": "MountainCar-v0",
    'input_net_type': 'MLP',
    'grayscale_transform': False,
    'markov_length': 10,
    'dilation': 2,
    "total_num_state_transitions": 3000000,
    "param_sharing": True,
    'nonlinearity': 'relu',

    "epochs": 6,
    "batch_size": 32,

    "parallel_agents": 12,
    "trajectory_length": 4000,

    "learning_rate_pol": {
		'decay_type': 'exponential',
        'decay_rate': 0.9,
		'initial': 0.00002,
        'min': 0.0000001,
		'verbose': True,
        #'decay_steps': 50,
	},
    "learning_rate_val": {
		'decay_type': 'exponential',
        'decay_rate': 0.8,
		'initial': 0.00002,
        'min': 0.0000001,
		'verbose': True,
	},
    
    "clipping_parameter": {
		'decay_type': 'exponential',
        'decay_rate': 0.95,
		'initial': .15,
		'min': .02,
        'verbose': True
	},

    "discount_factor": 0.99,
    "entropy_contrib_factor": 0.05,
    "vf_contrib_factor": 1.,
    
    'show_final_demo': True,
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