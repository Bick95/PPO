{
    "env": "MountainCarContinuous-v0",
    'input_net_type': 'MLP',  # vs 'CNN'
    'grayscale_transform': False,
    'markov_length': 2,
    "total_num_state_transitions": 10000000,
    "param_sharing": True,
    'nonlinearity': 'relu',

    'standard_dev': {
		#'decay_type': 'linear',
        'decay_type': 'trainable',
        #'decay_steps': 50,
        #'decay_rate': 0.01,
		#'initial': 1.,
		#'min': .05,
        'verbose': True
	},  # only used for continuous action spaces

    "epochs": 5,
    "batch_size": 32,

    "parallel_agents": 10,
    "trajectory_length": 4000,

    "learning_rate_pol": {
		'decay_type': 'exponential',
        'decay_rate': 0.8,
		'initial': 0.00001,
        'min': 0.000001,
		'verbose': True,
        #'decay_steps': 50,
	},
    "learning_rate_val": {
		'decay_type': 'exponential',
        'decay_rate': 0.8,
		'initial': 0.00001,
        'min': 0.000001,
		'verbose': True,
        #'decay_steps': 50,
	},
    
    "clipping_parameter": {
		'decay_type': 'exponential',
        #'decay_steps': 50,
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
        32,
		64,
		32,
	],
}