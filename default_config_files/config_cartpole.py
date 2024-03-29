{
    "env": "CartPole-v0",
    'input_net_type': 'MLP',
    'grayscale_transform': False,
    'markov_length': 4,
    'dilation': 1,
    "total_num_state_transitions": 200000,
    "param_sharing": True,

    "epochs": 5,
    "parallel_agents": 5,
    "trajectory_length": 600,
    "discount_factor": 0.99,
    "batch_size": 32,

    "learning_rate_pol": 0.0002,
    "learning_rate_val": 0.0002,

    "clipping_parameter": {
        'decay_type': 'linear',
        'initial_value': 0.15,
        'min_value': 0.005,
        'verbose': True,
    },

    "entropy_contrib_factor": 0.01,
    "vf_contrib_factor": 1.,
    
    'show_final_demo': True,
    'frame_duration': 0.08,  # Indicates how long rendered frames are shown to yield a nice and smooth visualization
    'max_render_time_steps': 300,  # For how many time steps to render the environment during finl demo

    'deterministic_eval': True,
    'stochastic_eval': True,
    'intermediate_eval_steps': 100,
    'time_steps_extensive_eval': 10000,

    'network_structure': [
		# Nr. of nodes for fully connected layers
		32,
		64,
		32,
	]			
    #'nonlinearity': 'relu'
}