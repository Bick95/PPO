{
    "env": "CartPole-v1",
    "total_num_state_transitions": 5000000,
    "epochs": 10,
    "parallel_agents": 10,
    "param_sharing": True,
    "learning_rate_pol": 0.0001,
    "learning_rate_val": 0.0001,
    "trajectory_length": 1000,
    "discount_factor": 0.99,
    "batch_size": 32,
    "clipping_constant": 0.2,
    "entropy_contrib_factor": 0.15,
    "vf_contrib_factor": .9,
    'input_net_type': 'MLP',
    'show_final_demo': True,
    'intermediate_eval_steps': 1000,
    #'standard_dev': torch.ones,
    'hidden_nodes_pol': [50, 50, 50],
    'hidden_nodes_vf': [50, 50, 50],
    #'nonlinearity': F.relu
}