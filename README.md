# Proximal Policy Optimization (PPO)
## A Comprehensive Implementation of Proximal Policy Optimization

### Introduction

This repository contains the reference implementation of the popular 
[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO) algorithm 
and accompanies my Final Master's Project report on the topic: 

<div align="center">
    "<b>
        Towards Delivering a Coherent Self-Contained Explanation of Proximal Policy Optimization
    </b>"
</div>

While the aforementioned report aims at 
*theoretically* explaining the PPO algorithm in a detailed, complete, and comprehensible way, 
the reference implementation provided in this repository aims at *practically* illustrating the PPO algorithm in a 
comprehensible way. 

For more information on the background of this work, the reader is referred to the 
corresponding report (link to be added later). 

### Setup
For an easy setup, it is recommended to install the requirements in a virtual environment. 

After having set up and activated some virtual environment, proceed as follows: 
1. Open the **terminal**
2. **Navigate** to the main directory of this repository, i.e. the directory containing the 
`requirements.txt` file
3. **Install** the requirements using pip as follows: `pip install -r requirements.txt`

This concludes the setup. 

### Training an Agent
Training (as well as replaying the behavior of previously trained agents) is initiated via the 
`main.py` file. 
Before training an agent, one has to set up a corresponding configuration file defining 
the conditions under which training and testing are performed. For convenience, some 
example configuration files are provided already. They can be found 
[here](https://github.com/Bick95/PPO/tree/main/default_config_files).  
Once, a configuration file is created or selected from the examples, training can be 
initiated as follows. 

Suppose, we want to train an agent on the OpenAI Gym'y CartPole-v0-Environment one of the 
provided default [configuration](https://github.com/Bick95/PPO/blob/main/default_config_files/config_cartpole.py) 
files. To train an agent using the aforementioned configuration, proceed as follows:
1. Open the **terminal**
2. **Navigate** to the main directory of this repository, i.e. the directory containing the 
`main.py` file
3. Run `python main.py -c='./default_config_files/config_cartpole.py'`. The `c`-flag 
indicates "**c**onfig". 

Upon completion of the training procedure, a new directory will be created inside the 
directory where also the `main.py` file is located. That directory will be called 
`train_results` and contain sub-directories. Each subdirectory inside `train_results` will 
contain the saved data associated with a performed test run. Such a sub-directory, having 
some name like `2021_06_27__22_58_00__tsrOOxrwle`, will contain the following elements:
1. `config.json`: a copy of the config file used to train and test the agent
2. `policy_model.pt`: Archive containing all the saved parameters of the policy net. Can be 
used to replay some agent later (only saved if requested by user via config file)
3. `val_net_model.pt`: Archive containing trainable parameters of state value network 
(only saved if requested by user via config file)
4. `train_stats.json`: File containing the training- and evaluation statistics

### Replaying a trained agent
...