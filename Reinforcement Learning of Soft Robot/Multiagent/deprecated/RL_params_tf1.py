# Import standard libraries
import time
from datetime import date
import os
import sys
import pathlib
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../..')

# Loading info from environment needed in this list
from env_params import maxNumSteps

# Defining Experiment Parameters
experimentNum = "Enter Experiment Name"

training_timesteps = 4_300_000

divisor = 1 # Number of times to stop mid-training to observe results

# Training Parameters
neural_network =[256,256]                       # Doubling the size of ANN
policyName = 'CustomPolicy_'+str(experimentNum)# Name of policy. This can be ignored.
gamma = 0.95                                   # Discount factor
n_steps = 1_000                                # Number of steps to run in each environment per update. Batchsize = n_steps*n_env
ent_coef = 0.001                                # Entropy coefficient
learning_rate = 0.001                        # Learning Rate, can be a funcion
vf_coef = .1                                   # Value Function Coefficient in Loss Function
max_grad_norm = 0.5                            # Clipping factor for gradients. Should prevent exploding gradients
lam = 0.95                                     # Factor for bias vs variance for GAE
batch_size = 100                             # Number of minibatches at each update.
noptepochs = 7                                 # Number of epochs each update
cliprange = 0.2                                # Cliprange for PPO
seed = 12345                                   # Seed for neural network initialization
nEnvs = 4                                      # Number of parallel environments


# Parameters for callback
num_ep_save = 2 # Calculate the mean reward for this number of episodes and save that model
check_freq = 50000 # After how many timesteps do we check the frequency


# Post training parameters
test=True                       # Whether testing should occur post training
num_tests=3                     # Number of tests with thispolicy to run
render=True                     # Whether to visualize the training
time_per_test = maxNumSteps    # Number of timesteps to run each results episode for.


"""
____________________________________________________________
Users should not have to change anything on below this comment
____________________________________________________________
"""

# Ensuring we are in the proper directory
experimentName = 'Experiment_{}'.format(experimentNum)

policy_kwargs = dict(
    net_arch = [dict(
        pi=neural_network,
        vf=neural_network
    )]
)
