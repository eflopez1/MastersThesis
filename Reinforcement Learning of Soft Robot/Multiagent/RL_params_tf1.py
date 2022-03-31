"""
This file contains all parameters needed for running reinforcememt learning training on the grabbing environmnet
"""

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

# Import custom functions
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv

# Loading info from environment needed in this list
from env_params import maxNumSteps, numBots

# Defining Experiment Parameters
experimentNum = "10d_entropy_framestacking_cont"
        
# Depending on how many bots there are in the system, 
# the number of training time steps will vary

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


class CheckpointCallback2(BaseCallback):
    """
    Callback for saving a model every `save_freq` steps
    A slight modification from the callback above

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix='rl_model', verbose=0):
        super(CheckpointCallback2, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.last_time_trigger = 0
        self.last_save_name = None

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        
        if (self.num_timesteps - self.last_time_trigger) >= self.save_freq:
            path = os.path.join(self.save_path, '{}_{}_steps'.format(self.name_prefix, self.num_timesteps))
            if self.last_save_name is None:
                self.last_save_name = path
            else:
                try:
                    os.remove(self.last_save_name +'.zip')
                except:
                    pass
                self.last_save_name = path

            self.model.save(path)
            self.last_time_trigger = self.num_timesteps
            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(path))
        return True