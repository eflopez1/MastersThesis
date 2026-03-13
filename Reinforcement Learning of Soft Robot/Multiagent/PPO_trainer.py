'''
Script to execute to generate training for model
'''

import yaml

from utils import load_yaml, eval_yaml
from Environment import parallel_env

env_params_path = 'env_params.yaml'
rl_params_path = 'rl_params.yaml'

# Wrapped here to allow for graceful termination
if __name__ == '__main__':

    # Load environment parameters
    envParams = load_yaml(env_params_path)
    envParams = eval_yaml(envParams)
    
    # Create the envuronment 
    training_env = parallel_env(envParams)