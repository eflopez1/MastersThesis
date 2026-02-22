# %% Main cell
"""
This script can be used to run an environment WITHOUT using any learned agent.

Of course, the user can also simply import a model here and call that agent
for actions to be taken.
"""

# Imports
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import eval_yaml, createVideo
from Environment import parallel_env

# Importing Parameters and all needed libraries
# DEPRECATED! Use YAML file instead!
# from env_params import *  

# Load envirinment parameters
config_file = 'env_params.yaml'
with open(config_file, 'r') as infile:
    envParams = yaml.safe_load(infile)

# Resolve any math equations used in the YAML file
envParams = eval_yaml(envParams)

# Create the environment
env = parallel_env(envParams)
_  = env.reset()

# Defining actions for simple L-Turn
actions=dict()
for agent in env.agents:
    f = 1
    actions[agent] = [f,0]

# Slowing down sim so it is easier to see
from time import sleep
slow = False # Set to True if you need the video to slow down.
             # Do not use if you are recording the video!

for _ in range(1):
    plt.close('all')
    obs = env.reset()

    for i in tqdm(range(envParams['maxNumSteps']//7)):
        if slow: sleep(.01)
        if envParams['render']:
            env.render(None)
        if i == 50:
            for agent in env.agents:
                actions[agent] = [0,0]
        obs, _, done, _ = env.step(actions)
        
        # if any(list(done.values())): 
        #     print('Done')
        #     break

    print('End of episode')
    
    if envParams['dataCollect']:
        env.dataExport()
    if envParams['saveVideo']: 
        createVideo(env.saveFolder, 
                    env.videoFolder, 
                    envParams['experimentName'], 
                    (envParams['width'], envParams['height']))
    env.close()
    plt.close('all')
