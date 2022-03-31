# %% Main cell
"""
This script can be used to run an environment WITHOUT using any learned agent.

Of course, the user can also simply import a model here and call that agent
for actions to be taken.
"""

# Importing Parameters and all needed libraries
from env_params import *
import time

# Ensure working directory is set proper          
filepath = os.path.dirname(os.path.realpath(__file__))

# Create the environment
env = pymunkEnv(**envParams)

if env.dataCollect:
    env.report_all_data = True

# Standard environment
action=[]
for num in range(numBots*2):
    if num%2==0: action.append(1) # Fx
    else: action.append(0)        # Fy

# Used for new action-space
# action = [1,0]

for _ in range(1):
    plt.close('all')
    obs = env.reset()

    if dataCollect:
        start_time = time.time()
    for i in tqdm(range(maxNumSteps)):

        if render:
            env.render()
        obs, rew, done, _ = env.step(action)

        if done: 
            print('Done')
            break

    print('End of episode')
    
    if dataCollect:
        end_time = time.time()
        runtime = end_time - start_time
        save_runtime(env.saveFolder, 'Runtime', runtime)
        env.dataExport()
        env.exportAllData()
    if saveVideo:
        createVideo(env.saveFolder, env.videoFolder, experimentName, (width, height))
    env.close()
    plt.close('all')
# %%
