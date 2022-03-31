# %% Main cell
"""
Main execution script for training the obstacle field environment
Uses stable baselines 2 as their means of describing the environment.
"""

if __name__ == '__main__':
    # Importing Environment and environment dependencies
    from env_params import *

    # Importing RL parameters and dependencies
    from RL_params_tf1 import *

    # Creating directory to save all results in
    mainDirectory = str(pathlib.Path(__file__).parent.absolute()) # Get the path of this file
    savefile = mainDirectory + '\\Experiment {} {}\\'.format(str(experimentNum), date.today())
    os.makedirs(savefile, exist_ok=True)
    
    # Importing and saving these files
    import env_params, RL_params_tf1
    copyfile(master_env.__file__,savefile+'Environment.py'.format(str(experimentNum)))
    copyfile(RL_params_tf1.__file__,savefile+'RL_params_tf1.py'.format(str(experimentNum)))
    copyfile(env_params.__file__,savefile+'env_params.py'.format(str(experimentNum)))
    copyfile(__file__, savefile+'Trainer_tf1.py'.format(str(experimentNum)))
   
    # Setting up Callback
    checkpoint_callback = CheckpointCallback2(
        save_freq=check_freq, 
        save_path = savefile 
    ) 

    # Assert that we are not gathering simulation information during training
    envParams['dataCollect']=False
    envParams['saveVideo']=False

    # Creating multiple environments for multiple workers
    training_env = make_vec_env(pymunkEnv, n_envs= nEnvs, env_kwargs=envParams, vec_env_cls=SubprocVecEnv, monitor_dir = savefile)

    # Environment and parametrs for sub-training runs
    env_testing = pymunkEnv 
    envParams_testing = envParams
    envParams_testing['dataCollect']=True
    envParams_testing['saveVideo']=True
    
    # Number of training timesteps for each sub iteration, afterwhich we check in on policy
    sub_timesteps = training_timesteps//divisor
    assert training_timesteps%divisor == 0, "The number of training timesteps is not divisble by {}!!!".format(divisor)

    # Train the model
    learn_start = time.time()

    # Create the RL model
    model = PPO2('MlpPolicy', training_env, verbose=1,
                gamma=gamma, 
                n_steps = n_steps, 
                ent_coef = ent_coef,
                learning_rate = learning_rate,
                vf_coef = vf_coef,
                nminibatches = nminibatches,
                noptepochs = noptepochs,
                cliprange = cliprange,
                tensorboard_log = savefile,
                seed = seed,
                policy_kwargs=policy_kwargs)

    # Continuing training of some other model
    # model = PPO2.load(
    #     "Experiment 21a_ModObsact_NumStep200 2022-02-28/rl_model_400000_steps.zip",
    #     env=training_env
    # )
    # model.tensorboard_log = savefile

    # Iterate through each sub timestep training
    iterations = training_timesteps//sub_timesteps
    for iter in range(iterations):
        iter+=1

        if iter==np.inf:
            none = input("There was a modification on sub_timesteps in Training Script!!\nPress Enter to acknowledge.\n")
            temp_sub_timesteps = 9_600_000 # Needed to complete iteration 4
            model.learn(total_timesteps=temp_sub_timesteps, callback=checkpoint_callback)
        else:
            model.learn(total_timesteps=sub_timesteps, callback=checkpoint_callback)
        model.save(savefile + 'Iteration_{}_agent'.format(iter))
        checkpoint_callback.last_time_trigger = 0 # Resetting the checkpoint callback back to 0

        for test_num in range(3):
            test_num+=1
            name = "Iteration_{}_v{}".format(iter,test_num)
            envParams_testing['experimentName'] = savefile + name
            env_testing_test = env_testing(**envParams_testing)
            env_testing_test.report_all_data = True
            obs = env_testing_test.reset()
            
            for j in tqdm(range(time_per_test)):
                action, _states = model.predict(obs, deterministic=False)
                obs, reward, done, info = env_testing_test.step(action)
            
                env_testing_test.render()
                    
                if done: 
                    break
            
            env_testing_test.dataExport() # Export the data from the simulation
            plt.close('all')
            createVideo(savefile, env_testing_test.videoFolder, name, (width, height))
            env_testing_test.exportAllData()
            env_testing_test.close()

    # Save information on how long training took
    learn_end = time.time()
    runtime = learn_end - learn_start
    save_runtime(savefile, 'Training_Time', runtime)
    
    # Save the model
    model.save(savefile + experimentName + '_agent')
