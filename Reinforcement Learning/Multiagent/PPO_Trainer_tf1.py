# %% 
# Main cell.
"""
Main execution script for training the grabbing environment
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
    copyfile(master_env.__file__,savefile+'Environment.py')
    copyfile(RL_params_tf1.__file__,savefile+'RL_params_tf1.py')
    copyfile(env_params.__file__,savefile+'env_params.py')
    copyfile(__file__, savefile+'Trainer.py')
   
    # Setting up Callback
    checkpoint_callback = CheckpointCallback2(
        save_freq=check_freq, 
        save_path = savefile 
    ) 

    # Assert that we are not gathering simulation information during training
    envParams['dataCollect']=False
    envParams['saveVideo']=False

    # Create the environment
    training_env = parallel_env(**envParams)
    if frame_stack > 0:
        training_env = ss.frame_stack_v1(training_env, frame_stack)
    training_env = ss.pettingzoo_env_to_vec_env_v1(training_env)
    training_env = ss.concat_vec_envs_v1(training_env, 1, num_cpus=1, base_class='stable_baselines')

    # Create the testing environment
    envParams_testing = envParams
    envParams_testing['dataCollect'] = True
    envParams_testing['saveVideo'] = True
    env_testing = parallel_env

    # Number of training time steps for each divisor of training
    sub_timesteps = training_timesteps//divisor
    assert training_timesteps%divisor ==0 ,"The number of training timesteps is not divisble by {}!!!".format(divisor)

    # Train the model
    learn_start = time.time()

    # Create the model
    # model = PPO2('MlpPolicy', training_env, verbose=1, policy_kwargs=policy_kwargs,
    #                 gamma=gamma, 
    #                 n_steps = n_steps, 
    #                 ent_coef = ent_coef,
    #                 learning_rate = learning_rate,
    #                 vf_coef = vf_coef,
    #                 nminibatches = batch_size,
    #                 noptepochs = noptepochs,
    #                 cliprange = cliprange,
    #                 tensorboard_log = savefile,
    #                 seed = seed)

    # Loading model to continue training
    model = PPO2.load(
        "Experiment 10d_entropy_framestacking 2022-02-07/rl_model_5700000_steps.zip",
        env = training_env
    )
    model.tensorboard_log = savefile

    iterations = training_timesteps//sub_timesteps
    for iter in range(iterations):
        iter += 1

        model.learn(total_timesteps=sub_timesteps, callback=checkpoint_callback)
        model.save(savefile + 'Iteration_{}_agent'.format(iter))
        checkpoint_callback.last_time_trigger = 0 # Resetting the checkpoint callback

        for test_num in range(3):
            test_num += 1
            name = "Iteration_{}_v{}".format(iter,test_num)
            envParams_testing['experimentName'] = savefile + name
            env_testing_test = env_testing(**envParams_testing)
            env_testing_test.report_all_data = True
            if frame_stack > 0:
               env_testing_test = ss.frame_stack_v1(env_testing_test, frame_stack)
            observation = env_testing_test.reset()  

            for i in tqdm(range(time_per_test)):
                actions = {}
                for agent in env_testing_test.agents:
                    obs = observation[agent]
                    action, _states = model.predict(obs, deterministic=True)
                    actions[agent] = action
                observation, _, dones, _ = env_testing_test.step(actions)

                env_testing_test.render()

                if dones[agent]:
                    break
            env_testing_test = env_testing_test.unwrapped

            env_testing_test.dataExport()
            env_testing_test.exportAllData()
            plt.close('all')
            createVideo(savefile, env_testing_test.videoFolder, name, (width, height))
            env_testing_test.close()

    learn_end = time.time()
    
    # Save information on how long training took
    runtime = learn_end - learn_start
    save_runtime(savefile, 'Training_Time', runtime)
    
    # Save the model
    model.save(savefile + experimentName + '_agent')
    