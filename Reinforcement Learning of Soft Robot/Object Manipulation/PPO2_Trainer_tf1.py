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
    copyfile(master_env.__file__,savefile+'Environment_PF.py')
    copyfile(RL_params_tf1.__file__,savefile+'RL_params_tf1.py')
    copyfile(env_params.__file__,savefile+'env_params.py')
    copyfile(__file__, savefile+'Trainer.py')
   
    # Setting up Callback
    checkpoint_callback = CheckpointCallback2(
        save_freq = check_freq,
        save_path = savefile
    )

    # Assert that we are not gathering simulation information during training
    envParams['dataCollect']=False
    envParams['saveVideo']=False

    # Creating multiple environments for multiple workers
    training_env = make_vec_env(pymunkEnv, n_envs= nEnvs, env_kwargs=envParams, vec_env_cls=SubprocVecEnv, monitor_dir = savefile)

    # Create an environment which will be used for mid-training results
    env_mid_training = pymunkEnv
    envParams_standard = envParams
    envParams_standard['dataCollect'] = True
    envParams_standard['saveVideo'] = True

    # Number of sub-timesteps for each training
    sub_timesteps = training_timesteps//divisor
    assert training_timesteps%divisor == 0, "The number of training sub timesteps is not divisible by {}!!".format(divisor)

    # Start training time
    learn_start = time.time()

    # Create the model 
    model = PPO2('MlpPolicy', training_env, verbose=1, policy_kwargs=policy_kwargs,
                    gamma=gamma, 
                    n_steps = n_steps, 
                    ent_coef = ent_coef,
                    learning_rate = learning_rate,
                    vf_coef = vf_coef,
                    nminibatches = nminibatches,
                    noptepochs = noptepochs,
                    cliprange = cliprange,
                    tensorboard_log = savefile,
                    seed = seed)
                    
    # pretraining the model
    # from stable_baselines.gail import ExpertDataset
    # pretrain_batchsize = 10
    # pretrain_epochs = 1000
    # dataset = ExpertDataset("Imitation_Learning/Expert_Trajectory_3/Expert_Trajectory_3.npz", traj_limitation=-1, batch_size=pretrain_batchsize, randomize=True)
    # model.pretrain(dataset,n_epochs=pretrain_epochs)

    # model = PPO2.load(
    #     'Experiment 31_phase3 2022-02-07/rl_model_8400000_steps.zip',
    #     env = training_env
    # )

    # Train the model over multiple iterations
    iterations = training_timesteps // sub_timesteps
    for iter in range(iterations):
        iter += 1

        if iter==np.inf:
            print('--'*20)
            none = input("There was a modification on sub_timesteps in Training Script!!\nPress Enter to acknowledge.\n")
            print('--'*20)
            temp_sub_timesteps = 1_600_000 # Needed to complete iteration 4
            model.learn(total_timesteps=temp_sub_timesteps, callback=checkpoint_callback)
        else:
            model.learn(total_timesteps=sub_timesteps, callback=checkpoint_callback)
        model.save(savefile + 'Iteration_{}_agent'.format(iter)) 
        checkpoint_callback.last_time_trigger = 0 # Resetting the checkpoint callback

        for test_num in range(3):
            test_num += 1
            name = "Iteration_{}_v{}".format(iter,test_num)
            envParams_standard['experimentName'] = savefile + name
            env_standard_test = env_mid_training(**envParams_standard)
            env_standard_test.report_all_data = True
            obs = env_standard_test.reset()
            
            for j in tqdm(range(time_per_test)):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env_standard_test.step(action)
            
                env_standard_test.render()
                    
                if done: 
                    break
            
            env_standard_test.dataExport() # Export the data from the simulation
            plt.close('all')
            createVideo(savefile, env_standard_test.videoFolder, name, (width, height))
            env_standard_test.exportAllData()
            env_standard_test.close()


    learn_end = time.time()
    
    # Save information on how long training took
    runtime = learn_end - learn_start
    save_runtime(savefile, 'Training_Time', runtime)
    
    # Save the model
    model.save(savefile + experimentName + '_agent')