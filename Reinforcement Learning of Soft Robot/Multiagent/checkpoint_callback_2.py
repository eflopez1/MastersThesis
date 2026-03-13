import os
from stable_baselines3.common.callbacks import BaseCallback

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