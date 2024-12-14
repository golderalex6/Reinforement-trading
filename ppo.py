import os
from pathlib import Path

from torch import nn,optim
from stable_baselines3 import PPO

from functional import agent
from metadata import ppo_metadata

class PpoTrading(agent):

    def __init__(self,config:dict = {}) -> None:
        """
        Initializes the class by loading metadata and setting up policy keyword arguments.

        Parameters:
        -----------
            None

        Returns:
        --------
            None
        """

        super().__init__(ppo_metadata.METADATA,config)

    def learn(self,total_timesteps:int = 10000) -> None:
        """
        Train the PPO model using the specified number of timesteps and save the trained model.

        Parameters:
        -----------
            total_timesteps : int, optional, default=10000
                The total number of timesteps to train the model.

        Returns:
        --------
            None
                Trains the model on the training environment and saves the trained model to disk.
        """

        self._setup()
        self._model = PPO(
                    env = self._env_train,
                    **self._metadata
                )
        self._model.learn(total_timesteps=total_timesteps)
        self._model.save(os.path.join(Path(__file__).parent,'models','ppo.zip'))

    def load(self,path:str = '') -> None:
        """
        Loads a pre-trained PPO model from the specified path.

        Defaults to './models/ppo.zip' in the script's directory if no path is provided.

        Parameters:
        -----------
            path (str): File path to the model. Defaults to an empty string.

        Returns:
        --------
            None
        """

        if path == '':
            path = os.path.join(Path(__file__).parent,'models','ppo.zip')
        self._model = PPO.load(path)

if __name__ == '__main__': 
    ppo=PpoTrading()
    ppo.learn()
    # ppo.load()
    ppo.evaluate()
