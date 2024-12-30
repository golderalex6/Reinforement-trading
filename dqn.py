import os
from pathlib import Path

import torch
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from functional import agent
from metadata import dqn_metadata

class DqnTrading(agent):

    def __init__(self,config:dict = {}) -> None:
        """
        Initializes a DQN trading agent with metadata loaded from a JSON file.

        Parameters:
        -----------
            None

        Returns:
        --------
            None
        """
        super().__init__(dqn_metadata.METADATA,config)


    def learn(self,total_timesteps:int = 40000):
        """
        Trains the DQN agent for a specified number of timesteps and saves the trained model.

        Parameters:
        -----------
            total_timesteps (int, optional): The total number of timesteps to train the model. Defaults to 40,000.

        Returns:
        --------
            None
        """ 
        
        self._setup()
        self._model = DQN(
                    env = self._env_train,
                    **self._metadata
                )

        self._model.learn(total_timesteps = total_timesteps)
        self._model.save(os.path.join(Path(__file__).parent,'models','dqn.zip'))

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
            path = os.path.join(Path(__file__).parent,'models','dqn.zip')
        self._model = DQN.load(path)

if __name__ == '__main__':
    dqn = DqnTrading()
    # dqn.learn()
    dqn.load()
    dqn.evaluate()
