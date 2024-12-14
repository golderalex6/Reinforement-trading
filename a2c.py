import os
from pathlib import Path

from stable_baselines3 import A2C

from functional import agent
from metadata import a2c_metadata
class A2cTrading(agent):

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

        super().__init__(a2c_metadata.METADATA,config)
        
    def learn(self,total_timesteps:int = 10000) -> None:
        """
        Trains an A2C (Advantage Actor-Critic) model and saves it.

        Parameters:
        -----------
            total_timesteps : int,total timesteps for training. Default is 10,000.

        Returns:
        --------
            None
        """

        self._setup()
        self._model = A2C(
                        env = self._env_train,
                        **self._metadata
                    )

        self._model.learn(total_timesteps=total_timesteps)
        self._model.save(os.path.join(Path(__file__).parent,'models','a2c.zip'))

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
            path = os.path.join(Path(__file__).parent,'models','a2c.zip')
        self._model = A2C.load(path)

if __name__ == '__main__': 
    config = {
            'policy_kwargs': {
                'net_arch':[100,10]
                }
        }
    a2c = A2cTrading(config)
    # a2c.learn()
    a2c.load()
    a2c.evaluate()

