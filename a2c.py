from functional import *

class a2c_trading(agent):

    def __init__(self):

        super().__init__()
        with open(os.path.join(Path(__file__).parent,'metadata','a2c_metadata.json'),'r+') as f:
            self._metadata = json.load(f)
        
        self._policy_kwargs = {
                    'net_arch':self._metadata['layers'],
                    'activation_fn':getattr(nn,self._metadata['activation']),
                    'optimizer_class':getattr(optim,self._metadata['optimizer']),
                }

    def learn(self,total_timesteps:int = 10000) -> None:

        self._setup()
        self._model = A2C(
                        policy = self._metadata['policy'],
                        env = self._env_train,
                        learning_rate = self._metadata['learning_rate'],
                        policy_kwargs=self._policy_kwargs,
                        verbose=1,
                    )

        self._model.learn(total_timesteps=total_timesteps)
        self._model.save(os.path.join(Path(__file__).parent,'models','a2c.zip'))

    def load(self,path:str = '') -> None:
        """
        Loads a pre-trained PPO model from the specified path.

        Defaults to './models/ppo.zip' in the script's directory if no path is provided.

        Args:
            path (str): File path to the model. Defaults to an empty string.

        Returns:
            None
        """

        if path == '':
            path = os.path.join(Path(__file__).parent,'models','a2c.zip')
        self._model = A2C.load(path)

if __name__ == '__main__': 
    a2c = a2c_trading()
    # a2c.learn()
    a2c.load()
    a2c.evaluate()

