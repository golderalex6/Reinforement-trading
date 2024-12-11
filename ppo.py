from functional import *

class PpoTrading(agent):

    def __init__(self) -> None:
        """
        Initialize the class by loading metadata and setting up the policy parameters.

        This constructor loads a JSON metadata file that contains model architecture details 
        and optimizer settings. It then sets up the policy network architecture and optimizer 
        class using the information from the metadata.

        Parameters:
        -----------
            None

        Returns:
        --------
            None
                Initializes the class with the loaded metadata and sets the policy network architecture 
                and optimizer parameters.
        """

        super().__init__()

        with open(os.path.join(Path(__file__).parent,'metadata','ppo_metadata.json'),'r+') as f:
            self._metadata = json.load(f)
        
        self._policy_kwargs = {
                    'net_arch':self._metadata['layers'],
                    'activation_fn':getattr(nn,self._metadata['activation']),
                    'optimizer_class':getattr(optim,self._metadata['optimizer'])
                }

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
                    policy = self._metadata['policy'],
                    env = self._env_train,
                    learning_rate = self._metadata['learning_rate'],
                    batch_size = self._metadata['batch_size'],
                    n_epochs = self._metadata['epochs'],
                    policy_kwargs=self._policy_kwargs,
                    verbose=1,
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
    # ppo.learn()
    ppo.load()
    ppo.evaluate()
