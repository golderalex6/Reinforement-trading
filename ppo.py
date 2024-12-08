from functional import *

class ppo_trading(agent):

    def __init__(self) -> None:

        super().__init__()
        with open(os.path.join(Path(__file__).parent,'metadata','ppo_metadata.json'),'r+') as f:

            metadata = json.load(f)
            self._layers = metadata['layers']
            self._activation = metadata['activation']
            self._optimizer = metadata['optimizer']
            self._epochs = metadata['epochs']
            self._batch_size = metadata['batch_size']
            self._learning_rate = metadata['learning_rate']
            self._policy = metadata['policy']
        
        self._policy_kwargs = {
                    'net_arch':self._layers,
                    'activation_fn':getattr(nn,self._activation),
                    'optimizer_class':getattr(optim,self._optimizer)
                }

    def learn(self,total_timesteps:int = 10000) -> None:

        self._setup()
        self._model = PPO(
                        policy = self._policy,
                        env = self._env_train,
                        batch_size = self._batch_size,
                        n_epochs = self._epochs,
                        policy_kwargs=self._policy_kwargs,
                        verbose=1
                    )
        self._model.learn(total_timesteps=total_timesteps)
        self._model.save(os.path.join(Path(__file__).parent,'models','ppo.zip'))

ppo=ppo_trading()
ppo.learn()
ppo.evaluate()
