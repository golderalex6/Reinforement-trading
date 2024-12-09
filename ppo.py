from functional import *

class ppo_trading(agent):

    def __init__(self) -> None:

        super().__init__()

        with open(os.path.join(Path(__file__).parent,'metadata','ppo_metadata.json'),'r+') as f:
            self._metadata = json.load(f)
        
        self._policy_kwargs = {
                    'net_arch':self._metadata['layers'],
                    'activation_fn':getattr(nn,self._metadata['activation']),
                    'optimizer_class':getattr(optim,self._metadata['optimizer'])
                }

    def learn(self,total_timesteps:int = 10000) -> None:

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
if __name__ == '__main__': 
    ppo=ppo_trading()
    ppo.learn()
    ppo.evaluate()
