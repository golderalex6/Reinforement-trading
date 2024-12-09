from functional import *

class LSTM_extractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            lstm_hidden_size:int = 128,
            lstm_num_layers:int = 1,
            lstm_dropout:float = 0.0
        ) -> None:

        super().__init__(observation_space, lstm_hidden_size)
        
        input_size = observation_space.shape[1]
        self.lstm = nn.LSTM(
                    input_size = input_size, 
                    hidden_size = lstm_hidden_size, 
                    num_layers = lstm_num_layers, 
                    batch_first = True, 
                    dropout = lstm_dropout
                )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(observations)
        return h_n[-1]

class dqn_trading(agent):

    def __init__(self) -> None:

        super().__init__()
        with open(os.path.join(Path(__file__).parent,'metadata','dqn_metadata.json'),'r+') as f:
            self._metadata = json.load(f)
        
        self._policy_kwargs = {
                    'net_arch':self._metadata['layers'],
                    'activation_fn':getattr(nn,self._metadata['activation']),
                    'optimizer_class':getattr(optim,self._metadata['optimizer']),
                    'features_extractor_class':LSTM_extractor,
                }

    def learn(self,total_timesteps:int = 40000):
        
        self._setup()
        self._model = DQN(
                    policy = self._metadata['policy'],
                    env = self._env_train,
                    learning_rate = self._metadata['learning_rate'],
                    batch_size = self._metadata['batch_size'],
                    policy_kwargs=self._policy_kwargs,
                    verbose=1,
                )

        self._model.learn(total_timesteps = total_timesteps)
        self._model.save(os.path.join(Path(__file__).parent,'models','dqn.zip'))

if __name__ == '__main__':
    dqn = dqn_trading()
    dqn.learn()
    dqn.evaluate()
