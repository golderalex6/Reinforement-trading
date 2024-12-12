import os
from pathlib import Path

import torch
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from functional import agent

class LstmExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            lstm_hidden_size:int = 128,
            lstm_num_layers:int = 1,
            lstm_dropout:float = 0.0
        ) -> None:
        """
        Initializes an LSTM-based neural network for processing sequential data in a reinforcement learning setting.

        Parameters:
        -----------
            observation_space (gym.Space): The observation space of the environment, defining the input shape for the network.
            lstm_hidden_size (int, optional): The number of features in the hidden state of the LSTM. Defaults to 128.
            lstm_num_layers (int, optional): The number of recurrent layers in the LSTM. Defaults to 1.
            lstm_dropout (float, optional): The dropout probability for the LSTM layers. Defaults to 0.0.

        Returns:
        --------
            None
        """

        super().__init__(observation_space, lstm_hidden_size)
        
        input_size = observation_space.shape[1]
        self.lstm = torch.nn.LSTM(
                    input_size = input_size, 
                    hidden_size = lstm_hidden_size, 
                    num_layers = lstm_num_layers, 
                    batch_first = True, 
                    dropout = lstm_dropout
                )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Processes input observations through the LSTM and returns the final hidden state.

        Parameters:
        -----------
            observations (torch.Tensor): Input tensor containing sequential data, with shape (batch_size, sequence_length, input_size).

        Returns:
        --------
            torch.Tensor: The final hidden state of the LSTM, with shape (batch_size, hidden_size).
        """
        lstm_out, (h_n, c_n) = self.lstm(observations)
        return h_n[-1]

class DqnTrading(agent):

    def __init__(self) -> None:
        """
        Initializes a DQN trading agent with metadata loaded from a JSON file.

        Parameters:
        -----------
            None

        Returns:
        --------
            None
        """

        super().__init__(os.path.join(Path(__file__).parent,'metadata','dqn.json'))
        
        self._policy_kwargs = {
                    'net_arch':self._metadata['layers'],
                    'activation_fn':getattr(torch.nn,self._metadata['activation']),
                    'optimizer_class':getattr(torch.optim,self._metadata['optimizer']),
                    'features_extractor_class':LstmExtractor,
                }

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
                    policy = self._metadata['policy'],
                    env = self._env_train,
                    learning_rate = self._metadata['learning_rate'],
                    batch_size = self._metadata['batch_size'],
                    policy_kwargs=self._policy_kwargs,
                    verbose=1,
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
