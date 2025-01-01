import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
import os
from pathlib import Path
import pickle
import typing

from trading_environment import TradingEnv
from functional import agent
from metadata import ql_metadata

class QLearning:
    def __init__(
            self,env:typing.Any,
            learning_rate:float = 0.01,
            discount_value:float = 0.99,
            epsilon:float = 0.99,
            epsilon_decay:float = 0.01,
            qtable_size:list = [7,20,2,50],
            render:bool = True,
            verbose:bool = True
        ) -> None:
        """

        """

        self._env = env
        self._render = render
        self._verbose = verbose

        self._learning_rate = learning_rate
        self._discount_value = discount_value

        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay

        self._qtable_size = qtable_size
        self._qtable_segment = (self._env.observation_space.high - self._env.observation_space.low) / self._qtable_size
        self._qtable = np.random.uniform(low=-2, high=-1, size=(self._qtable_size + [self._env.action_space.n]))

    def _convert_state(self,real_state:np.ndarray) -> typing.Iterable:
        """

        """
        q_state = (real_state - self._env.observation_space.low) // (self._qtable_segment*1.00001)
        return tuple(q_state.astype(int))

    def learn(self,total_timesteps = 10000) -> None:
        """

        """
        for step in range(total_timesteps):

            next_real_state,_=self._env.reset()
            current_state=self._convert_state(next_real_state)

            episode_reward = 0
            action_list = []

            while True:

                action = np.argmax(self._qtable[current_state])

                next_real_state, reward, terminate,truncated, _  = self._env.step(action=action)

                action_list.append(action)
                episode_reward += reward

                next_state = self._convert_state(next_real_state)
                current_q_value = self._qtable[current_state + (action,)]
                new_q_value = (1 - self._learning_rate) * current_q_value + self._learning_rate * (reward + self._discount_value * np.max(self._qtable[next_state]))
                self._qtable[current_state + (action,)] = new_q_value
                current_state = next_state

                if terminate:
                    if self._env.goal and self._verbose:
                        print(f"The agent reached the goal at : {step}")
                    break

                if self._render:
                    self._env.render()

    def predict(self,state:typing.Any,deterministic = False):
        """

        """
        action_values = self._qtable[self._convert_state(state)]
        action = None
        if deterministic:
            action = np.argmax(action_values)
        else:
            probabilities = np.e**action_values/np.sum(np.e**action_values)
            action = random.choices(range(len(action_values)),probabilities)[0]
        return action,None

    def save(self,path) -> None:
        """

        """

        with open(path,'wb') as f:
            pickle.dump(self._qtable,f)

    @staticmethod
    def load(path:str = '') -> None:
        """

        """
        with open(path,'rb') as f:
            qtable = pickle.load(f)

        return qtable

if __name__ == '__main__':
    seed = np.linspace(0,4*np.pi,10)

    m = pd.DataFrame(4*np.sin(0.5*seed)+10,columns = ['Close'])
    env = TradingEnv(df = m,target = 229.99)
    ql = QLearning(env,render = False)
    ql.learn(total_timesteps = 50000)
