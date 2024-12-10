import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json
import os
from pathlib import Path
from abc import ABC,abstractmethod
# from typing import Callable,Iterable,Any,Type
import typing

import gymnasium as gym
from trading_environment import trading_env
import torch
from torch import nn,optim
from stable_baselines3 import A2C,PPO,DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

plt.rc('figure',titleweight='bold',titlesize='large',figsize=(15,6))
plt.rc('axes',labelweight='bold',labelsize='large',titleweight='bold',titlesize='large',grid=True)


def max_drawdown(portforlio_history):
    portforlio_history=pd.Series(portforlio_history)
    running_max=portforlio_history.cummax()
    drawdown=(running_max-portforlio_history)/running_max
    drawdown=drawdown.max()
    return drawdown*100

def PnL(portforlio_history):
    portforlio_history=pd.Series(portforlio_history)
    start_portforlio=portforlio_history.iloc[0]
    end_portforlio=portforlio_history.iloc[-1]
    return end_portforlio-start_portforlio

def ROI(portforlio_history):
    portforlio_history=pd.Series(portforlio_history)
    start_portforlio=portforlio_history.iloc[0]
    end_portforlio=portforlio_history.iloc[-1]
    return (end_portforlio-start_portforlio)*100/start_portforlio


class agent(ABC):
    def __init__(self) -> None:
        self._model = None

        with open(os.path.join(Path(__file__).parent,'parameters.json'),'r+') as f:
            self._parameters=json.loads(f.read())

    def _setup(self) -> None:
        
        df=pd.read_csv(os.path.join(Path(__file__).parent,'data',self._parameters['symbol'],f"{self._parameters['symbol']}_1d.csv"),index_col=0)
        self._df_train=df.loc[self._parameters['start_date']:self._parameters['test_date']]
        self._df_test=df.loc[self._parameters['test_date']:self._parameters['end_date']]

        self._env_train = trading_env(df = self._df_train,window_size = self._parameters['window_size'])
        self._env_test = trading_env(df = self._df_test,window_size = self._parameters['window_size'])

    @abstractmethod
    def load(self) -> typing.Any:
        '''
        abstract method
        '''
        pass

    def predict(self,state:typing.Any) -> typing.Iterable:

        """
        Predicts the action(s) for a given state using the loaded PPO model.

        Args:
            state (Any): The input state for which to predict the action(s).

        Returns:
            Iterable: The predicted action(s) as output by the model.
        """
        return self._model.predict(state)

    def evaluate(self) -> None:
        """

        """
        self._setup()

        self.portforlio_history=[]
        self.action_list=[]

        current_state,info=self._env_test.reset()
        while True:
            action,_=self.predict(current_state)
            action=int(action)
            current_state,reward,terminated,truncated,info=self._env_test.step(action)

            self.portforlio_history.append(self._env_test.total)
            self.action_list.append(action)

            if terminated:
                break

        drawdown=max_drawdown(self.portforlio_history)
        pnl=PnL(self.portforlio_history)
        roi=ROI(self.portforlio_history)

        print(f'Max drawdown : {round(drawdown,3)}')
        print(f'PnL : {round(pnl,3)}')
        print(f'Roi : {round(roi,3)}')

        fg=plt.figure()
        ax=fg.add_subplot()
        self._df_test['Close'].plot(ax=ax)
        for i in range(len(self.action_list)):
            if self.action_list[i]==0:
                ax.text(i,self._df_test.iloc[i,0],'B',color='C2')
            elif self.action_list[i]==2:
                ax.text(i,self._df_test.iloc[i,0],'S',color='C3')
        ax.set_title(f"{self._parameters['symbol']} {self._parameters['test_date']}:{self._parameters['end_date']}")
        plt.show()
    
if __name__=='__main__':
    pass
