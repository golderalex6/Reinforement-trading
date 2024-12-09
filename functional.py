import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

from stable_baselines3 import A2C,PPO,DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gymnasium as gym
from trading_environment import trading_env

from torch import nn,optim
import torch
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


class agent():
    def __init__(self) -> None:
        # self.SEED = 44
        # self._model = None

        with open(os.path.join(Path(__file__).parent,'parameters.json'),'r+') as f:
            self._parameters=json.loads(f.read())

    def _setup(self) -> None:
        
        df=pd.read_csv(os.path.join(Path(__file__).parent,'data',self._parameters['symbol'],f"{self._parameters['symbol']}_1d.csv"),index_col=0)
        self._df_train=df.loc[self._parameters['start_date']:self._parameters['test_date']]
        self._df_test=df.loc[self._parameters['test_date']:self._parameters['end_date']]

        self._env_train = trading_env(df = self._df_train,window_size = self._parameters['window_size'])
        self._env_test = trading_env(df = self._df_test,window_size = self._parameters['window_size'])
    
    def evaluate(self) -> None:

        self.portforlio_history=[]
        self.action_list=[]

        current_state,info=self._env_test.reset()
        while True:
            action,_=self._model.predict(current_state)
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
