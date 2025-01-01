import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import typing
import os
from pathlib import Path

from trading_environment import TradingEnv
from stable_baselines3 import A2C,PPO,DQN
from qlearning import QLearning

def update_dict(main: dict, config: dict):
    """
    Recursively updates the `main` dictionary with values from the `config` dictionary.

    Parameters:
    -----------
        main : dict
            The primary dictionary that will be updated in-place.
        config : dict
            The dictionary containing updates. If a key in `config` corresponds to a dictionary, 
            the function performs a recursive update; otherwise, it overwrites the value in `main`.

    Returns:
    --------
        None
            The function modifies the `main` dictionary in-place and does not return any value.
    """
    
    for key, value in config.items():
        if isinstance(value, dict):
            update_dict(main[key], value)
        else:
            main[key] = value


def max_drawdown(portforlio_history:typing.Iterable) -> float:
    """
    Calculate the maximum drawdown of a portfolio based on its historical value.

    Parameters:
    -----------
        portfolio_history : iterable.An iterable (e.g., list or array) containing the portfolio's historical values over time.

    Returns:
    --------
        float
            The maximum drawdown as a percentage, representing the largest peak-to-trough decline
            in the portfolio's value.
    """

    portforlio_history=pd.Series(portforlio_history)
    running_max=portforlio_history.cummax()
    drawdown=(running_max-portforlio_history)/running_max
    drawdown=drawdown.max()
    return drawdown*100

def PnL(portforlio_history:typing.Iterable) -> float:
    """
    Calculate the profit and loss (PnL) of a portfolio based on its historical value.

    Parameters:
    -----------
        portfolio_history : iterable
            An iterable (e.g., list or array) containing the portfolio's historical values over time.

    Returns:
    --------
        float
            The difference between the final and initial portfolio values, representing the 
            profit or loss over the given period.
    """

    portforlio_history=pd.Series(portforlio_history)
    start_portforlio=portforlio_history.iloc[0]
    end_portforlio=portforlio_history.iloc[-1]
    return end_portforlio-start_portforlio

def ROI(portforlio_history:typing.Iterable) -> float:
    """
    Calculate the Return on Investment (ROI) of a portfolio based on its historical value.

    Parameters:
    -----------
        portfolio_history : iterable
            An iterable (e.g., list or array) containing the portfolio's historical values over time.

    Returns:
    --------
        float
            The percentage return on investment, calculated as the difference between the final
            and initial portfolio values, relative to the initial value.
    """

    portforlio_history=pd.Series(portforlio_history)
    start_portforlio=portforlio_history.iloc[0]
    end_portforlio=portforlio_history.iloc[-1]
    return (end_portforlio-start_portforlio)*100/start_portforlio

class Trading:
    
    def __init__(self,algorithm:typing.Callable,parameters:dict = {}) -> None:
        self._algorithm = algorithm
        self._model = None
        self._parameters = parameters

    def learn(self,total_timesteps:int = 10000,**kwargs) -> None:
        self._model = self._algorithm(**self._parameters)
        self._model.learn(total_timesteps,**kwargs)

    def save(self,path:str):
        self._model.save(path)

    def load(self,path) -> None:
        if self._model is None:
            self._model = self._algorithm.load(path)

    def predict(self,state:np.ndarray,desterministic:bool = False) -> typing.Iterable:
        return self._model.predict(state,desterministic)

    def evaluate(self,env:typing.Any,show_fig = True) -> dict:
        """
        Evaluate the model's performance on the test set by simulating trading actions and calculating key metrics.

        Parameters:
        -----------
            None

        Returns:
        --------
            None
                This method doesn't return any values, but prints the evaluation metrics (Max drawdown, PnL, ROI) 
                and displays a plot showing the trading actions ('B' for buy, 'S' for sell) along with the 
                closing price during the test period.
        """


        current_state,info=env.reset()

        portforlio_history=[env.total]
        action_list=[]

        while True:

            action,_=self.predict(current_state,False)
            action=int(action)
            current_state,reward,terminated,truncated,info=env.step(action)

            portforlio_history.append(env.total)
            action_list.append(action)

            if terminated:
                break

        drawdown=max_drawdown(portforlio_history)
        pnl=PnL(portforlio_history)
        roi=ROI(portforlio_history)

        print(f'Max drawdown : {round(drawdown,3)} %')
        print(f'PnL : {round(pnl,3)} $')
        print(f'Roi : {round(roi,3)} %')

        if show_fig:
            fg=plt.figure()
            ax_1=fg.add_subplot(2,1,1)
            ax_2=fg.add_subplot(2,1,2)

            env._close.plot(ax=ax_1)
            for i in range(len(action_list)):
                if action_list[i]==0:
                    ax_1.text(i,env._close.iloc[i],'B',color='C2')
                elif action_list[i]==2:
                    ax_1.text(i,env._close.iloc[i],'S',color='C3')

            ax_2.plot(portforlio_history)
            plt.tight_layout()
            plt.show()

        return  {
                'Max drawdown':drawdown,
                'PnL':pnl,
                'ROI':roi
            }

if __name__ == '__main__':
    seed = np.linspace(0,4*np.pi,10)
    # noise = 0.2*np.random.randn(1000)

    m = pd.DataFrame(4*np.sin(0.5*seed)+10,columns = ['Close'])
    env = TradingEnv(df = m,target = 229.99)
    params = {
            'env':env,
            'policy':'MlpPolicy',
            # 'render':False,
            'verbose':1,
            # 'qtable_size':[100,100,2,1200]
        }
    trade = Trading(DQN,params)
    trade.learn(20000)
    trade.evaluate(env)

