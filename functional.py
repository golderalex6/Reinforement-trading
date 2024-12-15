import pandas as pd
import matplotlib.pyplot as plt

import json
import os
from pathlib import Path
from abc import ABC,abstractmethod
import typing

from trading_environment import TradingEnv

plt.rc('figure',titleweight='bold',titlesize='large',figsize=(15,6))
plt.rc('axes',labelweight='bold',labelsize='large',titleweight='bold',titlesize='large',grid=True)

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


class agent():

    def __init__(self,metadata:dict,config:dict = {}) -> None:
        """
        Initialize the class by loading model parameters from a JSON file.

        Parameters:
        -----------
            None

        Returns:
        --------
            None
        """
        with open(os.path.join(Path(__file__).parent,'parameters.json'),'r+') as f:
            self._parameters=json.loads(f.read())

        self._metadata = metadata
        update_dict(self._metadata,config)

    def _setup(self) -> None:
        """
        Set up the training and testing environments by loading historical data and splitting it.

        Parameters:
        -----------
            None

        Returns:
        --------
            None
        """
        
        df=pd.read_csv(os.path.join(Path(__file__).parent,'data',self._parameters['symbol'],f"{self._parameters['symbol']}_1d.csv"),index_col=0)
        self._df_train=df.loc[self._parameters['start_date']:self._parameters['test_date']]
        self._df_test=df.loc[self._parameters['test_date']:self._parameters['end_date']]

        self._env_train = TradingEnv(df = self._df_train,window_size = self._parameters['window_size'])
        self._env_test = TradingEnv(df = self._df_test,window_size = self._parameters['window_size'])

    def get_config(self) -> dict:

        return self._metadata

    @abstractmethod
    def load(self) -> typing.Any:
        '''
        Abstract method
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
        return self._model.predict(state,deterministic = True)

    def evaluate(self,show_fig = True) -> dict:
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

        self._setup()

        portforlio_history=[]
        action_list=[]

        current_state,info=self._env_train.reset()
        while True:
            action,_=self.predict(current_state)
            action=int(action)
            current_state,reward,terminated,truncated,info=self._env_test.step(action)

            portforlio_history.append(self._env_test.total)
            action_list.append(action)

            if terminated:
                break

        drawdown=max_drawdown(portforlio_history)
        pnl=PnL(portforlio_history)
        roi=ROI(portforlio_history)

        print(f'Max drawdown : {round(drawdown,3)}')
        print(f'PnL : {round(pnl,3)}')
        print(f'Roi : {round(roi,3)}')

        if show_fig:
            fg=plt.figure()
            ax=fg.add_subplot()
            self._df_test['Close'].plot(ax=ax)
            for i in range(len(action_list)):
                if action_list[i]==0:
                    ax.text(i,self._df_test.iloc[i,0],'B',color='C2')
                elif action_list[i]==2:
                    ax.text(i,self._df_test.iloc[i,0],'S',color='C3')
            ax.set_title(f"{self._parameters['symbol']} {self._parameters['test_date']}:{self._parameters['end_date']}")
            plt.show()

        return  {
                'Max drawdown':drawdown,
                'PnL':pnl,
                'ROI':roi
            }

if __name__=='__main__':
    from metadata import a2c_metadata
    m = agent(a2c_metadata.METADATA)
    m._setup()
    print(m._df_train)
