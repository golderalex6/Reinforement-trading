import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import typing

import gymnasium as gym
plt.rc('figure',titleweight='bold',titlesize='large',figsize=(15,6))
plt.rc('axes',labelweight='bold',labelsize='large',titleweight='bold',titlesize='large',grid=True)

class TradingEnv(gym.Env):
    def __init__(self,df:pd.DataFrame,window_size=1,initial_balance=10**4) -> None:
        """
        Initializes a custom financial environment for reinforcement learning.

        This function sets up the environment with financial data, defines the observation 
        and action spaces, and initializes key attributes such as the balance, coin holdings, 
        and price tracking.

        Parameters:
        -----------
            df (pd.DataFrame): A DataFrame containing the financial data with at least a 'Close' column.
            window_size (int, optional): The size of the observation window for the environment. 
                Defaults to 1.
            initial_balance (float, optional): The initial USD balance for the simulation. 
                Defaults to 10,000.

        Returns:
        --------
            None
        """
        super().__init__()

        self.window_size=window_size
        self.df=df.copy()
        self.df['Diff_pct']=self.df['Close'].pct_change(1).fillna(0)*100
        self.processed_df=self.df[['Close','Diff_pct']]

        self.index=self.window_size
        
        self.observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size,self.processed_df.shape[1]),dtype=float)
        self.action_space=gym.spaces.Discrete(3)

        self.observation_max=self.processed_df.max().values
        self.observation_min=self.processed_df.min().values
        
        self.initial_balance=initial_balance
        self.usd=self.initial_balance
        self.coin=0

        self.total=self.initial_balance

        self.buy_price=self.df.iloc[self.index,0]
        self.sell_price=self.df.iloc[self.index,0]

    def step(self,action:int) -> typing.Iterable:
        """
        Performs a single step in the financial trading environment based on the given action.

        This method updates the environment's state, calculates the reward, and determines 
        whether the episode should terminate or truncate. It handles the buying, selling, 
        or holding of assets and adjusts the balance and coin holdings accordingly.

        Parameters:
        -----------
            action (int): The action to be taken. 
                - 0: Buy
                - 1: Hold
                - 2: Sell

        Returns:
        --------
            typing.Iterable: A tuple containing:
                - new_state (np.ndarray): The updated state of the environment after the action.
                - reward (float): The reward obtained from taking the action.
                - terminate (bool): Whether the episode has ended.
                - truncate (bool): Whether the episode was truncated.
                - info (dict): Additional information (empty dictionary in this case).
        """

        self.total=self.processed_df.iloc[self.index,0]*self.coin+self.usd
        reward=0
        new_state=self.processed_df.iloc[self.index-self.window_size+1:self.index+1].values
        terminate,truncate=False,False

        if self.index==self.df.shape[0]-1:
            terminate,truncate=True,True
        if action==2:
            if self.coin>0:
                reward=self.df.iloc[self.index,0]-self.buy_price
                self.sell_price=self.df.iloc[self.index,0]

                self.usd=self.coin*self.df.iloc[self.index,0]
                self.coin=0
            else:
                reward = -0.5
        elif action==1:
            if self.coin>0:
                reward=(self.df.iloc[self.index,0]-self.buy_price)*0.1
            else:
                reward=(self.sell_price-self.df.iloc[self.index,0])*0.1
        elif action==0:
            if self.usd>0:
                reward=self.sell_price-self.df.iloc[self.index,0]
                self.buy_price=self.df.iloc[self.index,0]

                self.coin=self.usd/self.df.iloc[self.index,0]
                self.usd=0
            else:
                reward = -0.5

        self.index+=1
        return new_state,reward,terminate,truncate,{}


    def reset(self,seed=None,options=None) -> typing.Iterable:     
        """
        Resets the environment to its initial state.

        This method initializes or resets all key attributes of the environment, such as 
        balance, coin holdings, and indices, to prepare for a new episode.

        Parameters:
        -----------
            seed (int, optional): A seed for reproducibility. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. 
                Defaults to None.

        Returns:
        --------
            typing.Iterable: A tuple containing:
                - initial_state (np.ndarray): The initial state of the environment.
                - info (dict): Additional information (empty dictionary in this case).
        """

        self.index=self.window_size

        self.buy_price=self.df.iloc[self.index,0]
        self.sell_price=self.df.iloc[self.index,0]

        self.total=self.initial_balance

        self.usd=self.initial_balance
        self.coin=0

        return self.processed_df.iloc[self.index-self.window_size:self.index].values,{}

if __name__=='__main__':
    pass
